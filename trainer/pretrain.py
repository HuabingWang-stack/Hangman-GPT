"""
trainer/pretrain.py
字符级 GPT 预训练 - 适配 EnhancedCharacterTokenizer
每个样本是一行单词；目标是下一字符预测。
集成wandb进行实验管理和跟踪，支持命令行参数配置
重构版本：解耦模型配置与训练配置，支持灵活的模型加载
增强Early Stopping输出
支持随机数据分割
"""
from __future__ import annotations
import argparse, os, math, json, time, pathlib, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb

# 🎯 使用我们的 EnhancedCharacterTokenizer
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from tokenizer import CharacterTokenizer  # 这是 EnhancedCharacterTokenizer 的别名


# ---------------- 数据分割函数 ----------------
def split_data_randomly(data_file, train_ratio, val_ratio, output_dir, random_seed=42):
    """随机分割数据"""
    print(f"🔀 Randomly splitting data: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)
    print(f"   Total words: {total_lines:,}")

    random.seed(random_seed)
    random.shuffle(lines)

    train_size = int(total_lines * train_ratio)
    val_size = int(total_lines * val_ratio)

    # 确保不超过总数据量
    if train_size + val_size > total_lines:
        val_size = total_lines - train_size

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]

    print(f"   Train words: {len(train_lines):,} ({len(train_lines) / total_lines:.1%})")
    print(f"   Val words: {len(val_lines):,} ({len(val_lines) / total_lines:.1%})")
    print(f"   Unused words: {total_lines - train_size - val_size:,}")

    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, 'train_split.txt')
    val_file = os.path.join(output_dir, 'val_split.txt')

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))

    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    print(f"✅ Split data saved: {train_file}, {val_file}")
    return train_file, val_file


# ---------------- 配置类 ----------------
class ModelConfig:
    """模型架构配置 - 独立于训练配置"""

    def __init__(self,
                 d_model: int = 256,
                 n_layer: int = 8,
                 n_head: int = 8,
                 dropout: float = 0.15,
                 max_seq_len: int = 32):
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.max_seq_len = max_seq_len

    def to_dict(self):
        return {
            'd_model': self.d_model,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            d_model=config_dict.get('d_model', 256),
            n_layer=config_dict.get('n_layer', 8),
            n_head=config_dict.get('n_head', 8),
            dropout=config_dict.get('dropout', 0.15),
            max_seq_len=config_dict.get('max_seq_len', 32)
        )


class TrainingConfig:
    """训练配置 - 用于训练过程的参数"""
    # 基础配置 - 默认值，可通过命令行参数覆盖
    batch_size = 512
    lr = 1e-3
    weight_decay = 0.1
    warmup = 300
    decay_epochs = 30
    min_lr = 1e-5
    epochs = 50
    patience = 8
    grad_clip = 1.0

    # 🎯 数据分割配置
    train_ratio = 0.8
    val_ratio = 0.1
    random_seed = 42

    # wandb配置
    project_name = "enhanced-char-gpt-pretrain"
    experiment_name = None  # 如果None则自动生成
    use_wandb = True
    wandb_entity = None  # 你的wandb用户名或团队名
    tags = []  # 实验标签
    notes = ""  # 实验笔记

    # 监控配置
    log_freq = 10  # 每N个batch记录一次
    save_freq = 5  # 每N个epoch保存样本
    watch_model = True  # 监控模型梯度和参数
    eval_every = 5  # 每n个epoch做一次详细评估
    save_samples = True  # 保存生成样本到wandb
    samples_per_epoch = 5  # 🎯 每个epoch生成的样本数


# 保持向后兼容性的Config类
Config = TrainingConfig


# ---------------- 数据 ----------------
class WordDataset(Dataset):
    def __init__(self, path: str, tokenizer: CharacterTokenizer, max_len: int):
        self.tok = tokenizer
        self.words = [w.strip() for w in pathlib.Path(path).read_text().splitlines() if w.strip()]
        self.max_len = max_len

        # 🎯 使用 EnhancedCharacterTokenizer 的接口
        self.bos_id = self.tok.bos_token_id
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id
        self.unk_id = self.tok.unk_token_id

        print(f"📊 WordDataset initialized:")
        print(f"   Words: {len(self.words):,}")
        print(f"   Max length: {max_len}")
        print(f"   Special tokens: bos={self.bos_id}, eos={self.eos_id}, pad={self.pad_id}, unk={self.unk_id}")

        # 验证一些样本
        self._verify_samples()

    def _verify_samples(self):
        """验证样本编码是否正确"""
        print("🔍 Verifying samples...")
        sample_size = min(3, len(self.words))

        for i in range(sample_size):
            word = self.words[i]
            encoded = self.tok.encode(word, add_special_tokens=True)
            decoded = self.tok.decode(encoded, skip_special_tokens=True)

            print(f"   Sample {i + 1}: '{word}' → {encoded} → '{decoded}'")

            if word.lower() != decoded.lower():
                print(f"   ⚠️  Encoding mismatch!")

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx].lower()

        # 🎯 使用 EnhancedCharacterTokenizer 编码
        try:
            # 直接编码单词（包含 bos/eos）
            ids = self.tok.encode(word, add_special_tokens=True)
            ids = ids[:self.max_len]  # 截断

            # 准备输入和目标序列
            x = ids[:-1] + [self.pad_id] * (self.max_len - len(ids) + 1)
            y = ids[1:] + [self.pad_id] * (self.max_len - len(ids) + 1)
            attn_mask = [1] * (len(ids) - 1) + [0] * (self.max_len - len(ids) + 1)

            # 确保长度正确
            x = x[:self.max_len]
            y = y[:self.max_len]
            attn_mask = attn_mask[:self.max_len]

            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(attn_mask,
                                                                                                      dtype=torch.long)

        except Exception as e:
            print(f"❌ Error encoding word '{word}': {e}")
            # 返回默认的空序列
            x = [self.bos_id, self.eos_id] + [self.pad_id] * (self.max_len - 2)
            y = [self.eos_id] + [self.pad_id] * (self.max_len - 1)
            attn_mask = [1] + [0] * (self.max_len - 1)

            return torch.tensor(x[:self.max_len], dtype=torch.long), \
                torch.tensor(y[:self.max_len], dtype=torch.long), \
                torch.tensor(attn_mask[:self.max_len], dtype=torch.long)


def make_loader(path, tok, max_len, batch, shuffle=False, num_workers=4):
    """创建数据加载器"""
    ds = WordDataset(path, tok, max_len)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True,
                      num_workers=num_workers, pin_memory=True)


# ---------------- 模型 ----------------
class GPT(nn.Module):
    def __init__(self, vocab_size: int, model_config: ModelConfig = None):
        super().__init__()
        # 向后兼容：如果没有提供model_config，使用全局Config
        if model_config is None:
            model_config = ModelConfig(
                d_model=getattr(Config, 'd_model', 256),
                n_layer=getattr(Config, 'n_layer', 8),
                n_head=getattr(Config, 'n_head', 8),
                dropout=getattr(Config, 'dropout', 0.15),
                max_seq_len=getattr(Config, 'max_seq_len', 64)
            )

        self.config = model_config
        self.max_seq_len = model_config.max_seq_len
        self.vocab_size = vocab_size

        print(f"🧠 Creating GPT model:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Model config: {model_config.to_dict()}")

        self.tok_emb = nn.Embedding(vocab_size, model_config.d_model)
        self.pos_emb = nn.Embedding(model_config.max_seq_len, model_config.d_model)
        self.drop = nn.Dropout(model_config.dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_config.d_model,
                nhead=model_config.n_head,
                dim_feedforward=model_config.d_model * 4,
                dropout=model_config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-norm
            )
            for _ in range(model_config.n_layer)
        ])

        self.norm = nn.LayerNorm(model_config.d_model)
        self.lm_head = nn.Linear(model_config.d_model, vocab_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(model_config.max_seq_len, model_config.max_seq_len, dtype=torch.bool), diagonal=1)
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x, attn_mask=None):
        B, T = x.shape
        device = x.device

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb(positions)
        h = self.drop(token_embeddings + position_embeddings)

        causal_mask = self.create_causal_mask(T)
        src_key_padding_mask = None
        if attn_mask is not None:
            src_key_padding_mask = (attn_mask == 0)

        for block in self.blocks:
            h = block(h, src_mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


# ---------------- 模型加载函数 ----------------
def load_pretrained_model(checkpoint_path: str, tokenizer_path: str, device: str = None):
    """通用的模型加载函数，自动处理配置"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 从checkpoint恢复配置
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        model_config = ModelConfig.from_dict(saved_config)
    else:
        # 从模型权重推断配置
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 推断配置
        d_model = state_dict['tok_emb.weight'].shape[1]
        vocab_size = state_dict['tok_emb.weight'].shape[0]
        max_seq_len = state_dict['pos_emb.weight'].shape[0]

        # 推断n_layer
        n_layer = 0
        for key in state_dict.keys():
            if key.startswith('blocks.') and '.self_attn.in_proj_weight' in key:
                layer_idx = int(key.split('.')[1])
                n_layer = max(n_layer, layer_idx + 1)

        # 推断n_head (假设标准配置)
        n_head = 8 if d_model >= 256 else 4

        model_config = ModelConfig(
            d_model=d_model,
            max_seq_len=max_seq_len,
            n_head=n_head,
            n_layer=n_layer,
            dropout=0.15  # 默认值
        )

        print(f"⚠️  推断模型配置: {model_config.to_dict()}")

    # 🎯 加载 EnhancedCharacterTokenizer
    tokenizer = CharacterTokenizer(config_path=tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # 创建模型
    model = GPT(vocab_size, model_config)

    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    return model, tokenizer, model_config


# ---------------- 评估和采样函数 ----------------
def generate_samples(model, tokenizer, device, num_samples=5, max_len=20, temperature=0.8, top_p=0.9):
    """🎯 生成样本用于评估 - 优化版本"""
    model.eval()
    samples = []

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for sample_idx in range(num_samples):
            # 从<bos>开始生成
            generated = [bos_id]

            for step in range(max_len):
                # 准备输入
                x = torch.tensor([generated], dtype=torch.long, device=device)
                if x.size(1) >= model.max_seq_len:
                    break

                # 获取logits
                logits = model(x)
                logits = logits[0, -1, :] / temperature  # 最后一个位置的logits

                # Top-p采样
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

                # 采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)

                if next_token == eos_id:
                    break

            # 🎯 使用 EnhancedCharacterTokenizer 解码
            try:
                word = tokenizer.decode(generated, skip_special_tokens=True)
                if word:
                    samples.append(word)
                else:
                    samples.append("[EMPTY]")
            except Exception as e:
                samples.append(f"[ERROR: {str(e)[:20]}]")

    return samples


def evaluate_model_detailed(model, val_loader, tokenizer, device, loss_fn):
    """详细评估模型"""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y, attn in val_loader:
            x, y, attn = x.to(device), y.to(device), attn.to(device)
            logits = model(x, attn_mask=attn)
            loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    val_ppl = math.exp(total_loss / total_samples)

    # 生成更多样本用于详细评估
    samples = generate_samples(model, tokenizer, device, num_samples=10)

    return {
        'val_ppl': val_ppl,
        'val_loss': total_loss / total_samples,
        'samples': samples
    }


def quick_evaluate_with_samples(model, val_loader, tokenizer, device, loss_fn):
    """🎯 快速验证并生成少量样本"""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        # 快速计算验证损失
        for x, y, attn in val_loader:
            x, y, attn = x.to(device), y.to(device), attn.to(device)
            logits = model(x, attn_mask=attn)
            loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    val_ppl = math.exp(total_loss / total_samples)

    # 🎯 每个epoch都生成5个样本
    samples = generate_samples(model, tokenizer, device, num_samples=Config.samples_per_epoch)

    return {
        'val_ppl': val_ppl,
        'val_loss': total_loss / total_samples,
        'samples': samples
    }


# ---------------- 主训练函数 ----------------
def main(args):
    # 记录开始时间
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Enhanced Character GPT Training")
    print(f"📱 Using device: {device}")

    # 🎯 数据分割
    print(f"\n📊 Data Splitting Configuration:")
    print(f"   Source file: {args.data}")
    print(f"   Train ratio: {Config.train_ratio} ({Config.train_ratio:.1%})")
    print(f"   Val ratio: {Config.val_ratio} ({Config.val_ratio:.1%})")
    print(f"   Random seed: {Config.random_seed}")

    # 创建临时目录用于存放分割后的数据
    split_dir = os.path.join(os.path.dirname(args.data), 'splits')

    # 执行数据分割
    train_file, val_file = split_data_randomly(
        data_file=args.data,
        train_ratio=Config.train_ratio,
        val_ratio=Config.val_ratio,
        output_dir=split_dir,
        random_seed=Config.random_seed
    )

    # 从命令行参数创建模型配置
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    )

    # 打印当前配置
    print(f"\n🔧 Model Configuration:")
    print(f"   Layers: {model_config.n_layer}")
    print(f"   Model dim: {model_config.d_model}")
    print(f"   Attention heads: {model_config.n_head}")
    print(f"   Max sequence length: {model_config.max_seq_len}")
    print(f"   Dropout: {model_config.dropout}")

    print(f"\n⚙️  Training Configuration:")
    print(f"   Learning rate: {Config.lr}")
    print(f"   Batch size: {Config.batch_size}")
    print(f"   Epochs: {Config.epochs}")
    print(f"   🎯 Early Stopping Patience: {Config.patience}")
    print(f"   🎯 Samples per epoch: {Config.samples_per_epoch}")

    # 🎯 加载 EnhancedCharacterTokenizer
    print(f"\n🔤 Loading Enhanced Character Tokenizer...")
    tok = CharacterTokenizer(config_path=args.tokenizer)
    vocab_size = tok.get_vocab_size()
    print(f"   Vocab size: {vocab_size}")

    # 加载数据 - 使用分割后的文件
    print(f"\n📚 Loading datasets...")
    train_loader = make_loader(train_file, tok, model_config.max_seq_len, Config.batch_size, shuffle=True)
    val_loader = make_loader(val_file, tok, model_config.max_seq_len, Config.batch_size, shuffle=False)
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")

    # 模型和优化器
    print(f"\n🧠 Creating model...")
    model = GPT(vocab_size, model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # 测试前向传播
    print(f"\n🧪 Testing model...")
    test_batch = next(iter(train_loader))
    test_x, test_y, test_attn = test_batch
    test_x = test_x[:4].to(device)  # 只用4个样本测试
    test_attn = test_attn[:4].to(device)

    with torch.no_grad():
        test_logits = model(test_x, attn_mask=test_attn)
        print(f"   Test output shape: {test_logits.shape}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")

    # 初始化wandb
    if Config.use_wandb:
        print(f"\n📊 Initializing wandb...")
        try:
            wandb_api = wandb.Api()
            print(f"   ✅ Logged in as: {wandb_api.viewer.username}")
        except Exception as e:
            print(f"   ❌ wandb login failed: {e}")
            print("   Please run: wandb login")
            return

        run_name = Config.experiment_name or f"enhanced-char-gpt-{int(time.time())}"

        # 准备标签
        tags = Config.tags.copy()
        if args.tags:
            tags.extend(args.tags)
        tags.append("enhanced-tokenizer")
        tags.append("samples-per-epoch")
        tags.append("random-split")  # 🎯 新标签

        run = wandb.init(
            project=Config.project_name,
            entity=Config.wandb_entity,
            name=run_name,
            tags=tags,
            notes=Config.notes or args.notes or "",
            config={
                # 模型配置
                **model_config.to_dict(),

                # 训练配置
                'batch_size': Config.batch_size,
                'lr': Config.lr,
                'weight_decay': Config.weight_decay,
                'warmup': Config.warmup,
                'decay_epochs': Config.decay_epochs,
                'min_lr': Config.min_lr,
                'epochs': Config.epochs,
                'patience': Config.patience,
                'grad_clip': Config.grad_clip,
                'samples_per_epoch': Config.samples_per_epoch,

                # 🎯 数据分割配置
                'data_file': args.data,
                'train_ratio': Config.train_ratio,
                'val_ratio': Config.val_ratio,
                'random_seed': Config.random_seed,
                'train_file': train_file,
                'val_file': val_file,

                # 数据配置
                'tokenizer_file': args.tokenizer,
                'vocab_size': vocab_size,
                'param_count': param_count,

                # tokenizer配置
                'tokenizer_type': 'EnhancedCharacterTokenizer',
                'special_tokens': {
                    'pad': tok.pad_token_id,
                    'bos': tok.bos_token_id,
                    'eos': tok.eos_token_id,
                    'unk': tok.unk_token_id,
                    'sep': tok.sep_token_id,
                },

                # 系统配置
                'device': str(device),
                'pytorch_version': torch.__version__,
            }
        )

        # 监控模型（可选，会占用一些资源）
        if Config.watch_model:
            wandb.watch(model, log="gradients", log_freq=100)

        wandb.log({"model_parameters": param_count})

    # 优化器和调度器
    opt = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

    # 修复学习率调度 - 确保不会除零
    def get_lr_schedule(step):
        total_steps = Config.epochs * len(train_loader)
        warmup_steps = Config.warmup
        decay_start_step = Config.decay_epochs * len(train_loader)

        if step < warmup_steps:
            return step / warmup_steps
        elif step < decay_start_step:
            return 1.0
        else:
            # 修复除零错误
            remaining_steps = total_steps - decay_start_step
            if remaining_steps <= 0:
                return Config.min_lr / Config.lr

            progress = (step - decay_start_step) / remaining_steps
            progress = min(progress, 1.0)  # 确保progress不超过1

            return Config.min_lr / Config.lr + 0.5 * (1 - Config.min_lr / Config.lr) * (
                    1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_schedule)

    # 🎯 使用tokenizer的pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    # 🎯 Early Stopping 初始化
    print(f"\n🛑 Early Stopping Configuration:")
    print(f"   Patience: {Config.patience} epochs")
    print(f"   Monitoring: Validation Perplexity (lower is better)")
    print(f"   Will save best model automatically")

    # 训练循环
    print(f"\n🔥 Starting training...")
    best_ppl, bad_epochs = 1e9, 0
    best_epoch = 0
    ppl_history = []  # 🎯 追踪PPL历史

    for ep in range(1, Config.epochs + 1):
        epoch_start_time = time.time()

        # 🎯 紧凑的epoch标题
        print(f"\n{'=' * 60}")
        print(f"🔥 EPOCH {ep}/{Config.epochs}")
        print(f"{'=' * 60}")

        # 训练
        model.train()
        train_loss = 0
        train_samples = 0

        for batch_idx, (x, y, attn) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)

            logits = model(x, attn_mask=attn)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
            opt.step()
            scheduler.step()
            opt.zero_grad()

            train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)

            # 实时日志记录
            if Config.use_wandb and (batch_idx + 1) % Config.log_freq == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_lr': scheduler.get_last_lr()[0],
                    'batch_step': ep * len(train_loader) + batch_idx,
                })

            # 🎯 紧凑的batch进度显示
            if (batch_idx + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_memory = torch.cuda.memory_allocated() / 1024 ** 3
                throughput = (batch_idx + 1) * Config.batch_size / (time.time() - epoch_start_time)

                print(f"   Batch {batch_idx + 1:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Mem: {current_memory:.1f}GB | "
                      f"Speed: {throughput:.0f} samples/s")

        train_ppl = math.exp(train_loss / train_samples)
        epoch_time = time.time() - epoch_start_time

        # 🎯 验证 - 每个epoch都生成样本
        if ep % Config.eval_every == 0 or ep == 1:
            # 详细评估
            eval_results = evaluate_model_detailed(model, val_loader, tok, device, loss_fn)
            val_ppl = eval_results['val_ppl']
            samples = eval_results['samples']
            eval_type = "详细"
        else:
            # 🎯 快速验证但仍生成样本
            eval_results = quick_evaluate_with_samples(model, val_loader, tok, device, loss_fn)
            val_ppl = eval_results['val_ppl']
            samples = eval_results['samples']
            eval_type = "快速"

        # 🎯 追踪历史
        ppl_history.append(val_ppl)

        # 记录指标
        elapsed_time = time.time() - start_time

        # 🎯 分析样本质量
        valid_samples = [s for s in samples if len(s) > 0 and not s.startswith("[ERROR") and s != "[EMPTY]"]
        empty_samples = len([s for s in samples if s == "[EMPTY]"])
        error_samples = len([s for s in samples if s.startswith("[ERROR")])

        sample_stats = {
            'total': len(samples),
            'valid': len(valid_samples),
            'empty': empty_samples,
            'error': error_samples,
            'valid_rate': len(valid_samples) / len(samples) if samples else 0
        }

        # 🎯 Early Stopping 逻辑与详细输出
        is_best = val_ppl < best_ppl
        if is_best:
            improvement = val_ppl - best_ppl  # 负数表示改进
            best_ppl = val_ppl
            best_epoch = ep
            bad_epochs = 0
            status_icon = "💾"
            status_text = f"BEST! (⬇{abs(improvement):.3f})"
        else:
            bad_epochs += 1
            status_icon = "⏳"
            status_text = f"No improvement ({bad_epochs}/{Config.patience})"

        # 🎯 计算改进趋势
        if len(ppl_history) >= 3:
            recent_trend = ppl_history[-1] - ppl_history[-3]
            trend_arrow = "📈" if recent_trend > 0 else "📉"
            trend_text = f"{trend_arrow} {recent_trend:+.3f}"
        else:
            trend_text = "🔄 初始"

        # 🎯 wandb日志记录 - 每个epoch都记录样本
        if Config.use_wandb:
            log_dict = {
                'epoch': ep,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time,
                'elapsed_time': elapsed_time,
                'best_ppl': best_ppl,
                'patience_counter': bad_epochs,
                'patience_remaining': Config.patience - bad_epochs,
                'best_epoch': best_epoch,
                'is_best': is_best,

                # 🎯 样本统计
                'sample_stats/total': sample_stats['total'],
                'sample_stats/valid': sample_stats['valid'],
                'sample_stats/empty': sample_stats['empty'],
                'sample_stats/error': sample_stats['error'],
                'sample_stats/valid_rate': sample_stats['valid_rate'],
            }

            # 🎯 每个epoch都添加样本表格
            if samples and Config.save_samples:
                sample_table = wandb.Table(columns=["epoch", "sample", "length", "valid", "type"])
                for i, sample in enumerate(samples):
                    is_valid = sample in valid_samples
                    sample_type = "valid" if is_valid else ("empty" if sample == "[EMPTY]" else "error")
                    sample_table.add_data(ep, sample, len(sample), is_valid, sample_type)
                log_dict[f"samples"] = sample_table

                # 也记录样本列表（用于搜索）
                log_dict['sample_list'] = samples

            wandb.log(log_dict)

        # 🎯 紧凑的一行结果显示 + Early Stopping信息
        samples_str = " | ".join([f"'{s}'" for s in valid_samples[:3]])  # 只显示前3个有效样本
        if len(valid_samples) > 3:
            samples_str += f" | +{len(valid_samples) - 3}more"
        if not samples_str:
            samples_str = f"无有效样本({sample_stats['empty']}空, {sample_stats['error']}错)"

        print(f"📊 Train PPL: {train_ppl:.3f} | Val PPL: {val_ppl:.3f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s | {eval_type}评估")
        print(f"💭 Samples: {samples_str}")

        # 🎯 Early Stopping 详细状态
        print(f"🛑 Early Stop: {status_icon} {status_text} | Best: {best_ppl:.3f} (Epoch {best_epoch}) | "
              f"Trend: {trend_text} | Remaining: {Config.patience - bad_epochs}")

        # Early stopping - 保存最佳模型
        if is_best:
            # 保存完整的配置和训练信息
            config_dict = {
                **model_config.to_dict(),
                'batch_size': Config.batch_size,
                'lr': Config.lr,
                'warmup': Config.warmup,
                'decay_epochs': Config.decay_epochs,
                'min_lr': Config.min_lr,
                'epochs': Config.epochs,
                'patience': Config.patience,
                'grad_clip': Config.grad_clip,
                'weight_decay': Config.weight_decay
            }

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config_dict,
                'model_config': model_config.to_dict(),  # 专门的模型配置
                'vocab_size': vocab_size,
                'tokenizer_config': {
                    'type': 'EnhancedCharacterTokenizer',
                    'vocab_size': vocab_size,
                    'special_tokens': {
                        'pad_token_id': tok.pad_token_id,
                        'bos_token_id': tok.bos_token_id,
                        'eos_token_id': tok.eos_token_id,
                        'unk_token_id': tok.unk_token_id,
                        'sep_token_id': tok.sep_token_id,
                    }
                },
                # 🎯 保存数据分割信息
                'data_split_info': {
                    'source_file': args.data,
                    'train_file': train_file,
                    'val_file': val_file,
                    'train_ratio': Config.train_ratio,
                    'val_ratio': Config.val_ratio,
                    'random_seed': Config.random_seed,
                },
                'epoch': ep,
                'val_ppl': val_ppl,
                'train_ppl': train_ppl,  # 🎯 也保存训练PPL
                'train_time': elapsed_time,
                'param_count': param_count,
                'best_samples': samples,  # 🎯 保存当前最佳样本
                'sample_stats': sample_stats,  # 🎯 保存样本统计
                'best_epoch': best_epoch,
                'ppl_history': ppl_history,  # 🎯 保存PPL历史
            }, args.out)

        # 🎯 Early stopping 触发
        if bad_epochs >= Config.patience:
            print(f"\n🛑 EARLY STOPPING TRIGGERED!")
            print(f"   🔴 No improvement for {Config.patience} consecutive epochs")
            print(f"   💾 Best model saved at epoch {best_epoch} with PPL {best_ppl:.3f}")
            print(f"   📊 PPL History (last 5): {ppl_history[-5:]}")
            break

    # 🎯 训练完成总结
    total_time = time.time() - start_time
    final_status = "Early Stopped" if bad_epochs >= Config.patience else "Completed"

    print(f"\n🏆 TRAINING {final_status.upper()}!")
    print(f"   🎯 Best validation perplexity: {best_ppl:.3f} (Epoch {best_epoch})")
    print(f"   📊 Final validation perplexity: {val_ppl:.3f} (Epoch {ep})")
    print(f"   ⏱️  Total training time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print(f"   💾 Model saved to: {args.out}")
    print(f"   🛑 Early stopping: Used {bad_epochs}/{Config.patience} patience")
    print(f"   📁 Split files: {train_file}, {val_file}")

    # 🎯 显示PPL改进历史
    if len(ppl_history) > 1:
        total_improvement = ppl_history[0] - best_ppl
        print(f"   📈 Total PPL improvement: {total_improvement:.3f} ({ppl_history[0]:.3f} → {best_ppl:.3f})")

    if Config.use_wandb:
        # 创建最终汇总
        wandb.run.summary.update({
            'final_best_ppl': best_ppl,
            'final_val_ppl': val_ppl,
            'best_epoch': best_epoch,
            'total_training_time': total_time,
            'total_epochs': ep,
            'final_epoch': ep,
            'training_completed': final_status == "Completed",
            'early_stopped': final_status == "Early Stopped",
            'early_stop_patience_used': bad_epochs,
            'early_stop_patience_total': Config.patience,
            'model_size_mb': param_count * 4 / (1024 * 1024),
            'tokenizer_type': 'EnhancedCharacterTokenizer',
            'samples_per_epoch': Config.samples_per_epoch,
            'ppl_improvement': ppl_history[0] - best_ppl if len(ppl_history) > 1 else 0,
            # 🎯 数据分割信息
            'data_split_seed': Config.random_seed,
            'train_ratio': Config.train_ratio,
            'val_ratio': Config.val_ratio,
        })

        # 自动注册最佳模型（如果效果足够好）
        if best_ppl < 10.0:  # 只有足够好的模型才注册
            model_artifact = wandb.Artifact(
                name=f"model-{run.name}",
                type="model",
                description=f"Enhanced Character GPT with PPL {best_ppl:.3f} ({final_status})",
                metadata={
                    "val_ppl": best_ppl,
                    "best_epoch": best_epoch,
                    "final_epoch": ep,
                    "param_count": param_count,
                    "architecture": "transformer",
                    "tokenizer": "EnhancedCharacterTokenizer",
                    "samples_per_epoch": Config.samples_per_epoch,
                    "early_stopped": final_status == "Early Stopped",
                    "random_split": True,
                    "train_ratio": Config.train_ratio,
                    "val_ratio": Config.val_ratio,
                    **model_config.to_dict(),
                }
            )
            model_artifact.add_file(args.out)
            wandb.log_artifact(model_artifact, aliases=["latest", "best"])
            print(f"✅ Model registered to wandb as model-{run.name}")

        wandb.finish()

    # 🎯 清理临时文件（可选）
    if not args.keep_splits:
        try:
            os.remove(train_file)
            os.remove(val_file)
            os.rmdir(split_dir)
            print(f"🧹 Cleaned up split files")
        except OSError:
            print(f"⚠️  Could not clean up split files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Enhanced Character GPT Pretraining with Random Data Splitting")

    # 🎯 数据参数 - 改为单个数据文件
    p.add_argument("--data", default="dataset/180k_10k_10k_50k/pretrain.txt", help="Source data file for splitting")
    p.add_argument("--tokenizer", default="scripts/enhanced_tokenizer/tokenizer.json",
                   help="Enhanced tokenizer config file")
    p.add_argument("--out", default="checkpoints/enhanced_char_gpt_pretrain.pth", help="Output model file")

    # 🎯 数据分割参数
    p.add_argument("--train-ratio", type=float, default=0.95, help="Training data ratio (default: 0.8)")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Validation data ratio (default: 0.1)")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed for data splitting")
    p.add_argument("--keep-splits", action="store_true", help="Keep split files after training")

    # 🔴 模型架构参数
    p.add_argument("--n-layer", type=int, default=8, help="Number of transformer layers")
    p.add_argument("--d-model", type=int, default=256, help="Model dimension")
    p.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    p.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    p.add_argument("--max-seq-len", type=int, default=64, help="Maximum sequence length")

    # 🔴 训练参数
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    p.add_argument("--warmup", type=int, default=300, help="Warmup steps")
    p.add_argument("--decay-epochs", type=int, default=30, help="Epochs before LR decay")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")

    # wandb参数
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--experiment-name", type=str, help="Custom experiment name")
    p.add_argument("--tags", nargs="+", help="Experiment tags")
    p.add_argument("--notes", help="Experiment notes")
    p.add_argument("--wandb-entity", help="Wandb entity/team name")

    args = p.parse_args()

    # 🔴 应用命令行参数到Config类
    Config.batch_size = args.batch_size
    Config.lr = args.lr
    Config.epochs = args.epochs
    Config.weight_decay = args.weight_decay
    Config.grad_clip = args.grad_clip
    Config.patience = args.patience
    Config.warmup = args.warmup
    Config.decay_epochs = args.decay_epochs
    Config.min_lr = args.min_lr

    # 🎯 数据分割配置
    Config.train_ratio = args.train_ratio
    Config.val_ratio = args.val_ratio
    Config.random_seed = args.random_seed

    # wandb配置
    if args.no_wandb:
        Config.use_wandb = False
    if args.experiment_name:
        Config.experiment_name = args.experiment_name
    if args.wandb_entity:
        Config.wandb_entity = args.wandb_entity
    if args.tags:
        Config.tags = args.tags
    if args.notes:
        Config.notes = args.notes

    main(args)