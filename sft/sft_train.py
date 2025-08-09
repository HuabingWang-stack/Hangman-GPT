"""
sft/sft_train.py - Pure Soft Targets版本
使用TRL进行SFT训练 - Wordle单字符预测任务
从checkpoints/pretrain.pth加载预训练模型，进行监督微调
🎯 只使用soft_targets分布计算loss，完全摒弃hard targets
支持KL散度和交叉熵两种loss类型
输出限制为单个a-z字符
每个epoch结束时sample validation数据并记录到wandb
🛑 支持early stopping防止过拟合
"""

import os
import sys
import json
import time
import argparse
import math
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# TRL imports
from trl import SFTTrainer, SFTConfig
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from datasets import Dataset as HFDataset
import wandb

# 我们的模块
from scripts.tokenizer import EnhancedCharacterTokenizer
from trainer.pretrain import GPT, ModelConfig, load_pretrained_model


@dataclass
class SFTTrainingConfig:
    """SFT训练配置 - Pure Soft Targets"""

    # 基础路径
    pretrain_model_path: str = "checkpoints/pretrain.pth"
    tokenizer_path: str = "scripts/enhanced_tokenizer/tokenizer.json"
    train_data_path: str = "sft/data/train.jsonl"
    test_data_path: str = "sft/data/test.jsonl"
    output_dir: str = "sft/models"

    # 训练参数 - RTX 4090优化版本
    learning_rate: float = 2e-5
    batch_size: int = 1024  # RTX 4090可以支持更大batch
    eval_batch_size: int = 64  # evaluation时可以更大
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 64
    gradient_accumulation_steps: int = 2  # 实际batch_size = 32*2 = 64
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000

    # SFT特定参数
    max_prompt_length: int = 48

    # 🎯 Pure Soft targets loss配置
    loss_type: str = "kl_divergence"  # "kl_divergence" or "cross_entropy"
    temperature_scaling: float = 1.0  # 温度缩放，用于调整输出分布的锐度
    label_smoothing: float = 0.0  # 标签平滑（仅用于cross_entropy）

    # 🛑 Early Stopping配置
    early_stopping_enabled: bool = True  # 是否启用early stopping
    early_stopping_patience: int = 5  # 连续多少次评估没有改善就停止
    early_stopping_threshold: float = 0.0001  # 改善的最小阈值
    early_stopping_metric: str = "eval_char_accuracy"  # 监控的指标
    early_stopping_greater_is_better: bool = True  # 指标越大越好还是越小越好

    # 验证和测试
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_char_accuracy"
    greater_is_better: bool = True

    # wandb配置
    use_wandb: bool = True
    project_name: str = "wordle-sft-pure-soft-targets"
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["sft", "wordle", "pure-soft-targets", "enhanced-tokenizer"])

    # 采样配置
    samples_per_epoch: int = 5  # 每个epoch采样5个验证样本

    # 其他
    seed: int = 42
    dataloader_num_workers: int = 8  # RTX 4090可以支持更多worker
    fp16: bool = True  # 使用混合精度训练
    gradient_checkpointing: bool = False  # 🔧 关闭gradient_checkpointing避免兼容性问题
    remove_unused_columns: bool = False


class EarlyStoppingCallback(TrainerCallback):
    """🛑 Early Stopping回调 - 防止过拟合"""

    def __init__(self, config: SFTTrainingConfig):
        self.config = config
        self.patience = config.early_stopping_patience
        self.threshold = config.early_stopping_threshold
        self.metric = config.early_stopping_metric
        self.greater_is_better = config.early_stopping_greater_is_better

        # 状态追踪
        self.best_metric = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.should_stop = False

        # 历史记录
        self.metric_history = []

        print(f"🛑 Early Stopping initialized:")
        print(f"   Metric: {self.metric}")
        print(f"   Patience: {self.patience}")
        print(f"   Threshold: {self.threshold}")
        print(f"   Greater is better: {self.greater_is_better}")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """在每次评估后检查是否需要早停"""
        if not self.config.early_stopping_enabled:
            return control

        # 🔧 修复：检查logs是否为None
        if logs is None:
            print(f"⚠️  Warning: Early stopping received None logs, skipping this evaluation")
            print(f"   Step: {state.global_step}, Epoch: {state.epoch:.2f}")
            return control

        print(f"🔍 Early stopping debug info:")
        print(f"   Step: {state.global_step}, Epoch: {state.epoch:.2f}")
        print(f"   Available metrics: {list(logs.keys())}")
        print(f"   Looking for metric: '{self.metric}'")

        current_metric = logs.get(self.metric)
        if current_metric is None:
            print(f"⚠️  Warning: Early stopping metric '{self.metric}' not found in logs")
            print(f"   Available metrics: {list(logs.keys())}")
            # 🔧 尝试查找相似的metric名称
            similar_metrics = [k for k in logs.keys() if 'char_accuracy' in k or 'accuracy' in k]
            if similar_metrics:
                print(f"   Similar metrics found: {similar_metrics}")
                print(f"   Consider using one of these as early_stopping_metric")
            return control

        print(f"✅ Found metric '{self.metric}' = {current_metric:.6f}")

        # 记录指标历史
        self.metric_history.append({
            'step': state.global_step,
            'epoch': state.epoch,
            'metric': current_metric
        })

        # 初始化最佳指标
        if self.best_metric is None:
            self.best_metric = current_metric
            self.wait_count = 0
            print(f"🎯 Early stopping baseline set: {self.metric}={current_metric:.6f}")
            self._log_to_wandb(state, current_metric, improved=True)
            return control

        # 检查是否有改善
        improved = self._is_improvement(current_metric, self.best_metric)

        if improved:
            self.best_metric = current_metric
            self.wait_count = 0
            print(f"✅ Early stopping metric improved: {self.metric}={current_metric:.6f} (best so far)")
            self._log_to_wandb(state, current_metric, improved=True)
        else:
            self.wait_count += 1
            print(
                f"⏳ Early stopping patience: {self.wait_count}/{self.patience} (no improvement: {self.metric}={current_metric:.6f} vs best {self.best_metric:.6f})")
            self._log_to_wandb(state, current_metric, improved=False)

            if self.wait_count >= self.patience:
                self.should_stop = True
                self.stopped_epoch = state.epoch
                control.should_training_stop = True
                print(f"🛑 Early stopping triggered!")
                print(f"   Stopped at epoch: {self.stopped_epoch:.2f}")
                print(f"   Best {self.metric}: {self.best_metric:.6f}")
                print(f"   Total patience exceeded: {self.wait_count}/{self.patience}")

                # 记录早停信息到wandb
                if self.config.use_wandb:
                    try:
                        wandb.log({
                            'early_stopping_triggered': True,
                            'early_stopping_epoch': self.stopped_epoch,
                            'early_stopping_step': state.global_step,
                            'early_stopping_best_metric': self.best_metric,
                            'early_stopping_patience_used': self.wait_count
                        })
                    except:
                        pass

        return control

    def _is_improvement(self, current_metric, best_metric):
        """判断当前指标是否比最佳指标有改善"""
        if self.greater_is_better:
            return current_metric > best_metric + self.threshold
        else:
            return current_metric < best_metric - self.threshold

    def _log_to_wandb(self, state, current_metric, improved):
        """记录early stopping状态到wandb"""
        if not self.config.use_wandb:
            return

        try:
            wandb.log({
                'early_stopping_metric': current_metric,
                'early_stopping_best_metric': self.best_metric,
                'early_stopping_wait_count': self.wait_count,
                'early_stopping_improved': improved,
                'step': state.global_step
            })
        except:
            pass

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时的总结"""
        if not self.config.early_stopping_enabled:
            return

        print(f"\n🛑 Early Stopping Summary:")

        # 🔧 修复：检查best_metric是否为None
        if self.best_metric is not None:
            print(f"   Final best {self.metric}: {self.best_metric:.6f}")

            if self.should_stop:
                print(f"   Training stopped early at epoch {self.stopped_epoch:.2f}")
                print(f"   Patience used: {self.wait_count}/{self.patience}")
            else:
                print(f"   Training completed normally")
                print(f"   Final patience count: {self.wait_count}/{self.patience}")

            # 显示指标历史趋势
            if len(self.metric_history) > 1:
                recent_metrics = [h['metric'] for h in self.metric_history[-5:]]  # 最近5次
                print(f"   Recent {self.metric} trend: {recent_metrics}")
        else:
            print(f"   ⚠️  No valid metrics received for {self.metric}")
            print(f"   Early stopping was not able to monitor training progress")
            print(f"   Possible reasons:")
            print(f"     - Evaluation was not triggered during training")
            print(f"     - The metric '{self.metric}' was not found in evaluation results")
            print(f"     - Evaluation failed or returned None logs")

        # 最终wandb记录
        if self.config.use_wandb:
            try:
                summary_data = {
                    'early_stopping_final_wait_count': self.wait_count,
                    'early_stopping_stopped_early': self.should_stop,
                    'early_stopping_metric_received': self.best_metric is not None,
                }

                if self.best_metric is not None:
                    summary_data.update({
                        'early_stopping_best_metric': self.best_metric,
                        'early_stopping_stopped_epoch': self.stopped_epoch if self.should_stop else None
                    })

                wandb.run.summary.update(summary_data)
            except:
                pass


class WordleSFTDataset(Dataset):
    """Wordle SFT数据集 - Pure Soft Targets专用"""

    def __init__(self,
                 data_path: str,
                 tokenizer: EnhancedCharacterTokenizer,
                 config: SFTTrainingConfig,
                 is_training: bool = True):

        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training

        # 🎯 预计算a-z字符的token ids映射
        self.a_z_char_to_id = {}
        self.a_z_id_to_char = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c in tokenizer.char_to_id:
                token_id = tokenizer.char_to_id[c]
                self.a_z_char_to_id[c] = token_id
                self.a_z_id_to_char[token_id] = c

        print(f"🔤 Found {len(self.a_z_char_to_id)} a-z characters in tokenizer")

        # 加载数据
        self.raw_data = self._load_jsonl(data_path)
        self.data = self.raw_data.copy()

        # 处理数据
        self.processed_data = self._process_data()

        print(f"📚 {'Train' if is_training else 'Test'} Dataset:")
        print(f"   Raw samples: {len(self.data):,}")
        print(f"   Processed samples: {len(self.processed_data):,}")
        print(f"   Max sequence length: {config.max_seq_length}")
        print(f"   🎯 Loss type: {config.loss_type}")
        print(f"   🎯 Temperature scaling: {config.temperature_scaling}")

        # 只显示少量样本
        if len(self.processed_data) > 0:
            self._show_samples(2)

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Failed to parse line {line_idx + 1}: {e}")
                    continue
        return data

    def get_raw_sample(self, idx: int) -> Dict:
        """获取原始数据样本（用于wandb记录）"""
        return self.raw_data[idx]

    def sample_raw_data(self, num_samples: int = 5) -> List[Dict]:
        """随机采样原始数据"""
        if len(self.raw_data) <= num_samples:
            return self.raw_data.copy()
        return random.sample(self.raw_data, num_samples)

    def _process_data(self) -> List[Dict]:
        """处理数据为SFT格式 - 只保留有soft_targets的样本"""
        processed = []
        no_soft_targets_count = 0
        invalid_soft_targets_count = 0

        for idx, item in enumerate(self.data):
            try:
                # 提取基本信息
                guess_data = item["guess"]
                prompt = guess_data["prompt"]
                completion = guess_data["completion"]
                label = item["label"]
                soft_targets = item.get("soft_targets", {})

                # 🎯 必须有soft_targets才能训练
                if not soft_targets or len(soft_targets) == 0:
                    no_soft_targets_count += 1
                    continue

                # 验证completion是单个字符且在a-z范围内
                if len(completion) != 1:
                    continue

                if completion not in 'abcdefghijklmnopqrstuvwxyz':
                    continue

                # 🎯 验证soft_targets的有效性
                valid_soft_targets = {}
                total_prob = 0.0
                for char, prob in soft_targets.items():
                    if char in 'abcdefghijklmnopqrstuvwxyz' and prob > 0:
                        valid_soft_targets[char] = float(prob)
                        total_prob += float(prob)

                if len(valid_soft_targets) == 0 or total_prob <= 0:
                    invalid_soft_targets_count += 1
                    continue

                # 编码prompt（不包含completion）
                encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)

                # 检查长度
                if len(encoded_prompt) >= self.config.max_seq_length:
                    continue

                # 🎯 转换soft_targets为tensor格式
                soft_targets_tensor = self._convert_soft_targets_to_tensor(valid_soft_targets)

                processed_item = {
                    'input_ids': encoded_prompt,  # 🎯 只包含prompt，不包含completion
                    'prompt_length': len(encoded_prompt),
                    'prompt': prompt,
                    'completion': completion,
                    'label': label,
                    'soft_targets': valid_soft_targets,
                    'soft_targets_tensor': soft_targets_tensor,  # [26]
                    'sample_id': idx,
                    'original_data': item
                }

                processed.append(processed_item)

            except Exception as e:
                print(f"❌ Error processing sample {idx}: {e}")
                continue

        print(f"📊 Data filtering results:")
        print(f"   ❌ No soft targets: {no_soft_targets_count}")
        print(f"   ❌ Invalid soft targets: {invalid_soft_targets_count}")
        print(f"   ✅ Valid samples: {len(processed)}")

        return processed

    def _convert_soft_targets_to_tensor(self, soft_targets: Dict[str, float]) -> torch.Tensor:
        """
        将soft_targets字典转换为tensor格式
        返回长度为26的tensor，对应a-z字符的概率分布
        """
        # 创建26维的零向量
        soft_tensor = torch.zeros(26, dtype=torch.float32)

        # 填充概率值
        total_prob = 0.0
        for char, prob in soft_targets.items():
            if char in 'abcdefghijklmnopqrstuvwxyz':
                char_idx = ord(char) - ord('a')  # a=0, b=1, ..., z=25
                soft_tensor[char_idx] = float(prob)
                total_prob += float(prob)

        # 归一化确保概率和为1
        if total_prob > 0:
            soft_tensor = soft_tensor / total_prob
        else:
            # 如果概率和为0，使用均匀分布作为fallback
            soft_tensor = torch.ones(26, dtype=torch.float32) / 26

        return soft_tensor

    def _show_samples(self, num_samples: int = 2):
        """显示样本"""
        print(f"\n📝 Sample Data:")
        for i in range(min(num_samples, len(self.processed_data))):
            item = self.processed_data[i]
            soft_targets = item['soft_targets']

            print(f"   Sample {i + 1}: '{item['prompt']}' → '{item['completion']}' (label: {item['label']})")

            # 显示前3个最高概率的字符
            sorted_targets = sorted(soft_targets.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"     Top soft targets: {sorted_targets}")
            print(f"     Target distribution sum: {sum(soft_targets.values()):.6f}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # 🎯 只准备prompt，不包含completion
        input_ids = item['input_ids'].copy()
        prompt_length = len(input_ids)

        # Padding
        max_len = self.config.max_seq_length
        if len(input_ids) < max_len:
            padding_length = max_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)

        # 确保长度正确
        input_ids = input_ids[:max_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids],
                                           dtype=torch.long),
            'prompt': item['prompt'],
            'completion': item['completion'],
            'label': item['label'],
            'soft_targets': item['soft_targets'],
            'soft_targets_tensor': item['soft_targets_tensor'],  # [26]
            'prompt_length': prompt_length,
            'sample_id': item['sample_id'],
            'original_data': item['original_data']
        }


class WordleTokenizerWrapper:
    """包装EnhancedCharacterTokenizer以兼容HuggingFace格式"""

    def __init__(self, tokenizer: EnhancedCharacterTokenizer):
        self.tokenizer = tokenizer

        # HuggingFace兼容属性
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        self.unk_token = '<unk>'

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.unk_token_id = tokenizer.unk_token_id

        self.vocab_size = tokenizer.vocab_size

        # 🔧 修复：添加缺失的属性
        self.model_input_names = ["input_ids"]  # 用于group_by_length功能
        self.model_max_length = 512  # 默认最大长度
        self.name_or_path = "wordle_tokenizer"  # tokenizer名称
        self.is_fast = False  # 不是fast tokenizer

        # 添加更多HuggingFace tokenizer需要的属性
        self.deprecation_warnings = {}
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        # 特殊token属性
        self.special_tokens_map = {
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
        }

    def encode(self, text, add_special_tokens=True, **kwargs):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {'input_ids': self.encode(text, **kwargs)}
        elif isinstance(text, list):
            return {'input_ids': [self.encode(t, **kwargs) for t in text]}

    def save_pretrained(self, save_directory, **kwargs):
        """保存tokenizer"""
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        try:
            self.tokenizer.save_config(tokenizer_path)
            print(f"✅ Tokenizer saved to {tokenizer_path}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save tokenizer: {e}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """加载tokenizer"""
        tokenizer_path = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = EnhancedCharacterTokenizer(config_path=tokenizer_path)
            return cls(tokenizer)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    def __len__(self):
        """返回vocab size"""
        return self.vocab_size

    def convert_tokens_to_ids(self, tokens):
        """转换tokens为ids"""
        if isinstance(tokens, str):
            return self.tokenizer.char_to_id.get(tokens, self.unk_token_id)
        return [self.tokenizer.char_to_id.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """转换ids为tokens"""
        if isinstance(ids, int):
            return self.tokenizer.id_to_char.get(ids, self.unk_token)
        return [self.tokenizer.id_to_char.get(id, self.unk_token) for id in ids]


class ValidationSamplingCallback(TrainerCallback):
    """每个epoch结束时采样validation数据并记录到wandb的回调"""

    def __init__(self, eval_dataset, enhanced_tokenizer, config):
        self.eval_dataset = eval_dataset
        self.enhanced_tokenizer = enhanced_tokenizer
        self.config = config

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """在每个epoch结束时执行"""
        if not self.config.use_wandb:
            return

        try:
            print(f"\n📊 Sampling {self.config.samples_per_epoch} validation samples for epoch {state.epoch}...")

            # 随机采样原始数据
            sampled_data = self.eval_dataset.sample_raw_data(self.config.samples_per_epoch)

            # 对每个样本进行预测
            predictions = self._predict_samples(model, sampled_data)

            # 创建wandb表格
            self._log_to_wandb(sampled_data, predictions, state.epoch)

        except Exception as e:
            print(f"❌ Error in validation sampling: {e}")
            import traceback
            traceback.print_exc()

    def on_evaluate(self, args, state, control, logs=None, model=None, **kwargs):
        """🔧 重写on_evaluate以确保在每次评估时都有采样（不仅仅是epoch结束）"""
        # 只在steps策略下进行采样，避免重复
        if (hasattr(args, 'eval_strategy') and args.eval_strategy == "steps" and
                self.config.use_wandb and state.global_step % args.eval_steps == 0):

            try:
                print(
                    f"\n📊 Sampling {self.config.samples_per_epoch} validation samples for step {state.global_step}...")

                # 随机采样原始数据
                sampled_data = self.eval_dataset.sample_raw_data(self.config.samples_per_epoch)

                # 对每个样本进行预测
                if model is not None:
                    predictions = self._predict_samples(model, sampled_data)

                    # 创建wandb表格
                    self._log_to_wandb(sampled_data, predictions, f"step_{state.global_step}")

            except Exception as e:
                print(f"❌ Error in validation sampling on evaluate: {e}")

    def _predict_samples(self, model, sampled_data: List[Dict]) -> List[str]:
        """对采样数据进行预测"""
        if model is None:
            return ['?'] * len(sampled_data)

        model.eval()
        predictions = []

        try:
            model_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            for sample in sampled_data:
                try:
                    # 提取信息
                    guess_data = sample.get("guess", {})
                    prompt = guess_data.get("prompt", "")

                    if not prompt:
                        predictions.append('?')
                        continue

                    # 编码prompt
                    prompt_encoded = self.enhanced_tokenizer.encode(prompt, add_special_tokens=False)

                    if len(prompt_encoded) == 0:
                        predictions.append('?')
                        continue

                    # 准备输入
                    input_ids = torch.tensor([prompt_encoded], dtype=torch.long, device=model_device)

                    # 使用我们GPT模型的参数名
                    logits = model(x=input_ids)

                    # 获取最后一个位置的预测
                    last_position_logits = logits[0, -1, :]

                    # 限制输出到a-z字符
                    a_z_indices = []
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c in self.enhanced_tokenizer.char_to_id:
                            a_z_indices.append(self.enhanced_tokenizer.char_to_id[c])

                    if len(a_z_indices) == 0:
                        predictions.append('?')
                        continue

                    # 创建mask，只允许a-z字符
                    masked_logits = torch.full_like(last_position_logits, float('-inf'))
                    masked_logits[a_z_indices] = last_position_logits[a_z_indices]

                    # 预测
                    pred_char_id = torch.argmax(masked_logits).item()
                    pred_char = self.enhanced_tokenizer.decode([pred_char_id])

                    # 确保预测是单个a-z字符
                    if len(pred_char) == 1 and pred_char in 'abcdefghijklmnopqrstuvwxyz':
                        predictions.append(pred_char)
                    else:
                        predictions.append('?')

                except Exception as e:
                    predictions.append('?')

        return predictions

    def _log_to_wandb(self, sampled_data: List[Dict], predictions: List[str], epoch_or_step):
        """记录到wandb"""
        try:
            if not sampled_data or not predictions:
                return

            # 创建表格 - 包含所有required字段
            table = wandb.Table(columns=[
                "epoch_or_step",
                "prompt",
                "completion",
                "predicted_char",
                "is_correct",
                "label",
                "soft_targets"
            ])

            # 添加数据
            for i, (sample, prediction) in enumerate(zip(sampled_data, predictions)):
                try:
                    guess_data = sample.get("guess", {})
                    prompt = guess_data.get("prompt", "")
                    completion = guess_data.get("completion", "")
                    label = sample.get("label", "")
                    soft_targets = sample.get("soft_targets", {})

                    # 判断预测是否正确
                    is_correct = prediction == completion

                    # 格式化soft_targets为字符串
                    soft_targets_str = json.dumps(soft_targets, separators=(',', ':'))

                    table.add_data(
                        str(epoch_or_step),
                        prompt,
                        completion,
                        prediction,
                        is_correct,
                        label,
                        soft_targets_str
                    )
                except Exception as e:
                    continue

            # 记录到wandb
            wandb.log({
                f"validation_samples_{epoch_or_step}": table,
            })

            # 计算准确率
            correct_predictions = sum(1 for p, s in zip(predictions, sampled_data)
                                      if p == s.get("guess", {}).get("completion", ""))
            accuracy = correct_predictions / len(predictions) if predictions else 0

            # 记录统计信息
            wandb.log({
                f"validation_sample_accuracy_{epoch_or_step}": accuracy,
                f"validation_samples_count_{epoch_or_step}": len(predictions),
                f"validation_correct_count_{epoch_or_step}": correct_predictions,
            })

            print(f"✅ Logged {len(predictions)} validation samples to wandb")
            print(f"   Sample accuracy: {accuracy:.3f} ({correct_predictions}/{len(predictions)})")

            # 显示样本详情
            print(f"   Predictions:")
            for i, (sample, prediction) in enumerate(zip(sampled_data, predictions)):
                try:
                    completion = sample.get("guess", {}).get("completion", "")
                    label = sample.get("label", "")
                    status = "✅" if prediction == completion else "❌"
                    print(f"     {i + 1}. {status} '{completion}' → '{prediction}' (label: '{label}')")
                except Exception as e:
                    print(f"     {i + 1}. ❌ Error displaying sample")

        except Exception as e:
            print(f"❌ Error logging to wandb: {e}")
            import traceback
            traceback.print_exc()


class WordleSFTTrainer(Trainer):
    """🎯 Pure Soft Targets训练器"""

    def __init__(self, *args, sft_config=None, enhanced_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_config = sft_config
        self.enhanced_tokenizer = enhanced_tokenizer
        self.label_names = []  # 🎯 不使用labels

        # 🎯 预计算a-z字符的token indices
        self.a_z_token_indices = []
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c in enhanced_tokenizer.char_to_id:
                self.a_z_token_indices.append(enhanced_tokenizer.char_to_id[c])

        print(f"🎯 Pure soft targets training:")
        print(f"   Loss type: {sft_config.loss_type}")
        print(f"   A-Z token indices: {len(self.a_z_token_indices)}")
        print(f"   Temperature scaling: {sft_config.temperature_scaling}")
        if sft_config.loss_type == "cross_entropy":
            print(f"   Label smoothing: {sft_config.label_smoothing}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """🎯 计算pure soft targets loss"""

        # 提取输入
        input_ids = inputs['input_ids']  # [batch, seq_len]
        attention_mask = inputs.get('attention_mask', None)
        soft_targets_tensor = inputs['soft_targets_tensor']  # [batch, 26]
        prompt_lengths = inputs['prompt_length']  # [batch]

        # 使用我们GPT模型的参数名
        outputs = model(x=input_ids, attn_mask=attention_mask)
        logits = outputs  # [batch, seq_len, vocab_size]

        # 🎯 计算Soft Targets Loss
        soft_loss = self._compute_soft_targets_loss(
            logits, soft_targets_tensor, prompt_lengths
        )

        if soft_loss is None:
            # Fallback: 如果无法计算soft loss，使用小的dummy loss
            soft_loss = torch.tensor(0.01, device=logits.device, requires_grad=True)

        # 记录loss到wandb
        if self.sft_config.use_wandb:
            try:
                wandb.log({
                    'soft_loss': soft_loss.item(),
                    'loss_type': self.sft_config.loss_type
                })
            except:
                pass

        if return_outputs:
            class SimpleOutputs:
                def __init__(self, logits):
                    self.logits = logits

            return soft_loss, SimpleOutputs(logits)

        return soft_loss

    def _compute_soft_targets_loss(self, logits, soft_targets_tensor, prompt_lengths):
        """
        🎯 计算pure soft targets loss
        支持KL散度和交叉熵两种loss类型

        Args:
            logits: [batch, seq_len, vocab_size]
            soft_targets_tensor: [batch, 26] - a-z字符的概率分布
            prompt_lengths: [batch] - 每个样本的prompt长度
        """
        try:
            batch_size = logits.size(0)
            device = logits.device

            # 移动tensors到正确的设备
            if soft_targets_tensor.device != device:
                soft_targets_tensor = soft_targets_tensor.to(device)
            if prompt_lengths.device != device:
                prompt_lengths = prompt_lengths.to(device)

            valid_losses = []

            for i in range(batch_size):
                # 获取completion预测位置的logits
                prompt_len = prompt_lengths[i].item()
                if prompt_len <= 0 or prompt_len > logits.size(1):
                    continue

                # 在prompt最后位置预测下一个字符（completion）
                completion_logits = logits[i, prompt_len - 1, :]  # [vocab_size]

                # 🎯 提取a-z字符的logits
                if len(self.a_z_token_indices) == 0:
                    continue

                a_z_logits = completion_logits[self.a_z_token_indices]  # [26]

                # 应用温度缩放
                if self.sft_config.temperature_scaling != 1.0:
                    a_z_logits = a_z_logits / self.sft_config.temperature_scaling

                # 获取对应的soft targets
                target_dist = soft_targets_tensor[i]  # [26]

                # 确保target distribution是有效的
                if target_dist.sum() <= 0:
                    continue

                # 🎯 根据loss_type计算loss
                if self.sft_config.loss_type == "kl_divergence":
                    # KL散度loss
                    log_probs = F.log_softmax(a_z_logits, dim=-1)
                    sample_loss = F.kl_div(log_probs, target_dist, reduction='sum')

                elif self.sft_config.loss_type == "cross_entropy":
                    # 交叉熵loss
                    if self.sft_config.label_smoothing > 0:
                        # 带标签平滑的交叉熵
                        smoothed_targets = self._apply_label_smoothing(target_dist, self.sft_config.label_smoothing)
                        log_probs = F.log_softmax(a_z_logits, dim=-1)
                        sample_loss = -(smoothed_targets * log_probs).sum()
                    else:
                        # 标准交叉熵
                        log_probs = F.log_softmax(a_z_logits, dim=-1)
                        sample_loss = -(target_dist * log_probs).sum()

                else:
                    raise ValueError(f"Unsupported loss type: {self.sft_config.loss_type}")

                valid_losses.append(sample_loss)

            if len(valid_losses) == 0:
                return None

            # 平均所有valid samples的loss
            avg_soft_loss = torch.stack(valid_losses).mean()

            return avg_soft_loss

        except Exception as e:
            print(f"❌ Error computing soft targets loss: {e}")
            return None

    def _apply_label_smoothing(self, targets, smoothing):
        """应用标签平滑"""
        n_classes = targets.size(-1)
        smoothed = targets * (1 - smoothing) + smoothing / n_classes
        return smoothed

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """预测步骤"""
        # 将输入移动到正确的设备
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')

        model.eval()

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        with torch.no_grad():
            try:
                logits = model(x=input_ids, attn_mask=attention_mask)

                # 计算loss
                soft_loss = self._compute_soft_targets_loss(
                    logits,
                    inputs['soft_targets_tensor'],
                    inputs['prompt_length']
                )

                if soft_loss is None:
                    soft_loss = torch.tensor(0.0, device=model_device)

                if prediction_loss_only:
                    return (soft_loss, None, None)
                else:
                    return (soft_loss, logits, None)

            except Exception as e:
                dummy_loss = torch.tensor(0.0, device=model_device, requires_grad=True)
                dummy_logits = torch.zeros((input_ids.size(0), input_ids.size(1), self.model.vocab_size),
                                           device=model_device)
                return (dummy_loss, dummy_logits, None)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        """评估循环"""

        try:
            # 调用父类的评估循环
            output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                             metric_key_prefix)

            # 确保output.metrics存在
            if not hasattr(output, 'metrics') or output.metrics is None:
                from transformers.trainer_utils import EvalLoopOutput
                output = EvalLoopOutput(
                    predictions=None,
                    label_ids=None,
                    metrics={},
                    num_samples=len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
                )

            # 确保基本的eval_loss存在
            if f'{metric_key_prefix}_loss' not in output.metrics:
                output.metrics[f'{metric_key_prefix}_loss'] = 0.0

            # 添加自定义评估指标
            custom_metrics = self._compute_eval_metrics(dataloader, metric_key_prefix)

            # 合并自定义指标
            if custom_metrics:
                output.metrics.update(custom_metrics)

            return output

        except Exception as e:
            print(f"❌ Error in evaluation_loop: {e}")
            import traceback
            traceback.print_exc()

            # 返回一个基本的结果避免训练中断
            from transformers.trainer_utils import EvalLoopOutput
            basic_metrics = {
                f'{metric_key_prefix}_loss': 0.0,
                f'{metric_key_prefix}_char_accuracy': 0.0,
                f'{metric_key_prefix}_runtime': 0.0,
                f'{metric_key_prefix}_samples_per_second': 0.0,
                f'{metric_key_prefix}_steps_per_second': 0.0,
            }

            return EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics=basic_metrics,
                num_samples=len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
            )

    def _compute_eval_metrics(self, dataloader, metric_key_prefix="eval"):
        """计算评估指标"""
        model = self.model
        model.eval()

        total_samples = 0
        correct_predictions = 0
        total_loss = 0.0
        num_batches = 0

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔍 Computing eval metrics on device: {model_device}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # 限制评估batch数量以节省时间
                    if batch_idx >= 50:
                        break

                    # 移动到设备
                    input_ids = batch['input_ids'].to(model_device)
                    attention_mask = batch['attention_mask'].to(model_device)
                    soft_targets_tensor = batch['soft_targets_tensor'].to(model_device)
                    prompt_lengths = batch['prompt_length'].to(model_device)

                    if input_ids.size(0) == 0:
                        continue

                    # 使用我们GPT模型的参数名
                    logits = model(x=input_ids, attn_mask=attention_mask)

                    # 计算loss
                    batch_loss = self._compute_soft_targets_loss(
                        logits, soft_targets_tensor, prompt_lengths
                    )

                    if batch_loss is not None:
                        total_loss += batch_loss.item()
                        num_batches += 1

                    # 逐样本分析
                    batch_size = input_ids.size(0)
                    for i in range(batch_size):
                        try:
                            # 获取样本信息
                            prompt = batch['prompt'][i]
                            true_completion = batch['completion'][i]
                            prompt_length = prompt_lengths[i].item()

                            if prompt_length > 0 and prompt_length <= logits.size(1):
                                # 预测completion
                                pred_logits = logits[i, prompt_length - 1, :]

                                # 限制输出到a-z字符
                                a_z_indices = [self.enhanced_tokenizer.char_to_id[c] for c in
                                               'abcdefghijklmnopqrstuvwxyz'
                                               if c in self.enhanced_tokenizer.char_to_id]

                                # 创建mask
                                masked_logits = torch.full_like(pred_logits, float('-inf'))
                                if len(a_z_indices) > 0:
                                    masked_logits[a_z_indices] = pred_logits[a_z_indices]

                                pred_char_id = torch.argmax(masked_logits).item()
                                pred_char = self.enhanced_tokenizer.decode([pred_char_id])

                                # 统计
                                total_samples += 1
                                if pred_char == true_completion:
                                    correct_predictions += 1

                        except Exception as e:
                            continue

                except Exception as e:
                    print(f"⚠️  Warning: Error in eval batch {batch_idx}: {e}")
                    continue

        # 计算最终指标
        char_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # 记录指标
        metrics = {
            f'{metric_key_prefix}_char_accuracy': char_accuracy,
            f'{metric_key_prefix}_total_samples': total_samples,
            f'{metric_key_prefix}_correct_predictions': correct_predictions,
            f'{metric_key_prefix}_avg_loss': avg_loss,
        }

        print(f"\n📊 {metric_key_prefix.title()} Custom Metrics:")
        print(f"   Character Accuracy: {char_accuracy:.4f} ({correct_predictions}/{total_samples})")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   📤 Returning metrics: {list(metrics.keys())}")

        return metrics


def create_data_collator(tokenizer):
    """创建数据整理器 - Pure Soft Targets版本"""

    def collate_fn(batch):
        # 提取所有字段
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        # 🎯 Soft targets相关字段
        soft_targets_tensor = torch.stack([item['soft_targets_tensor'] for item in batch])
        prompt_lengths = torch.tensor([item['prompt_length'] for item in batch], dtype=torch.long)

        # 其他字段
        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        labels_list = [item['label'] for item in batch]
        soft_targets = [item['soft_targets'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]
        original_data = [item['original_data'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 🎯 Soft targets字段
            'soft_targets_tensor': soft_targets_tensor,
            'prompt_length': prompt_lengths,
            # 其他字段
            'prompt': prompts,
            'completion': completions,
            'label': labels_list,
            'soft_targets': soft_targets,
            'sample_id': sample_ids,
            'original_data': original_data
        }

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="🎯 Wordle SFT Training with Pure Soft Targets + Early Stopping")

    # 基础参数
    parser.add_argument("--pretrain_model", default="checkpoints/pretrain.pth", help="预训练模型路径")
    parser.add_argument("--tokenizer", default="scripts/enhanced_tokenizer/tokenizer.json", help="Tokenizer配置文件")
    parser.add_argument("--train_data", default="sft/data/train.jsonl", help="训练数据路径")
    parser.add_argument("--test_data", default="sft/data/test.jsonl", help="测试数据路径")
    parser.add_argument("--output_dir", default="sft/models", help="输出目录")

    # 训练参数 - RTX 4090优化
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="评估批大小")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--max_seq_length", type=int, default=64, help="最大序列长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志记录步数")
    parser.add_argument("--eval_steps", type=int, default=200, help="评估步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数")

    # 🎯 Pure Soft targets参数
    parser.add_argument("--loss_type", choices=["kl_divergence", "cross_entropy"], default="kl_divergence",
                        help="Loss类型: kl_divergence (默认) 或 cross_entropy")
    parser.add_argument("--temperature_scaling", type=float, default=1.0, help="输出logits的温度缩放")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="标签平滑系数 (仅用于cross_entropy)")

    # 🛑 Early Stopping参数
    parser.add_argument("--no_early_stopping", action="store_true", help="禁用early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=8,
                        help="Early stopping patience (默认: 8)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0001,
                        help="Early stopping改善阈值 (默认: 0.0001)")
    parser.add_argument("--early_stopping_metric", default="eval_char_accuracy",
                        help="Early stopping监控指标 (默认: eval_char_accuracy)")

    # 快速调试模式
    parser.add_argument("--debug", action="store_true", help="调试模式：小batch，快速evaluation")

    # 采样参数
    parser.add_argument("--samples_per_epoch", type=int, default=5, help="每个epoch采样的验证样本数")

    # 策略参数
    parser.add_argument("--no_load_best", action="store_true", help="禁用load_best_model_at_end")

    # wandb参数
    parser.add_argument("--no_wandb", action="store_true", help="禁用wandb")
    parser.add_argument("--project_name", default="wordle-sft-pure-soft-targets", help="wandb项目名")
    parser.add_argument("--experiment_name", help="实验名称")
    parser.add_argument("--tags", nargs="+", help="wandb标签")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="数据加载器工作进程数")

    args = parser.parse_args()

    # 调试模式参数覆盖
    if args.debug:
        print("🔧 Debug mode enabled: using lightweight parameters")
        args.batch_size = 4
        args.eval_batch_size = 8
        args.num_epochs = 1
        args.eval_steps = 20  # 🔧 更快的evaluation
        args.logging_steps = 5
        args.dataloader_num_workers = 2
        args.samples_per_epoch = 2
        args.early_stopping_patience = 2  # 调试模式下降低patience

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 创建配置
    config = SFTTrainingConfig(
        pretrain_model_path=args.pretrain_model,
        tokenizer_path=args.tokenizer,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        samples_per_epoch=args.samples_per_epoch,
        load_best_model_at_end=not args.no_load_best,
        # 🎯 Pure Soft targets配置
        loss_type=args.loss_type,
        temperature_scaling=args.temperature_scaling,
        label_smoothing=args.label_smoothing,
        # 🛑 Early stopping配置
        early_stopping_enabled=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_greater_is_better=True if args.early_stopping_metric in ["eval_char_accuracy"] else False,
        # wandb配置
        use_wandb=not args.no_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        tags=args.tags or [],
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers
    )

    print("🚀 Starting Wordle SFT Training with Pure Soft Targets + Early Stopping")
    print(f"📱 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"🔧 Configuration: batch_size={config.batch_size}, eval_steps={config.eval_steps}")
    print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   🎯 Loss type: {config.loss_type}")
    print(f"   🎯 Temperature scaling: {config.temperature_scaling}")
    if config.loss_type == "cross_entropy" and config.label_smoothing > 0:
        print(f"   🎯 Label smoothing: {config.label_smoothing}")
    print(f"   🛑 Early stopping: {'enabled' if config.early_stopping_enabled else 'disabled'}")
    if config.early_stopping_enabled:
        print(f"   🛑 Early stopping metric: {config.early_stopping_metric}")
        print(f"   🛑 Early stopping patience: {config.early_stopping_patience}")
    print(f"   Workers: {config.dataloader_num_workers}")

    # 初始化wandb
    if config.use_wandb:
        run_name = config.experiment_name or f"pure-soft-{config.loss_type}-{int(time.time())}"
        tags = config.tags + [config.loss_type]
        if config.early_stopping_enabled:
            tags.append("early-stopping")

        wandb.init(
            project=config.project_name,
            name=run_name,
            tags=tags,
            config=config.__dict__
        )

    # 加载预训练模型和tokenizer
    print(f"\n🔤 Loading tokenizer from {config.tokenizer_path}")
    tokenizer = EnhancedCharacterTokenizer(config_path=config.tokenizer_path)

    print(f"🧠 Loading pretrained model from {config.pretrain_model_path}")
    model, _, model_config = load_pretrained_model(
        config.pretrain_model_path,
        config.tokenizer_path
    )

    # 包装tokenizer
    wrapped_tokenizer = WordleTokenizerWrapper(tokenizer)

    print(f"   Model vocab size: {model.vocab_size}")
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")

    # 加载数据集
    print(f"\n📚 Loading datasets...")
    train_dataset = WordleSFTDataset(config.train_data_path, tokenizer, config, is_training=True)
    test_dataset = WordleSFTDataset(config.test_data_path, tokenizer, config, is_training=False)

    if len(train_dataset) == 0:
        print("❌ Error: No valid training samples found!")
        print("   Make sure your data contains soft_targets field")
        return

    if len(test_dataset) == 0:
        print("❌ Error: No valid test samples found!")
        return

    # 创建数据整理器
    data_collator = create_data_collator(tokenizer)

    # 🛑 创建callbacks
    callbacks = []

    # Validation sampling callback
    validation_callback = ValidationSamplingCallback(test_dataset, tokenizer, config)
    callbacks.append(validation_callback)

    # Early stopping callback
    if config.early_stopping_enabled:
        early_stopping_callback = EarlyStoppingCallback(config)
        callbacks.append(early_stopping_callback)

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.eval_strategy if config.load_best_model_at_end else "steps",
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        report_to="wandb" if config.use_wandb else None,
        run_name=config.experiment_name,
        seed=config.seed,
        logging_first_step=True,
        save_total_limit=3,
        eval_delay=0,
        prediction_loss_only=False,
        # RTX 4090 优化
        dataloader_pin_memory=True,
    )

    # 创建训练器
    trainer = WordleSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=wrapped_tokenizer,
        callbacks=callbacks,  # 🛑 添加callbacks
        sft_config=config,
        enhanced_tokenizer=tokenizer
    )

    trainer.label_names = []  # 🎯 Pure soft targets不需要labels

    print(f"\n🔥 Starting training...")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   🎯 Will evaluate every {config.eval_steps} steps")
    print(f"   🎯 Samples per epoch: {config.samples_per_epoch}")
    print(f"   🎯 Wandb logging: {'enabled' if config.use_wandb else 'disabled'}")
    if config.early_stopping_enabled:
        print(f"   🛑 Early stopping: patience={config.early_stopping_patience}, metric={config.early_stopping_metric}")

    # 开始训练
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # 保存最终模型
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)

    # 手动保存tokenizer配置
    tokenizer_save_path = os.path.join(final_model_path, "tokenizer.json")
    try:
        tokenizer.save_config(tokenizer_save_path)
        print(f"✅ Tokenizer saved to {tokenizer_save_path}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save tokenizer: {e}")

    # 最终评估
    print(f"\n🧪 Final evaluation...")
    eval_results = trainer.evaluate()

    print(f"\n🏆 Training completed!")
    print(f"   Total time: {training_time:.2f}s ({training_time / 60:.1f}min)")
    print(f"   Final eval loss: {eval_results.get('eval_loss', 'N/A')}")
    print(f"   Final char accuracy: {eval_results.get('eval_char_accuracy', 'N/A')}")
    print(f"   Model saved to: {final_model_path}")

    # 🛑 显示early stopping信息
    if config.early_stopping_enabled:
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                if callback.should_stop:
                    print(f"   🛑 Training stopped early due to early stopping")
                else:
                    print(f"   ✅ Training completed without early stopping")
                break

    # wandb总结
    if config.use_wandb:
        summary_data = {
            'final_eval_loss': eval_results.get('eval_loss', 0),
            'final_char_accuracy': eval_results.get('eval_char_accuracy', 0),
            'training_time': training_time,
            'total_epochs': config.num_epochs,
            'model_path': final_model_path,
            'samples_per_epoch': config.samples_per_epoch,
            'effective_batch_size': config.batch_size * config.gradient_accumulation_steps,
            'loss_type': config.loss_type,
            'temperature_scaling': config.temperature_scaling,
            'label_smoothing': config.label_smoothing,
            'early_stopping_enabled': config.early_stopping_enabled
        }

        # 🛑 添加early stopping相关信息
        if config.early_stopping_enabled:
            summary_data.update({
                'early_stopping_patience': config.early_stopping_patience,
                'early_stopping_metric': config.early_stopping_metric,
                'early_stopping_threshold': config.early_stopping_threshold
            })

        wandb.run.summary.update(summary_data)
        wandb.finish()

    return trainer, eval_results


if __name__ == "__main__":
    main()