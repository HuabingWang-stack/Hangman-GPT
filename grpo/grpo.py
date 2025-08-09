# grpo/grpo.py - Clean Step-wise PPO训练主脚本
import os
import math
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import json
import numpy as np
import gc
from config import CleanPPOConfig
from dataset import HangmanDataset
from trajectory_generator import PPOTrajectoryGenerator
from reward_calculator import SimplifiedStepwiseRewardCalculator
from grpo_trainer import CleanStepwisePPOTrainer
from evaluator import StepWiseHangmanEvaluator
from custom_model import load_sft_model_with_wrappers


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024 ** 3
        cached_memory = torch.cuda.memory_reserved(0) / 1024 ** 3
        free_memory = total_memory - cached_memory

        print(f"💾 GPU Memory Status:")
        print(f"   Total: {total_memory:.1f} GB")
        print(f"   Allocated: {allocated_memory:.1f} GB")
        print(f"   Cached: {cached_memory:.1f} GB")
        print(f"   Free: {free_memory:.1f} GB")

        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': free_memory
        }
    return None


def load_models_with_memory_management(sft_model_path, tokenizer_path, device):
    """内存优化的模型加载"""
    print("🔄 Loading models with memory management...")

    clear_gpu_memory()
    initial_memory = check_gpu_memory()

    try:
        print("   Attempting normal model loading...")
        model, tokenizer = load_sft_model_with_wrappers(sft_model_path, tokenizer_path)

        model_params = sum(p.numel() for p in model.parameters())
        model_size_gb = model_params * 4 / 1024 ** 3
        print(f"   Model size: {model_params:,} parameters (~{model_size_gb:.1f} GB)")

        if device.type == 'cuda':
            if initial_memory and initial_memory['free'] < model_size_gb * 2.5:
                print(
                    f"⚠️ Insufficient GPU memory. Available: {initial_memory['free']:.1f} GB, Need: ~{model_size_gb * 2.5:.1f} GB")
                raise RuntimeError("Insufficient GPU memory")

            print("   Loading main model to GPU...")
            model.to(device)
            torch.cuda.empty_cache()

            print("   Loading reference model...")
            reference_model, _ = load_sft_model_with_wrappers(sft_model_path, tokenizer_path)
            reference_model.to(device)
            torch.cuda.empty_cache()

        else:
            print("   Loading reference model for CPU...")
            reference_model, _ = load_sft_model_with_wrappers(sft_model_path, tokenizer_path)

        check_gpu_memory()
        return model, reference_model, tokenizer, device

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"❌ GPU loading failed: {e}")
        print("🔄 Attempting CPU fallback...")

        clear_gpu_memory()
        device = torch.device('cpu')
        print("   Loading models on CPU...")

        model, tokenizer = load_sft_model_with_wrappers(sft_model_path, tokenizer_path)
        reference_model, _ = load_sft_model_with_wrappers(sft_model_path, tokenizer_path)

        model.to(device)
        reference_model.to(device)

        print("✅ Models loaded on CPU successfully")
        return model, reference_model, tokenizer, device


def create_config_from_args(args):
    """从命令行参数创建配置"""
    config = CleanPPOConfig()

    print(f"🎯 Clean Step-wise PPO 配置参数覆盖过程:")

    overridden_params = []

    # 核心PPO参数映射
    param_mapping = {
        'kl_coeff': 'kl_coeff',
        'clip_epsilon': 'clip_epsilon',
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size',
        'num_epochs': 'num_epochs',
        'weight_decay': 'weight_decay',
        'max_grad_norm': 'max_grad_norm',
        'correct_prediction_reward': 'correct_prediction_reward',
        'wrong_prediction_penalty': 'wrong_prediction_penalty',
        'repeat_guess_penalty': 'repeat_guess_penalty',
        'efficiency_bonus_factor': 'efficiency_bonus_factor',
        'protect_threshold': 'protect_threshold',
        'focus_threshold': 'focus_threshold',
        'short_word_length': 'short_word_length',
    }

    # 应用参数覆盖
    for arg_name, config_attr in param_mapping.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            old_value = getattr(config, config_attr, None)
            new_value = getattr(args, arg_name)

            if old_value != new_value:
                setattr(config, config_attr, new_value)
                overridden_params.append(f"   {config_attr}: {old_value} → {new_value}")

    # 特殊参数处理
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_project:
        config.wandb_project = args.wandb_project

    if overridden_params:
        print("📝 以下参数被命令行覆盖:")
        for param in overridden_params:
            print(param)
    else:
        print("📝 所有参数使用默认值")

    # 🎯 根据clean step-wise PPO参数自动生成output_dir
    if not args.output_dir:
        device_suffix = "cpu" if args.force_cpu else "gpu"
        config.output_dir = (
            f"grpo/models/clean_stepwise_ppo_{device_suffix}_b{config.batch_size}_lr{config.learning_rate:.0e}_"
            f"kl{config.kl_coeff:.2f}_clip{config.clip_epsilon:.2f}"
        )
        print(f"🗂️  自动生成输出目录: {config.output_dir}")

    return config


def setup_wandb(config: CleanPPOConfig, device):
    """初始化wandb"""
    device_suffix = "cpu" if device.type == 'cpu' else "gpu"
    run_name = (f"clean_stepwise_ppo_{device_suffix}_b{config.batch_size}_lr{config.learning_rate:.0e}_"
                f"kl{config.kl_coeff:.2f}_clip{config.clip_epsilon:.2f}")

    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={**config.__dict__, "device": device.type},
        tags=["clean_stepwise_ppo", "step_accuracy", device_suffix]
    )


def log_stepwise_trajectory_to_wandb(trajectory_group: dict, game_count: int, category: str):
    """🎯 记录step-wise轨迹到wandb（清理版）"""
    word = trajectory_group["word"]
    trajectories = trajectory_group["trajectories"]

    # 收集step-wise统计
    all_step_accuracies = []
    all_confidences = []

    for traj in trajectories:
        step_accuracies = traj.get("step_accuracies", [])
        all_step_accuracies.extend([1.0 if acc else 0.0 for acc in step_accuracies])

        # 从步骤中提取置信度
        for step in traj.get("steps", []):
            confidence = step.get("prediction_confidence", 0.0)
            all_confidences.append(confidence)

    # 🎯 记录step-wise指标（清理版）
    wandb.log({
        f"clean_stepwise_ppo/{category}_words/step_accuracy": np.mean(
            all_step_accuracies) if all_step_accuracies else 0.0,
        f"clean_stepwise_ppo/{category}_words/avg_confidence": np.mean(all_confidences) if all_confidences else 0.0,
        f"clean_stepwise_ppo/{category}_words/total_steps": len(all_step_accuracies),
        f"clean_stepwise_ppo/{category}_words/correct_steps": sum(all_step_accuracies),
        f"clean_stepwise_ppo/{category}_words/word_length": len(word),
        "game_count": game_count
    })


def calculate_stepwise_category_performance(trajectory_groups, word_categories):
    """🎯 计算各类别词汇的step-wise性能（清理版）"""
    category_performance = {"PROTECT": [], "FOCUS": [], "IMPROVE": []}

    for group in trajectory_groups:
        word = group["word"]
        category = word_categories.get(word, "IMPROVE")

        # 收集step-wise指标
        all_step_accuracies = []

        for traj in group["trajectories"]:
            step_accuracies = traj.get("step_accuracies", [])
            all_step_accuracies.extend([1.0 if acc else 0.0 for acc in step_accuracies])

        if all_step_accuracies:
            step_accuracy = np.mean(all_step_accuracies)
            category_performance[category].append(step_accuracy)

    category_stats = {}
    for category in ["PROTECT", "FOCUS", "IMPROVE"]:
        step_accuracies = category_performance[category]

        if step_accuracies:
            category_stats[f"{category.lower()}_step_accuracy"] = np.mean(step_accuracies)
            category_stats[f"{category.lower()}_word_count"] = len(step_accuracies)
        else:
            category_stats[f"{category.lower()}_step_accuracy"] = 0.0
            category_stats[f"{category.lower()}_word_count"] = 0

    return category_stats


def monitor_stepwise_training_safety(loss_info, epoch, batch_num):
    """🎯 Step-wise训练安全监控（清理版）"""
    safety_alerts = []

    # 检查loss异常
    total_loss = loss_info.get("total_loss", 0.0)
    if isinstance(total_loss, torch.Tensor):
        total_loss = total_loss.item()

    if total_loss > 8.0:
        safety_alerts.append(f"High total loss: {total_loss:.4f}")

    # 检查step accuracy
    step_accuracy = loss_info.get("overall_step_accuracy", 0.0)
    if step_accuracy < 0.1:
        safety_alerts.append(f"Very low step accuracy: {step_accuracy:.3f}")

    # 检查极端loss事件
    extreme_count = loss_info.get("extreme_loss_count", 0)
    if extreme_count > 0:
        safety_alerts.append(f"Extreme loss events: {extreme_count}")

    if safety_alerts:
        print(f"⚠️  Step-wise Safety Alert - Epoch {epoch}, Batch {batch_num}:")
        for alert in safety_alerts:
            print(f"   {alert}")

        # 记录到wandb
        wandb.log({
            "stepwise_safety/alert_count": len(safety_alerts),
            "stepwise_safety/high_loss": total_loss > 8.0,
            "stepwise_safety/low_step_accuracy": step_accuracy < 0.1,
            "stepwise_safety/extreme_loss_events": extreme_count,
            "epoch": epoch,
            "batch": batch_num
        })

    return len(safety_alerts)


def adaptive_batch_size_for_device(device, base_batch_size):
    """根据设备类型自适应调整batch size"""
    if device.type == 'cpu':
        return max(2, base_batch_size // 4)
    else:
        memory_info = check_gpu_memory()
        if memory_info and memory_info['free'] < 4.0:
            return max(2, base_batch_size // 2)
        else:
            return base_batch_size


def main():
    # 🎯 Clean Step-wise PPO命令行参数
    parser = argparse.ArgumentParser(description="Clean Step-wise PPO Training")

    # 🎯 核心PPO参数
    parser.add_argument("--kl_coeff", type=float, default=0.08, help="KL regularization coefficient")
    parser.add_argument("--clip_epsilon", type=float, default=0.15, help="PPO clipping parameter")
    parser.add_argument("--learning_rate", type=float, default=8e-5, help="Learning rate")

    # 奖励计算参数
    parser.add_argument("--correct_prediction_reward", type=float, default=2.0, help="Correct prediction reward")
    parser.add_argument("--wrong_prediction_penalty", type=float, default=-0.5, help="Wrong prediction penalty")
    parser.add_argument("--repeat_guess_penalty", type=float, default=-1.0, help="Repeat guess penalty")
    parser.add_argument("--efficiency_bonus_factor", type=float, default=0.2, help="Efficiency bonus factor")

    # 分层训练参数
    parser.add_argument("--protect_threshold", type=float, default=0.75, help="PROTECT category threshold")
    parser.add_argument("--focus_threshold", type=float, default=0.45, help="FOCUS category threshold")
    parser.add_argument("--short_word_length", type=int, default=5, help="Short word length threshold")

    # PPO基础参数
    parser.add_argument("--k_trajectories", type=int, default=4, help="每个词生成的轨迹数")
    parser.add_argument("--max_steps", type=int, default=10, help="最大步数")

    # 训练基础参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小（trajectory groups）")
    parser.add_argument("--num_epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="梯度裁剪范数")

    # 路径参数
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")
    parser.add_argument("--sft_model_path", type=str, default="sft/models/4xdataset-8layer-final/checkpoint-6000",
                        help="SFT模型路径")
    parser.add_argument("--pretrain_path", type=str, default="dataset/225k_10k_5k_10k/pretrain.txt",
                        help="预训练数据路径")
    parser.add_argument("--sft_path", type=str, default="dataset/225k_10k_5k_10k/sft.txt", help="SFT数据路径")
    parser.add_argument("--grpo_path", type=str, default="dataset/225k_10k_5k_10k/grpo.txt", help="PPO数据路径")

    # 内存管理参数
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU训练")
    parser.add_argument("--max_memory_gb", type=float, default=None, help="最大GPU内存使用限制")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", help="启用梯度检查点")

    # 训练控制参数
    parser.add_argument("--eval_interval", type=int, default=3, help="评估间隔")
    parser.add_argument("--save_interval", type=int, default=5, help="保存间隔")
    parser.add_argument("--log_trajectory_every", type=int, default=20, help="轨迹记录间隔")

    # 学习率调度参数
    parser.add_argument("--use_lr_scheduler", action="store_true", default=True, help="使用学习率调度器")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="调度器耐心值")
    parser.add_argument("--scheduler_factor", type=float, default=0.75, help="调度器衰减因子")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="最小学习率")

    # Early stopping参数
    parser.add_argument("--use_early_stopping", action="store_true", default=True, help="使用早停")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="早停耐心值")

    # Wandb参数
    parser.add_argument("--wandb_project", type=str, default="hangman-clean-stepwise-ppo", help="Wandb项目名")

    # 安全监控参数
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--enable_safety_monitoring", action="store_true", default=True, help="启用安全监控")

    args = parser.parse_args()

    # 设备选择和内存检查
    if args.force_cpu:
        device = torch.device('cpu')
        print("🖥️  Forced CPU mode enabled")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            memory_info = check_gpu_memory()
            if memory_info and memory_info['free'] < 2.0:
                print(f"⚠️ Low GPU memory ({memory_info['free']:.1f} GB), consider using --force_cpu")
        else:
            device = torch.device('cpu')
            print("🖥️  CUDA not available, using CPU")

    # 创建配置
    config = create_config_from_args(args)

    # 根据设备自适应调整batch size
    original_batch_size = config.batch_size
    config.batch_size = adaptive_batch_size_for_device(device, config.batch_size)
    if config.batch_size != original_batch_size:
        print(f"🔧 Adaptive batch size: {original_batch_size} → {config.batch_size}")

    # 保存配置
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump({**config.__dict__, "device": device.type}, f, indent=2)

    print(f"\n🎯 Clean Step-wise PPO Parameters:")
    print(f"   🖥️  Device: {device}")
    print(f"   🔧 KL coefficient: {config.kl_coeff}")
    print(f"   🔧 Clip epsilon: {config.clip_epsilon}")
    print(f"   🔧 Learning rate: {config.learning_rate}")
    print(f"   🔧 Batch size: {config.batch_size} trajectory groups")
    print(f"   🎯 Trajectories per word: {args.k_trajectories}")

    # 初始化wandb
    setup_wandb(config, device)

    # 加载数据
    dataset = HangmanDataset(config.pretrain_path, config.sft_path, config.grpo_path)

    print(f"Using device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 内存优化的模型加载
    print("Loading models with memory optimization...")
    try:
        model, reference_model, tokenizer, actual_device = load_models_with_memory_management(
            config.sft_model_path,
            "scripts/enhanced_tokenizer/tokenizer.json",
            device
        )
        device = actual_device
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    model.train()
    reference_model.eval()

    # 冻结reference model
    for param in reference_model.parameters():
        param.requires_grad = False

    print(
        f"✅ Models loaded on {device}. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 梯度检查点
    if args.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")

    # 🎯 初始化Clean Step-wise PPO组件
    trajectory_generator = PPOTrajectoryGenerator()

    # 🎯 使用简化的奖励计算器
    reward_calculator = SimplifiedStepwiseRewardCalculator(
        correct_prediction_reward=config.correct_prediction_reward,
        wrong_prediction_penalty=config.wrong_prediction_penalty,
        repeat_guess_penalty=config.repeat_guess_penalty,
        efficiency_bonus_factor=config.efficiency_bonus_factor
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 🎯 使用Clean Step-wise PPO Trainer
    ppo_trainer = CleanStepwisePPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        kl_coeff=config.kl_coeff,
        clip_epsilon=config.clip_epsilon,
        device=device,
        max_grad_norm=config.max_grad_norm,
        base_learning_rate=config.learning_rate,
        reward_calculator=reward_calculator
    )

    # 🎯 使用Step-wise评估器
    evaluator = StepWiseHangmanEvaluator(
        model, tokenizer, device=device,
        max_steps=args.max_steps
    )

    # 学习率调度器
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )
        print(f"✅ Learning rate scheduler enabled")

    # Early stopping
    best_step_accuracy = 0.0
    patience_counter = 0

    # 安全监控
    total_safety_alerts = 0
    consecutive_safe_epochs = 0

    # 训练计数器
    game_count = 0

    print("🎯 Starting Clean Step-wise PPO training...")
    print("🎯 Algorithm: Step-wise Policy Optimization without probability alignment")
    print("🎯 Objective: Maximize single-step accuracy through pure PPO")

    # 主训练循环
    for epoch in range(config.num_epochs):
        print(f"\n🎯 Epoch {epoch + 1}/{config.num_epochs} - Clean Step-wise PPO Training")
        model.train()

        # 每个epoch开始时清理内存
        clear_gpu_memory()

        # 重置统计
        ppo_trainer.reset_training_stats()

        epoch_losses = []
        epoch_policy_losses = []
        epoch_kl_losses = []
        epoch_step_accuracies = []
        total_steps = 0
        total_correct_steps = 0
        successful_batches = 0
        failed_batches = 0
        total_groups_processed = 0
        epoch_safety_alerts = 0

        # 收集epoch数据
        all_epoch_trajectory_groups = []
        all_word_categories = {}

        # 根据设备调整训练数据量
        if config.debug_mode:
            train_words = dataset.get_train_batch(min(config.batch_size, 4))
        elif device.type == 'cpu':
            train_words = dataset.get_train_batch(min(config.batch_size * 2, 16))
        else:
            train_words = dataset.get_train_batch(min(config.batch_size * 4, 32))

        # 分批处理
        batch_size = 3 if device.type == 'cpu' else config.batch_size
        if config.debug_mode:
            batch_size = min(batch_size, 2)

        total_batches = math.ceil(len(train_words) / batch_size)

        print(f"📊 Processing {len(train_words)} words in {total_batches} batches")
        print(f"   Batch size: {batch_size} trajectory groups")

        for batch_idx, batch_start in enumerate(range(0, len(train_words), batch_size)):
            batch_end = min(batch_start + batch_size, len(train_words))
            word_batch = train_words[batch_start:batch_end]

            game_count += len(word_batch)
            batch_num = batch_idx + 1

            try:
                print(
                    f"\n🎯 Clean Step-wise PPO Batch {batch_num}/{total_batches}: Processing {len(word_batch)} trajectory groups")

                # 批次开始时清理内存
                if device.type == 'cuda':
                    clear_gpu_memory()

                # 生成trajectory groups
                trajectory_groups_batch = []

                for word_idx, word in enumerate(word_batch):
                    print(f"   Generating trajectories for '{word}' ({word_idx + 1}/{len(word_batch)})")

                    # 生成轨迹
                    trajectory_group = trajectory_generator.generate_trajectories_for_word(word, args.k_trajectories)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    all_epoch_trajectory_groups.append(trajectory_group)

                    # 分析trajectory group（使用difficulty classifier）
                    word_category = ppo_trainer.difficulty_classifier.classify_word(word,
                                                                                    trajectory_group["trajectories"])
                    all_word_categories[word] = word_category

                    trajectory_groups_batch.append(trajectory_group)
                    total_groups_processed += 1

                    # 计算基本统计
                    all_step_accuracies = []
                    for traj in trajectory_group["trajectories"]:
                        step_accuracies = traj.get("step_accuracies", [])
                        all_step_accuracies.extend([1.0 if acc else 0.0 for acc in step_accuracies])

                    step_accuracy = np.mean(all_step_accuracies) if all_step_accuracies else 0.0

                    print(f"      ✅ '{word}' ({word_category}): step_acc={step_accuracy:.1%}, "
                          f"total_steps={len(all_step_accuracies)}")

                print(f"   📊 {len(trajectory_groups_batch)}/{len(word_batch)} groups for Clean Step-wise PPO")

                # Clean Step-wise PPO训练
                loss_info = ppo_trainer.train_step_clean_ppo(trajectory_groups_batch)

                # 训练后清理内存
                if device.type == 'cuda':
                    clear_gpu_memory()

                # 安全监控
                if config.enable_safety_monitoring:
                    safety_alert_count = monitor_stepwise_training_safety(loss_info, epoch + 1, batch_num)
                    epoch_safety_alerts += safety_alert_count

                # 处理loss信息
                total_loss = loss_info.get("total_loss", 0.0)
                policy_loss = loss_info.get("policy_loss", 0.0)
                kl_loss = loss_info.get("kl_loss", 0.0)
                current_kl_coeff = loss_info.get("current_kl_coeff", config.kl_coeff)
                backprop_success = loss_info.get("backprop_success", False)

                # 🎯 Step-wise指标
                step_accuracy = loss_info.get("overall_step_accuracy", 0.0)
                total_steps_batch = loss_info.get("total_steps", 0)
                correct_steps_batch = loss_info.get("total_correct_steps", 0)

                category_stats = loss_info.get("category_stats", {})

                # 转换tensor为数值
                if isinstance(total_loss, torch.Tensor):
                    total_loss = total_loss.item()
                if isinstance(policy_loss, torch.Tensor):
                    policy_loss = policy_loss.item()
                if isinstance(kl_loss, torch.Tensor):
                    kl_loss = kl_loss.item()

                if math.isnan(total_loss) or math.isinf(total_loss):
                    print(f"⚠️ Invalid loss for batch {batch_num}, skipping...")
                    failed_batches += 1
                    continue

                if backprop_success:
                    epoch_losses.append(total_loss)
                    epoch_policy_losses.append(policy_loss)
                    epoch_kl_losses.append(kl_loss)
                    epoch_step_accuracies.append(step_accuracy)

                    total_steps += total_steps_batch
                    total_correct_steps += correct_steps_batch

                    successful_batches += 1

                    print(f"✅ Batch {batch_num} Clean Step-wise PPO success:")
                    print(f"   Total loss: {total_loss:.6f}")
                    print(f"   Policy loss: {policy_loss:.6f}")
                    print(f"   KL loss: {kl_loss:.6f}")
                    print(f"   🎯 Step accuracy: {step_accuracy:.1%}")
                    print(f"   🎯 Steps: {correct_steps_batch}/{total_steps_batch}")
                    print(f"   Category distribution: {category_stats}")
                else:
                    failed_batches += 1
                    print(f"❌ Failed Clean Step-wise PPO backprop for batch {batch_num}")

                # 记录trajectory（降低频率）
                if batch_num % max(1, args.log_trajectory_every // 5) == 0:
                    try:
                        if trajectory_groups_batch:
                            first_group = trajectory_groups_batch[0]
                            category = all_word_categories.get(first_group["word"], "IMPROVE")
                            log_stepwise_trajectory_to_wandb(first_group, game_count, category)
                    except Exception as e:
                        print(f"⚠️ Failed to log trajectory: {e}")

                # 记录训练指标
                if batch_num % max(1, total_batches // 5) == 0:
                    try:
                        current_lr = optimizer.param_groups[0]['lr']

                        # 内存监控
                        memory_log = {}
                        if device.type == 'cuda':
                            memory_info = check_gpu_memory()
                            memory_log = {
                                "memory/gpu_allocated": memory_info['allocated'],
                                "memory/gpu_cached": memory_info['cached'],
                                "memory/gpu_free": memory_info['free']
                            }

                        wandb.log({
                            "clean_stepwise_ppo/batch_total_loss": total_loss,
                            "clean_stepwise_ppo/batch_policy_loss": policy_loss,
                            "clean_stepwise_ppo/batch_kl_loss": kl_loss,
                            "clean_stepwise_ppo/current_kl_coeff": current_kl_coeff,
                            "clean_stepwise_ppo/batch_step_accuracy": step_accuracy,
                            "clean_stepwise_ppo/batch_total_steps": total_steps_batch,
                            "clean_stepwise_ppo/batch_correct_steps": correct_steps_batch,
                            "clean_stepwise_ppo/learning_rate": current_lr,
                            "clean_stepwise_ppo/protect_words": category_stats.get("PROTECT", 0),
                            "clean_stepwise_ppo/focus_words": category_stats.get("FOCUS", 0),
                            "clean_stepwise_ppo/improve_words": category_stats.get("IMPROVE", 0),
                            "stepwise_safety/batch_alert_count": safety_alert_count,
                            "game_count": game_count,
                            "epoch": epoch,
                            "batch_number": batch_num,
                            **memory_log
                        })
                    except Exception as e:
                        print(f"⚠️ Failed to log metrics: {e}")

            except Exception as e:
                failed_batches += 1
                print(f"❌ Error processing batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()

                clear_gpu_memory()
                continue

        # Epoch统计
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses) if epoch_policy_losses else 0.0
        avg_kl_loss = sum(epoch_kl_losses) / len(epoch_kl_losses) if epoch_kl_losses else 0.0
        avg_step_accuracy = sum(epoch_step_accuracies) / len(epoch_step_accuracies) if epoch_step_accuracies else 0.0

        total_batches_attempted = successful_batches + failed_batches
        batch_success_rate = successful_batches / total_batches_attempted if total_batches_attempted > 0 else 0.0
        overall_step_accuracy = total_correct_steps / max(total_steps, 1)

        # 更新KL系数
        new_kl_coeff = ppo_trainer.update_epoch_stats({
            'avg_kl_loss': avg_kl_loss,
            'avg_policy_loss': avg_policy_loss
        })

        # 计算各类别性能
        category_performance = calculate_stepwise_category_performance(all_epoch_trajectory_groups, all_word_categories)

        # 获取训练统计
        training_stats = ppo_trainer.get_training_stats()

        # 安全性评估
        if epoch_safety_alerts == 0:
            consecutive_safe_epochs += 1
        else:
            consecutive_safe_epochs = 0
        total_safety_alerts += epoch_safety_alerts

        print(f"\n📊 Epoch {epoch + 1} Clean Step-wise PPO Summary:")
        print(f"   Device: {device}")
        print(f"   Average total loss: {avg_epoch_loss:.6f}")
        print(f"   Average policy loss: {avg_policy_loss:.6f}")
        print(f"   Average KL loss: {avg_kl_loss:.6f}")
        print(f"   🎯 Average step accuracy: {avg_step_accuracy:.1%}")
        print(f"   🎯 Overall step accuracy: {overall_step_accuracy:.1%} ({total_correct_steps}/{total_steps})")
        print(f"   Successful batches: {successful_batches}/{total_batches_attempted}")
        print(f"   🛡️  Safety alerts this epoch: {epoch_safety_alerts}")
        print(f"   🛡️  Consecutive safe epochs: {consecutive_safe_epochs}")

        # 显示内存状态
        if device.type == 'cuda':
            check_gpu_memory()

        # 记录epoch级别指标
        try:
            current_lr = optimizer.param_groups[0]['lr']

            memory_log = {}
            if device.type == 'cuda':
                memory_info = check_gpu_memory()
                memory_log = {
                    "memory/epoch_gpu_allocated": memory_info['allocated'],
                    "memory/epoch_gpu_cached": memory_info['cached'],
                    "memory/epoch_gpu_free": memory_info['free']
                }

            wandb.log({
                "clean_stepwise_ppo_epoch/avg_total_loss": avg_epoch_loss,
                "clean_stepwise_ppo_epoch/avg_policy_loss": avg_policy_loss,
                "clean_stepwise_ppo_epoch/avg_kl_loss": avg_kl_loss,
                "clean_stepwise_ppo_epoch/avg_step_accuracy": avg_step_accuracy,
                "clean_stepwise_ppo_epoch/overall_step_accuracy": overall_step_accuracy,
                "clean_stepwise_ppo_epoch/batch_success_rate": batch_success_rate,
                "clean_stepwise_ppo_epoch/learning_rate": current_lr,
                "clean_stepwise_ppo_epoch/number": epoch + 1,
                "clean_stepwise_ppo_epoch/device": device.type,
                "stepwise_safety/epoch_alert_count": epoch_safety_alerts,
                "stepwise_safety/consecutive_safe_epochs": consecutive_safe_epochs,
                "stepwise_safety/total_alert_count": total_safety_alerts,
                **category_performance,
                **{f"training_{k}": v for k, v in training_stats.items()},
                **memory_log
            })
        except Exception as e:
            print(f"⚠️ Failed to log epoch metrics: {e}")

        # 评估
        if (epoch + 1) % config.eval_interval == 0:
            print("\n🔍 Running Clean Step-wise PPO evaluation...")
            model.eval()

            try:
                # 根据设备调整评估数据量
                if config.debug_mode:
                    eval_words = list(dataset.eval_words)[:8]
                elif device.type == 'cpu':
                    eval_words = list(dataset.eval_words)[:16]
                else:
                    eval_words = list(dataset.eval_words)[:32]

                eval_results = evaluator.evaluate(eval_words, num_games_per_word=1)

                eval_step_accuracy = eval_results.get('step_accuracy', 0.0)
                eval_game_success_rate = eval_results.get('game_success_rate', 0.0)
                eval_avg_confidence = eval_results.get('avg_prediction_confidence', 0.0)
                eval_total_games = eval_results.get('total_games', 0)

                train_efficiency = batch_success_rate * overall_step_accuracy

                print(f"📊 Clean Step-wise PPO Evaluation Results:")
                print(f"   Device: {device}")
                print(f"   🎯 Step Accuracy: {eval_step_accuracy * 100:.1f}%")
                print(f"   Game Success Rate: {eval_game_success_rate * 100:.1f}%")
                print(f"   🎯 Avg Prediction Confidence: {eval_avg_confidence:.3f}")
                print(f"   Total Games: {eval_total_games}")
                print(f"   Training Efficiency: {train_efficiency:.3f}")

                # 记录评估结果
                eval_memory_log = {}
                if device.type == 'cuda':
                    memory_info = check_gpu_memory()
                    eval_memory_log = {
                        "memory/eval_gpu_allocated": memory_info['allocated'],
                        "memory/eval_gpu_free": memory_info['free']
                    }

                wandb.log({
                    "clean_stepwise_ppo_eval/step_accuracy": eval_step_accuracy,
                    "clean_stepwise_ppo_eval/game_success_rate": eval_game_success_rate,
                    "clean_stepwise_ppo_eval/avg_confidence": eval_avg_confidence,
                    "clean_stepwise_ppo_eval/total_games": eval_total_games,
                    "clean_stepwise_ppo_eval/training_efficiency": train_efficiency,
                    "clean_stepwise_ppo_eval/consecutive_safe_epochs": consecutive_safe_epochs,
                    "clean_stepwise_ppo_eval/device": device.type,
                    "clean_stepwise_ppo_eval/epoch": epoch + 1,
                    **eval_memory_log
                })

                # 学习率调度
                if scheduler:
                    scheduler.step(eval_step_accuracy)  # 🎯 基于step accuracy调度
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"📈 Current learning rate: {current_lr:.2e}")

                # Early stopping
                if config.use_early_stopping:
                    if eval_step_accuracy > best_step_accuracy:
                        best_step_accuracy = eval_step_accuracy
                        patience_counter = 0

                        # 保存最佳模型
                        best_model_path = os.path.join(config.output_dir, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'eval_step_accuracy': eval_step_accuracy,
                            'eval_game_success_rate': eval_game_success_rate,
                            'device': device.type,
                            'config': config.__dict__,
                            'clean_stepwise_ppo_stats': {
                                'category_performance': category_performance,
                                'training_stats': training_stats,
                                'best_step_accuracy': eval_step_accuracy,
                                'total_safety_alerts': total_safety_alerts,
                                'consecutive_safe_epochs': consecutive_safe_epochs,
                                'algorithm': 'Clean Step-wise PPO'
                            }
                        }, os.path.join(best_model_path, "model.pth"))
                        print(
                            f"💾 Best Clean Step-wise PPO model saved with step accuracy: {eval_step_accuracy:.1%}")
                    else:
                        patience_counter += 1
                        print(f"⏳ No improvement for {patience_counter} epoch(s)")

                    if patience_counter >= config.early_stop_patience:
                        print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                        print(f"🏆 Best step accuracy: {best_step_accuracy:.1%}")
                        break

            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()

            model.train()

        # 保存模型
        if (epoch + 1) % config.save_interval == 0:
            try:
                save_path = os.path.join(config.output_dir, f"epoch_{epoch + 1}")
                os.makedirs(save_path, exist_ok=True)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'device': device.type,
                    'config': config.__dict__,
                    'clean_stepwise_ppo_stats': {
                        'category_performance': category_performance,
                        'training_stats': training_stats,
                        'avg_policy_loss': avg_policy_loss,
                        'avg_kl_loss': avg_kl_loss,
                        'avg_step_accuracy': avg_step_accuracy,
                        'epoch_safety_alerts': epoch_safety_alerts,
                        'consecutive_safe_epochs': consecutive_safe_epochs
                    }
                }, os.path.join(save_path, "model.pth"))

                tokenizer.save_config(os.path.join(save_path, "tokenizer.json"))
                print(f"✅ Clean Step-wise PPO model saved to {save_path}")

            except Exception as e:
                print(f"❌ Failed to save model: {e}")

    # 最终保存
    try:
        final_save_path = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)

        final_training_stats = ppo_trainer.get_training_stats()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_epoch': config.num_epochs,
            'best_step_accuracy': best_step_accuracy,
            'device': device.type,
            'config': config.__dict__,
            'clean_stepwise_ppo_final_stats': {
                'final_training_stats': final_training_stats,
                'total_safety_alerts': total_safety_alerts,
                'final_consecutive_safe_epochs': consecutive_safe_epochs,
                'algorithm': 'Clean Step-wise PPO - Pure step-level policy optimization'
            }
        }, os.path.join(final_save_path, "model.pth"))

        tokenizer.save_config(os.path.join(final_save_path, "tokenizer.json"))
        print(f"✅ Final Clean Step-wise PPO model saved to {final_save_path}")

    except Exception as e:
        print(f"❌ Failed to save final model: {e}")

    # 训练总结
    final_training_stats = ppo_trainer.get_training_stats()

    print(f"\n🎉 Clean Step-wise PPO Training completed!")
    print(f"🖥️  Final device: {device}")
    print(f"🏆 Best step accuracy achieved: {best_step_accuracy:.1%}")
    print(f"🎯 Clean Step-wise PPO Summary:")
    print(f"   Algorithm: Pure step-level policy optimization without probability alignment")
    print(f"   Overall step accuracy: {final_training_stats.get('overall_step_accuracy', 0.0):.1%}")
    print(f"   🛡️  Total safety alerts: {total_safety_alerts}")
    print(f"   🛡️  Final consecutive safe epochs: {consecutive_safe_epochs}")

    wandb.finish()


if __name__ == "__main__":
    main()