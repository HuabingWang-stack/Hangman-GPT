# config.py - Clean Step-wise PPO配置参数
from dataclasses import dataclass
import torch


@dataclass
class CleanPPOConfig:
    # 🎯 Clean Step-wise PPO核心参数
    kl_coeff: float = 0.08  # KL正则化系数
    clip_epsilon: float = 0.15  # PPO clipping参数
    learning_rate: float = 8e-5  # 学习率
    base_learning_rate: float = 8e-5  # 基础学习率（兼容性）

    # 🎯 词汇分类阈值
    protect_threshold: float = 0.75  # PROTECT类别阈值
    focus_threshold: float = 0.45  # FOCUS类别阈值
    short_word_length: int = 5  # 短词长度阈值

    # 🎯 分层训练权重
    protect_category_weight: float = 0.7  # 保护词汇训练权重
    focus_category_weight: float = 1.5  # 重点词汇训练权重
    improve_category_weight: float = 1.0  # 平衡词汇训练权重

    # 🎯 分层KL权重倍数
    protect_kl_multiplier: float = 1.5  # 保护词汇KL约束倍数
    focus_kl_multiplier: float = 0.7  # 重点词汇KL约束倍数
    improve_kl_multiplier: float = 1.0  # 平衡词汇KL约束倍数

    # 🎯 分层学习率倍数
    protect_lr_multiplier: float = 0.6  # 保护词汇学习率倍数
    focus_lr_multiplier: float = 1.4  # 重点词汇学习率倍数
    improve_lr_multiplier: float = 1.0  # 平衡词汇学习率倍数

    # 🎯 奖励计算参数（简化版）
    correct_prediction_reward: float = 2.0  # 猜对字符的基础奖励
    wrong_prediction_penalty: float = -0.5  # 猜错的惩罚
    repeat_guess_penalty: float = -1.0  # 重复猜测的惩罚
    efficiency_bonus_factor: float = 0.2  # 效率奖励因子

    # 🎯 PPO基础参数
    k_trajectories: int = 4  # 每个词生成的轨迹数
    alpha: float = 0.2  # 奖励缩放因子（向后兼容）
    max_grad_norm: float = 0.5  # 梯度裁剪范数
    max_steps: int = 10  # 最大步数

    # 🎯 训练参数
    batch_size: int = 6  # 批次大小（trajectory groups）
    num_epochs: int = 25  # 训练轮数
    weight_decay: float = 0.01  # 权重衰减

    # 🎯 路径参数
    output_dir: str = ""  # 输出目录
    sft_model_path: str = "sft/models/4xdataset-8layer-final/checkpoint-6000"
    pretrain_path: str = "dataset/225k_10k_5k_10k/pretrain.txt"
    sft_path: str = "dataset/225k_10k_5k_10k/sft.txt"
    grpo_path: str = "dataset/225k_10k_5k_10k/grpo.txt"

    # 🎯 控制参数
    eval_interval: int = 3  # 评估间隔
    save_interval: int = 5  # 保存间隔
    log_trajectory_every: int = 20  # 轨迹记录间隔

    # 🎯 学习率调度
    use_lr_scheduler: bool = True  # 使用学习率调度器
    scheduler_patience: int = 5  # 调度器耐心值
    scheduler_factor: float = 0.75  # 调度器衰减因子
    min_lr: float = 1e-6  # 最小学习率

    # 🎯 Early stopping
    use_early_stopping: bool = True  # 使用早停
    early_stop_patience: int = 8  # 早停耐心值
    early_stop_metric: str = "step_accuracy"  # 基于step accuracy

    # 🎯 调试和监控
    debug_mode: bool = False  # 调试模式
    wandb_project: str = "hangman-clean-stepwise-ppo"  # 项目名

    # 🎯 安全监控参数
    enable_safety_monitoring: bool = True  # 启用安全监控
    max_loss_per_step: float = 3.0  # 单个步骤最大loss
    max_loss_std_per_batch: float = 1.5  # 批次loss标准差阈值
    gradient_norm_threshold: float = 100.0  # 梯度范数异常阈值

    # 🎯 Step-wise评估参数
    step_accuracy_target: float = 0.65  # 目标单步准确率

    # 🎯 内存优化参数
    enable_gradient_checkpointing: bool = False  # 梯度检查点
    max_memory_usage_gb: float = 8.0  # 最大内存使用

    # 🎯 采样参数
    exploration_rate: float = 0.1  # 探索率（vs贪心）

    # 🎯 设备配置
    device: str = "auto"  # 设备选择

    @classmethod
    def get_trainer_config(cls):
        """获取训练器配置"""
        return {
            'kl_coeff': cls.kl_coeff,
            'clip_epsilon': cls.clip_epsilon,
            'max_grad_norm': cls.max_grad_norm,
            'base_learning_rate': cls.base_learning_rate,
        }

    @classmethod
    def get_reward_calculator_config(cls):
        """获取奖励计算器配置"""
        return {
            'correct_prediction_reward': cls.correct_prediction_reward,
            'wrong_prediction_penalty': cls.wrong_prediction_penalty,
            'repeat_guess_penalty': cls.repeat_guess_penalty,
            'efficiency_bonus_factor': cls.efficiency_bonus_factor,
        }

    @classmethod
    def get_difficulty_config(cls):
        """获取难度分类配置"""
        return {
            'protect_threshold': cls.protect_threshold,
            'focus_threshold': cls.focus_threshold,
            'short_word_length': cls.short_word_length,
            'protect_lr_multiplier': cls.protect_lr_multiplier,
            'focus_lr_multiplier': cls.focus_lr_multiplier,
            'improve_lr_multiplier': cls.improve_lr_multiplier,
            'protect_category_weight': cls.protect_category_weight,
            'focus_category_weight': cls.focus_category_weight,
            'improve_category_weight': cls.improve_category_weight,
            'protect_kl_multiplier': cls.protect_kl_multiplier,
            'focus_kl_multiplier': cls.focus_kl_multiplier,
            'improve_kl_multiplier': cls.improve_kl_multiplier,
        }

    @classmethod
    def get_full_config(cls):
        """获取完整配置字典"""
        config_dict = {}
        for field in cls.__dataclass_fields__:
            config_dict[field] = getattr(cls, field)
        return config_dict

    def __post_init__(self):
        """初始化后处理"""
        # 确保兼容性
        if hasattr(self, 'base_learning_rate') and self.base_learning_rate != self.learning_rate:
            self.learning_rate = self.base_learning_rate

        # 设备配置
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        """转换为字典"""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def update_from_args(self, args):
        """从命令行参数更新配置"""
        overridden_params = []

        # 参数映射
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
            'k_trajectories': 'k_trajectories',
            'max_steps': 'max_steps',
            'eval_interval': 'eval_interval',
            'save_interval': 'save_interval',
            'log_trajectory_every': 'log_trajectory_every',
            'use_lr_scheduler': 'use_lr_scheduler',
            'scheduler_patience': 'scheduler_patience',
            'scheduler_factor': 'scheduler_factor',
            'min_lr': 'min_lr',
            'use_early_stopping': 'use_early_stopping',
            'early_stop_patience': 'early_stop_patience',
            'wandb_project': 'wandb_project',
            'debug_mode': 'debug_mode',
            'enable_safety_monitoring': 'enable_safety_monitoring',
            'output_dir': 'output_dir',
        }

        # 应用参数覆盖
        for arg_name, config_attr in param_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                old_value = getattr(self, config_attr, None)
                new_value = getattr(args, arg_name)

                if old_value != new_value:
                    setattr(self, config_attr, new_value)
                    overridden_params.append(f"   {config_attr}: {old_value} → {new_value}")

        # 特殊处理
        if hasattr(args, 'base_learning_rate') and args.base_learning_rate is not None:
            if args.base_learning_rate != self.base_learning_rate:
                old_value = self.base_learning_rate
                self.base_learning_rate = args.base_learning_rate
                self.learning_rate = args.base_learning_rate  # 同步
                overridden_params.append(f"   base_learning_rate: {old_value} → {args.base_learning_rate}")

        return overridden_params


# 向后兼容
GRPOConfig = CleanPPOConfig
Config = CleanPPOConfig