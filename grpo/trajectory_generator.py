# trajectory_generator.py - 明确用途的轨迹生成器
import random
import numpy as np
from typing import List, Dict, Any, Set
from collections import Counter


class MultiPurposeTrajectoryGenerator:
    """多用途轨迹生成器（SFT + PPO）"""

    def __init__(self, mode: str = "ppo"):
        """
        Args:
            mode: "sft" for supervised fine-tuning, "ppo" for reinforcement learning
        """
        self.mode = mode.lower()
        if self.mode not in ["sft", "ppo"]:
            raise ValueError("Mode must be 'sft' or 'ppo'")

    def generate_trajectories_for_word(self, word: str, num_trajectories: int = 4) -> Dict[str, Any]:
        """为单个词生成轨迹"""
        word = word.lower()
        trajectories = []

        for _ in range(num_trajectories):
            if self.mode == "sft":
                trajectory = self._generate_sft_trajectory(word)
            else:  # ppo
                trajectory = self._generate_ppo_trajectory(word)

            if trajectory:
                trajectories.append(trajectory)

        return {
            "word": word,
            "trajectories": trajectories,
            "mode": self.mode
        }

    def _generate_sft_trajectory(self, word: str) -> Dict[str, Any]:
        """生成SFT训练轨迹（带soft targets）"""
        unique_chars = len(set(word))
        max_total_guesses = unique_chars + 5

        # 随机选择总猜测次数
        total_guesses = random.randint(0, max_total_guesses)
        correct_guesses = random.randint(0, min(total_guesses, unique_chars - 1))
        wrong_guesses = min(total_guesses - correct_guesses, 5)

        # 选择要猜对的字符
        word_chars = list(set(word))
        correct_chars = random.sample(word_chars, correct_guesses) if correct_guesses > 0 else []

        # 选择要猜错的字符
        alphabet = set('abcdefghijklmnopqrstuvwxyz')
        available_wrong = alphabet - set(word) - set(correct_chars)
        wrong_chars = random.sample(list(available_wrong), min(wrong_guesses, len(available_wrong)))

        # 组合并打乱猜测顺序
        all_guesses = correct_chars + wrong_chars
        random.shuffle(all_guesses)

        # 构建轨迹步骤
        steps = []
        guessed_so_far = set()
        current_wrong = 0

        for i, char in enumerate(all_guesses):
            # 构建当前游戏状态
            partial_word = ''.join(c if c in guessed_so_far else '_' for c in word)
            guessed_list = ','.join(sorted(guessed_so_far)) if guessed_so_far else ''

            prompt = f"{partial_word}[SEP]{guessed_list}[SEP]{current_wrong}/6"

            # 计算soft targets（SFT模式特有）
            remaining_chars = set(word) - guessed_so_far
            soft_targets = self._calculate_soft_targets(remaining_chars)

            step = {
                "prompt": prompt,
                "action": char,
                "is_correct": char in word,
                "soft_targets": soft_targets  # SFT模式特有
            }

            steps.append(step)
            guessed_so_far.add(char)

            if char not in word:
                current_wrong += 1

        return {
            "word": word,
            "steps": steps,
            "total_guesses": len(steps),
            "correct_guesses": len(correct_chars),
            "wrong_guesses": current_wrong,
            "mode": "sft"
        }

    def _generate_ppo_trajectory(self, word: str) -> Dict[str, Any]:
        """生成PPO训练轨迹（无soft targets）"""
        # PPO模式：模拟真实游戏过程
        trajectory = self._simulate_random_game(word)

        # 添加PPO特有的信息
        steps = trajectory["steps"]
        step_accuracies = [step["is_correct"] for step in steps]

        # 计算step-wise奖励（在reward calculator中会重新计算，这里只是占位）
        step_rewards = [1.0 if acc else -0.5 for acc in step_accuracies]

        trajectory.update({
            "step_accuracies": step_accuracies,
            "step_rewards": step_rewards,  # 占位，实际由reward calculator计算
            "mode": "ppo"
        })

        return trajectory

    def _simulate_random_game(self, word: str) -> Dict[str, Any]:
        """模拟随机游戏过程"""
        guessed_letters = set()
        wrong_guesses = 0
        max_wrong = 6
        steps = []

        alphabet = list('abcdefghijklmnopqrstuvwxyz')

        while wrong_guesses < max_wrong and not set(word).issubset(guessed_letters):
            # 构建当前游戏状态
            partial_word = ''.join(c if c in guessed_letters else '_' for c in word)
            guessed_list = ','.join(sorted(guessed_letters)) if guessed_letters else ''
            prompt = f"{partial_word}[SEP]{guessed_list}[SEP]{wrong_guesses}/6"

            # 随机选择下一个字符（模拟随机策略）
            available_chars = [c for c in alphabet if c not in guessed_letters]
            if not available_chars:
                break

            action = random.choice(available_chars)
            is_correct = action in word

            step = {
                "prompt": prompt,
                "action": action,
                "is_correct": is_correct
            }

            steps.append(step)
            guessed_letters.add(action)

            if not is_correct:
                wrong_guesses += 1

        return {
            "word": word,
            "steps": steps,
            "final_wrong_guesses": wrong_guesses,
            "success": set(word).issubset(guessed_letters),
            "total_steps": len(steps)
        }

    def _calculate_soft_targets(self, remaining_chars: Set[str]) -> Dict[str, float]:
        """计算soft targets（仅SFT模式）"""
        if not remaining_chars:
            # 均匀分布
            return {c: 1.0 / 26 for c in 'abcdefghijklmnopqrstuvwxyz'}

        # 基于字符数量的softmax分布
        char_counts = Counter(remaining_chars)
        total_count = sum(char_counts.values())

        # 温度参数
        temperature = 1.0

        soft_targets = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c in char_counts:
                # 在剩余字符中
                score = char_counts[c] / temperature
            else:
                # 不在剩余字符中
                score = 0.01 / temperature

            soft_targets[c] = score

        # Softmax归一化
        exp_scores = {c: np.exp(score) for c, score in soft_targets.items()}
        total_exp = sum(exp_scores.values())

        normalized_targets = {c: exp_score / total_exp for c, exp_score in exp_scores.items()}

        return normalized_targets

    def generate_batch(self, words: List[str], trajectories_per_word: int = 4) -> List[Dict[str, Any]]:
        """批量生成轨迹"""
        batch = []

        for word in words:
            trajectory_group = self.generate_trajectories_for_word(word, trajectories_per_word)
            batch.append(trajectory_group)

        return batch


class SFTTrajectoryGenerator(MultiPurposeTrajectoryGenerator):
    """SFT专用轨迹生成器"""

    def __init__(self):
        super().__init__(mode="sft")


class PPOTrajectoryGenerator(MultiPurposeTrajectoryGenerator):
    """PPO专用轨迹生成器"""

    def __init__(self):
        super().__init__(mode="ppo")


# 向后兼容
TrajectoryGenerator = PPOTrajectoryGenerator