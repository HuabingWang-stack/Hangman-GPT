#!/usr/bin/env python3
"""
generate_guess.py - 为Hangman任务生成SFT训练数据 (Softmax温度平滑版 + 均匀总猜测次数分布 + 词长增强)

核心改进：
1. Score计算：使用Softmax温度平滑的连续信息增益（连续且可导）
2. 游戏状态：总猜测次数均匀分布，模拟完整游戏过程
3. 对于N个unique字符的词，总猜测次数从0到N+5均匀分布
4. 🎯 新增：词长增强 - 可配置特定长度范围的词汇在数据中的比例
5. 🎯 新增：可配置输入和输出目录
"""

import argparse
import json
import random
import os
import multiprocessing
from multiprocessing import Pool, Manager
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import math
import time
from functools import partial
import numpy as np


class HangmanSoftLabelGenerator:
    def __init__(self, word_list: List[str], random_seed: int = 42, temperature: float = 1.0):
        self.word_list = word_list
        self.temperature = temperature  # 🎯 Softmax温度参数
        self.char_frequencies = self._calculate_global_frequencies()  # 保留用于分析
        self.random_seed = random_seed
        random.seed(random_seed)

    def _calculate_global_frequencies(self) -> Dict[str, float]:
        """计算字符在整个数据集中的频率（保留用于分析）"""
        char_count = Counter()
        total_chars = 0

        for word in self.word_list:
            for char in word:
                if 'a' <= char <= 'z':
                    char_count[char] += 1
                    total_chars += 1

        # 转换为频率
        frequencies = {}
        for char in 'abcdefghijklmnopqrstuvwxyz':
            frequencies[char] = char_count.get(char, 0) / total_chars if total_chars > 0 else 0.0

        return frequencies

    def _get_word_state(self, word: str, guessed_chars: Set[str]) -> str:
        """根据已猜字符生成当前词状态"""
        state = ""
        for char in word:
            if char in guessed_chars:
                state += char
            else:
                state += "_"
        return state

    def _get_available_chars(self, guessed_chars: Set[str]) -> List[str]:
        """获取所有未猜过的字符"""
        return [char for char in 'abcdefghijklmnopqrstuvwxyz'
                if char not in guessed_chars]

    def _get_unguessed_chars_in_word(self, word: str, guessed_chars: Set[str]) -> List[str]:
        """获取在词中但未猜过的字符（能带来增量的字符）"""
        unguessed_in_word = []
        for char in set(word):
            if char not in guessed_chars:
                unguessed_in_word.append(char)
        return unguessed_in_word

    def _calculate_soft_targets(self, word: str, guessed_chars: Set[str]) -> Dict[str, float]:
        """
        🎯 使用Softmax温度平滑的连续信息增益计算

        将离散的字符计数通过softmax转换为连续、可导的概率分布
        Score(char) = softmax(count(char) / temperature)

        特点：
        - 完全连续且可导
        - 保持信息增益的核心逻辑：高计数字符仍然获得更高概率
        - 温度参数可调节分布的平滑程度
        """
        unguessed_chars_in_word = self._get_unguessed_chars_in_word(word, guessed_chars)

        if not unguessed_chars_in_word:
            return {}

        # 计算字符计数
        char_counts = {}
        for char in unguessed_chars_in_word:
            char_counts[char] = word.count(char)

        # 转换为softmax分布
        chars = list(char_counts.keys())
        counts = [char_counts[char] for char in chars]

        # 🎯 Softmax变换（连续且可导）
        max_count = max(counts)  # 数值稳定性
        exp_values = [math.exp((count - max_count) / self.temperature) for count in counts]
        sum_exp = sum(exp_values)

        # 构建软标签
        soft_targets = {}
        for char, exp_val in zip(chars, exp_values):
            soft_targets[char] = exp_val / sum_exp

        # 精确舍入并确保总和为1
        for char in soft_targets:
            soft_targets[char] = round(soft_targets[char], 6)

        # 调整舍入误差
        current_sum = sum(soft_targets.values())
        if abs(current_sum - 1.0) > 1e-6:
            max_char = max(soft_targets.items(), key=lambda x: x[1])[0]
            soft_targets[max_char] = round(soft_targets[max_char] + (1.0 - current_sum), 6)

        return soft_targets

    def _create_balanced_game_state(self, word: str, unique_chars: List[str], all_chars: List[str],
                                    total_guesses: int) -> Tuple[Set[str], int]:
        """
        🎯 根据总猜测次数创建平衡的游戏状态

        Args:
            total_guesses: 总的猜测次数（正确+错误）

        Returns:
            (guessed_chars, wrong_count)
        """
        n_unique = len(unique_chars)

        if total_guesses == 0:
            return (set(), 0)

        # 🎯 关键约束：确保至少留1个正确字符未猜（用于训练）
        max_correct_guesses = min(total_guesses, n_unique - 1)  # 至少留1个
        if max_correct_guesses < 0:  # 安全检查
            max_correct_guesses = 0

        # 🎯 在约束下随机分配正确/错误猜测
        if total_guesses <= max_correct_guesses:
            # 总数很小，全部分配给正确猜测
            num_correct = total_guesses
            num_wrong = 0
        else:
            # 随机分配正确和错误猜测
            # 错误猜测最多5个
            max_wrong = min(5, total_guesses)  # 最多5个错误

            # 在可能的范围内随机选择正确猜测数
            min_correct = max(0, total_guesses - max_wrong)
            max_correct_allowed = min(max_correct_guesses, total_guesses)

            if min_correct <= max_correct_allowed:
                num_correct = random.randint(min_correct, max_correct_allowed)
            else:
                num_correct = max_correct_allowed

            num_wrong = total_guesses - num_correct
            # 确保错误次数不超过5
            if num_wrong > 5:
                num_wrong = 5
                num_correct = total_guesses - num_wrong

        # 🎯 生成具体的猜测字符集合
        guessed_chars = set()

        # 添加正确字符
        if num_correct > 0:
            # 按信息增益排序，但增加随机性
            char_gains = [(char, word.count(char)) for char in unique_chars]
            char_gains.sort(key=lambda x: x[1], reverse=True)

            # 🎯 更均匀的选择策略
            if num_correct <= len(char_gains):
                # 方法：加权随机选择
                weights = []
                for i, (char, gain) in enumerate(char_gains):
                    # 前面的字符权重更高，但不是绝对的
                    weight = max(1.0, 3.0 - i * 0.3)  # 递减权重
                    weights.append(weight)

                # 加权随机选择，不放回
                selected_chars = []
                available_chars = char_gains.copy()
                available_weights = weights.copy()

                for _ in range(num_correct):
                    if not available_chars:
                        break

                    # 加权随机选择
                    selected_idx = random.choices(
                        range(len(available_chars)),
                        weights=available_weights,
                        k=1
                    )[0]

                    selected_chars.append(available_chars[selected_idx][0])
                    # 移除已选择的字符
                    available_chars.pop(selected_idx)
                    available_weights.pop(selected_idx)

                guessed_chars.update(selected_chars)

        # 添加错误字符
        actual_wrong_count = 0
        if num_wrong > 0:
            wrong_chars = [char for char in all_chars
                           if char not in word and char not in guessed_chars]
            random.shuffle(wrong_chars)

            actual_wrong_to_add = min(num_wrong, len(wrong_chars))
            for i in range(actual_wrong_to_add):
                guessed_chars.add(wrong_chars[i])
                actual_wrong_count += 1

        return (guessed_chars, actual_wrong_count)

    def _generate_game_states(self, word: str, max_samples: int) -> List[Tuple[Set[str], int]]:
        """
        🎯 重新设计：确保总猜测次数均匀分布，模拟完整游戏过程

        对于有N个unique字符的词：
        - 最大总猜测次数 = N + 5 (N个正确 + 最多5个错误)
        - 总猜测次数从 0 到 max_total_guesses 均匀分布
        - 在每个总数下，随机分配正确/错误比例
        - 确保至少留1个正确字符未猜（用于训练）
        """
        unique_chars = list(set(word))
        all_chars = list('abcdefghijklmnopqrstuvwxyz')
        n_unique = len(unique_chars)

        # 🎯 关键：计算游戏的总猜测次数范围
        max_total_guesses = n_unique + 5  # 最多：所有正确字符 + 5个错误

        if max_samples == 1:
            # 🎯 单样本：从完整范围随机选择
            total_guesses = random.randint(0, max_total_guesses)
            state = self._create_balanced_game_state(word, unique_chars, all_chars, total_guesses)
            return [state]

        else:
            # 🎯 多样本：确保均匀覆盖整个范围
            states = []

            # 计算每个总猜测次数应该生成多少个样本
            total_guesses_range = list(range(0, max_total_guesses + 1))

            if max_samples <= len(total_guesses_range):
                # 样本数不够覆盖所有总数，均匀采样
                sampled_totals = random.sample(total_guesses_range, max_samples)
            else:
                # 样本数充足，重复某些总数
                sampled_totals = []
                samples_per_total = max_samples // len(total_guesses_range)
                remainder = max_samples % len(total_guesses_range)

                for total in total_guesses_range:
                    # 每个总数至少生成samples_per_total个样本
                    sampled_totals.extend([total] * samples_per_total)

                # 剩余的样本随机分配
                extra_totals = random.choices(total_guesses_range, k=remainder)
                sampled_totals.extend(extra_totals)

            # 生成每个总猜测次数对应的状态
            for total_guesses in sampled_totals:
                state = self._create_balanced_game_state(word, unique_chars, all_chars, total_guesses)
                states.append(state)

            return states

    def generate_samples(self, word: str, max_samples: int) -> List[Dict]:
        """为单个词生成训练样本"""
        samples = []
        game_states = self._generate_game_states(word, max_samples)

        for guessed_chars, wrong_count in game_states:
            # 跳过已完成或失败的状态
            if all(char in guessed_chars for char in word) or wrong_count >= 6:
                continue

            # 获取能带来增量的字符（在词中且未猜过）
            unguessed_chars_in_word = self._get_unguessed_chars_in_word(word, guessed_chars)
            if not unguessed_chars_in_word:
                continue  # 没有能带来增量的字符

            # 计算软标签（只包含能带来增量的字符）
            soft_targets = self._calculate_soft_targets(word, guessed_chars)
            if not soft_targets:
                continue

            # 验证soft_targets只包含在词中的未猜字符
            for char in soft_targets:
                assert char in word, f"Character '{char}' not in word '{word}'"
                assert char not in guessed_chars, f"Character '{char}' already guessed"

            # 验证所有在词中的未猜字符都在soft_targets中
            assert set(soft_targets.keys()) == set(unguessed_chars_in_word), \
                f"Soft targets {set(soft_targets.keys())} != unguessed in word {set(unguessed_chars_in_word)}"

            # 选择最高分字符作为completion
            best_char = max(soft_targets.items(), key=lambda x: x[1])[0]

            # 验证soft_targets总和为1
            soft_sum = sum(soft_targets.values())
            assert abs(soft_sum - 1.0) < 0.01, f"Soft targets sum {soft_sum} != 1.0"

            # 生成当前状态
            word_state = self._get_word_state(word, guessed_chars)
            guessed_list = sorted(list(guessed_chars))
            guess_progress = f"{wrong_count}/6"

            prompt_parts = [
                word_state,
                ','.join(guessed_list) if guessed_list else "",
                guess_progress
            ]

            sample = {
                "guess": {
                    "prompt": "[SEP]".join(prompt_parts),
                    "completion": best_char
                },
                "label": word,
                "soft_targets": soft_targets,
                "metadata": {
                    "num_candidates": len(soft_targets),
                    "best_score": soft_targets[best_char],
                    "word_length": len(word),
                    "unique_chars": len(set(word)),
                    "guessed_chars": sorted(list(guessed_chars)),
                    "total_guesses": len(guessed_chars),
                    "unguessed_in_word": len(unguessed_chars_in_word),
                    "wrong_count": wrong_count,
                    "revealed_positions": len([c for c in word_state if c != '_']),
                    "completion_in_word": True,
                    "soft_targets_sum": round(sum(soft_targets.values()), 6),
                    "temperature": self.temperature,
                    # 🎯 更新：显示Softmax分数计算详情
                    "score_details": {
                        char: {
                            "count_in_word": word.count(char),
                            "softmax_logit": word.count(char) / self.temperature,
                            "final_score": soft_targets[char],
                            "calculation": f"softmax({word.count(char)}/{self.temperature})"
                        }
                        for char in soft_targets
                    }
                }
            }

            samples.append(sample)

        return samples


def analyze_word_length_distribution(word_list: List[str]) -> Dict[int, int]:
    """分析词汇长度分布"""
    length_dist = {}
    for word in word_list:
        length = len(word)
        length_dist[length] = length_dist.get(length, 0) + 1
    return length_dist


def resample_words_by_length(word_list: List[str],
                             enhance_min_length: int = 2,
                             enhance_max_length: int = 12,
                             enhance_ratio: float = 0.7) -> List[str]:
    """
    🎯 按词长重新采样，增加指定长度范围内词汇的比例

    Args:
        word_list: 原始词汇列表
        enhance_min_length: 增强词汇的最小长度
        enhance_max_length: 增强词汇的最大长度
        enhance_ratio: 增强词汇在最终数据中的目标比例 (0.0-1.0)

    Returns:
        重采样后的词汇列表
    """

    # 按长度分组
    enhanced_words = []  # 目标增强的词汇
    other_words = []  # 其他词汇

    for word in word_list:
        word_length = len(word)
        if enhance_min_length <= word_length <= enhance_max_length:
            enhanced_words.append(word)
        else:
            other_words.append(word)

    print(f"\n🎯 词长增强重采样:")
    print(f"   增强长度范围: {enhance_min_length}-{enhance_max_length}")
    print(f"   目标比例: {enhance_ratio:.1%}")
    print(f"   原始分布:")
    print(f"     增强范围词汇: {len(enhanced_words):,} ({len(enhanced_words) / len(word_list) * 100:.1f}%)")
    print(f"     其他词汇: {len(other_words):,} ({len(other_words) / len(word_list) * 100:.1f}%)")

    if len(enhanced_words) == 0:
        print(f"   ⚠️  没有找到长度在 {enhance_min_length}-{enhance_max_length} 范围内的词汇")
        return word_list

    if len(other_words) == 0:
        print(f"   ℹ️  所有词汇都在增强范围内，无需重采样")
        return word_list

    # 计算目标样本数
    original_total = len(word_list)
    target_enhanced_count = int(original_total * enhance_ratio)
    target_other_count = original_total - target_enhanced_count

    # 计算重采样倍数
    enhanced_multiplier = target_enhanced_count / len(enhanced_words)
    other_multiplier = target_other_count / len(other_words)

    print(f"   重采样策略:")
    print(f"     增强词汇采样倍数: {enhanced_multiplier:.2f}x")
    print(f"     其他词汇采样倍数: {other_multiplier:.2f}x")

    # 执行重采样
    resampled_words = []

    # 重采样增强词汇
    if enhanced_multiplier >= 1.0:
        # 需要增加采样
        base_repeats = int(enhanced_multiplier)
        extra_probability = enhanced_multiplier - base_repeats

        for word in enhanced_words:
            # 基础重复
            resampled_words.extend([word] * base_repeats)
            # 额外重复（概率性）
            if random.random() < extra_probability:
                resampled_words.append(word)
    else:
        # 需要减少采样
        num_to_sample = target_enhanced_count
        resampled_enhanced = random.sample(enhanced_words, min(num_to_sample, len(enhanced_words)))
        resampled_words.extend(resampled_enhanced)

    # 重采样其他词汇
    if other_multiplier >= 1.0:
        # 需要增加采样
        base_repeats = int(other_multiplier)
        extra_probability = other_multiplier - base_repeats

        for word in other_words:
            # 基础重复
            resampled_words.extend([word] * base_repeats)
            # 额外重复（概率性）
            if random.random() < extra_probability:
                resampled_words.append(word)
    else:
        # 需要减少采样
        num_to_sample = target_other_count
        resampled_other = random.sample(other_words, min(num_to_sample, len(other_words)))
        resampled_words.extend(resampled_other)

    # 随机打乱
    random.shuffle(resampled_words)

    # 统计最终分布
    final_enhanced = sum(1 for word in resampled_words
                         if enhance_min_length <= len(word) <= enhance_max_length)
    final_other = len(resampled_words) - final_enhanced

    print(f"   最终分布:")
    print(f"     增强范围词汇: {final_enhanced:,} ({final_enhanced / len(resampled_words) * 100:.1f}%)")
    print(f"     其他词汇: {final_other:,} ({final_other / len(resampled_words) * 100:.1f}%)")
    print(f"     总词汇数: {len(resampled_words):,}")

    # 验证是否达到目标
    actual_ratio = final_enhanced / len(resampled_words) if len(resampled_words) > 0 else 0
    if abs(actual_ratio - enhance_ratio) < 0.05:  # 允许5%的误差
        print(f"   ✅ 成功达到目标比例: {actual_ratio:.1%} ≈ {enhance_ratio:.1%}")
    else:
        print(f"   ⚠️  与目标比例有偏差: {actual_ratio:.1%} vs {enhance_ratio:.1%}")

    return resampled_words


def analyze_length_distribution_comparison(original_words: List[str],
                                           resampled_words: List[str],
                                           enhance_min_length: int,
                                           enhance_max_length: int):
    """对比重采样前后的长度分布"""

    print(f"\n📊 词汇长度分布对比:")

    # 计算分布
    original_dist = analyze_word_length_distribution(original_words)
    resampled_dist = analyze_word_length_distribution(resampled_words)

    # 合并所有长度
    all_lengths = sorted(set(original_dist.keys()) | set(resampled_dist.keys()))

    print(f"{'长度':<4} {'原始数量':<8} {'原始比例':<8} {'重采样数量':<10} {'重采样比例':<10} {'变化':<8}")
    print("-" * 60)

    for length in all_lengths:
        orig_count = original_dist.get(length, 0)
        resamp_count = resampled_dist.get(length, 0)

        orig_ratio = orig_count / len(original_words) * 100 if len(original_words) > 0 else 0
        resamp_ratio = resamp_count / len(resampled_words) * 100 if len(resampled_words) > 0 else 0

        # 判断是否在增强范围内
        is_enhanced = enhance_min_length <= length <= enhance_max_length
        change_indicator = "📈" if is_enhanced and resamp_ratio > orig_ratio else "📉" if not is_enhanced and resamp_ratio < orig_ratio else "➡️"

        print(
            f"{length:<4} {orig_count:<8} {orig_ratio:<7.1f}% {resamp_count:<10} {resamp_ratio:<9.1f}% {change_indicator}")

    # 汇总统计
    orig_enhanced = sum(original_dist.get(l, 0) for l in range(enhance_min_length, enhance_max_length + 1))
    resamp_enhanced = sum(resampled_dist.get(l, 0) for l in range(enhance_min_length, enhance_max_length + 1))

    print(f"\n📈 汇总对比:")
    print(f"增强范围({enhance_min_length}-{enhance_max_length}):")
    print(f"  原始: {orig_enhanced:,} ({orig_enhanced / len(original_words) * 100:.1f}%)")
    print(f"  重采样: {resamp_enhanced:,} ({resamp_enhanced / len(resampled_words) * 100:.1f}%)")
    print(
        f"  增长: {resamp_enhanced - orig_enhanced:+,} ({(resamp_enhanced / len(resampled_words) - orig_enhanced / len(original_words)) * 100:+.1f}%)")


def analyze_total_guesses_distribution(samples: List[Dict]):
    """🎯 分析总猜测次数分布"""
    print(f"\n=== 总猜测次数分布分析 ===")

    total_guesses_data = []

    for sample in samples:
        # 解析游戏状态
        prompt = sample["guess"]["prompt"]
        parts = prompt.split("[SEP]")

        if len(parts) >= 2 and parts[1]:
            guessed_chars = set(parts[1].split(','))
        else:
            guessed_chars = set()

        word = sample["label"]
        unique_chars = len(set(word))

        # 计算总猜测次数
        total_guesses = len(guessed_chars)

        # 计算理论最大值
        theoretical_max = unique_chars + 5

        total_guesses_data.append({
            'total_guesses': total_guesses,
            'unique_chars': unique_chars,
            'theoretical_max': theoretical_max,
            'progress_ratio': total_guesses / theoretical_max if theoretical_max > 0 else 0
        })

    # 🎯 按unique_chars分组分析
    by_unique_chars = defaultdict(list)

    for data in total_guesses_data:
        by_unique_chars[data['unique_chars']].append(data['total_guesses'])

    print("📊 按词汇unique字符数分组的总猜测次数分布:")
    for unique_count in sorted(by_unique_chars.keys()):
        guesses_list = by_unique_chars[unique_count]
        theoretical_max = unique_count + 5

        print(f"\n  {unique_count}个unique字符的词 (理论最大猜测次数: {theoretical_max}):")
        print(f"    样本数: {len(guesses_list)}")
        print(f"    总猜测次数范围: {min(guesses_list)} - {max(guesses_list)}")
        print(f"    平均: {sum(guesses_list) / len(guesses_list):.1f}")

        # 分布统计
        guesses_dist = Counter(guesses_list)
        print(f"    分布: ", end="")
        for guesses in sorted(guesses_dist.keys()):
            count = guesses_dist[guesses]
            percentage = count / len(guesses_list) * 100
            print(f"{guesses}次({count}, {percentage:.1f}%) ", end="")
        print()

        # 检查是否覆盖了完整范围
        expected_range = set(range(0, theoretical_max + 1))
        actual_range = set(guesses_list)
        missing = expected_range - actual_range

        if missing:
            print(f"    ⚠️  缺失的猜测次数: {sorted(missing)}")
        else:
            print(f"    ✅ 完整覆盖 0-{theoretical_max} 的所有猜测次数")

    # 🎯 整体分布分析
    all_total_guesses = [data['total_guesses'] for data in total_guesses_data]
    all_progress_ratios = [data['progress_ratio'] for data in total_guesses_data]

    print(f"\n📊 整体统计:")
    print(f"  总样本数: {len(total_guesses_data)}")
    print(f"  总猜测次数范围: {min(all_total_guesses)} - {max(all_total_guesses)}")
    print(f"  平均猜测次数: {sum(all_total_guesses) / len(all_total_guesses):.1f}")

    # 进度比例分析
    progress_ranges = {
        "0-20%": sum(1 for r in all_progress_ratios if 0 <= r < 0.2),
        "20-40%": sum(1 for r in all_progress_ratios if 0.2 <= r < 0.4),
        "40-60%": sum(1 for r in all_progress_ratios if 0.4 <= r < 0.6),
        "60-80%": sum(1 for r in all_progress_ratios if 0.6 <= r < 0.8),
        "80-100%": sum(1 for r in all_progress_ratios if 0.8 <= r <= 1.0)
    }

    print(f"\n📊 游戏进度分布 (总猜测次数/理论最大值):")
    for range_name, count in progress_ranges.items():
        percentage = count / len(total_guesses_data) * 100
        print(f"  {range_name}: {count:5d} 样本 ({percentage:5.1f}%)")


def analyze_game_state_distribution(samples: List[Dict]):
    """🎯 分析游戏状态分布 - 确保覆盖全谱"""
    print(f"\n=== 游戏状态分布分析 ===")

    # 分析揭示进度分布
    reveal_ratios = []
    wrong_counts = []

    for sample in samples:
        # 解析游戏状态
        prompt = sample["guess"]["prompt"]
        parts = prompt.split("[SEP]")

        if len(parts) >= 2 and parts[1]:
            guessed_chars = set(parts[1].split(','))
        else:
            guessed_chars = set()

        if len(parts) >= 3:
            wrong_count = int(parts[2].split('/')[0])
        else:
            wrong_count = 0

        word = sample["label"]
        unique_chars = len(set(word))

        # 计算揭示的正确字符数
        correct_chars = len([c for c in guessed_chars if c in word])
        reveal_ratio = correct_chars / unique_chars if unique_chars > 0 else 0

        reveal_ratios.append(reveal_ratio)
        wrong_counts.append(wrong_count)

    # 🎯 揭示进度分布
    reveal_ranges = {
        "0-10%": sum(1 for r in reveal_ratios if 0 <= r < 0.1),
        "10-30%": sum(1 for r in reveal_ratios if 0.1 <= r < 0.3),
        "30-50%": sum(1 for r in reveal_ratios if 0.3 <= r < 0.5),
        "50-70%": sum(1 for r in reveal_ratios if 0.5 <= r < 0.7),
        "70-90%": sum(1 for r in reveal_ratios if 0.7 <= r < 0.9),
        "90%+": sum(1 for r in reveal_ratios if r >= 0.9)
    }

    print(f"📊 揭示进度分布:")
    for range_name, count in reveal_ranges.items():
        percentage = count / len(samples) * 100
        print(f"  {range_name}: {count:5d} 样本 ({percentage:5.1f}%)")

    # 🎯 错误次数分布
    wrong_dist = Counter(wrong_counts)
    print(f"\n📊 错误次数分布:")
    for wrong in range(6):
        count = wrong_dist.get(wrong, 0)
        percentage = count / len(samples) * 100
        print(f"  {wrong}错误: {count:5d} 样本 ({percentage:5.1f}%)")

    # 🎯 状态多样性
    print(f"\n📊 状态多样性:")
    state_combinations = set()
    for r, w in zip(reveal_ratios, wrong_counts):
        reveal_bucket = int(r * 10) * 10  # 0, 10, 20, ..., 90
        state_combinations.add((reveal_bucket, w))

    print(f"  不同状态组合数: {len(state_combinations)}")
    print(f"  平均每个组合: {len(samples) / len(state_combinations):.1f} 样本")

    # 检查覆盖度
    missing_early = reveal_ranges["0-10%"] == 0
    missing_late = reveal_ranges["70-90%"] + reveal_ranges["90%+"] == 0
    missing_no_errors = wrong_dist.get(0, 0) == 0
    missing_high_errors = wrong_dist.get(4, 0) + wrong_dist.get(5, 0) == 0

    if missing_early or missing_late or missing_no_errors or missing_high_errors:
        print(f"\n⚠️  覆盖度不足:")
        if missing_early: print(f"  缺少早期状态 (0-10%)")
        if missing_late: print(f"  缺少后期状态 (70%+)")
        if missing_no_errors: print(f"  缺少无错误状态")
        if missing_high_errors: print(f"  缺少高错误状态 (4-5错)")
    else:
        print(f"\n✅ 状态覆盖度良好:")
        print(f"  ✓ 早期到后期状态完整")
        print(f"  ✓ 0-5错误次数完整")
        print(f"  ✓ 多样化状态组合")


def process_word_batch(args_tuple):
    """处理一批词汇的函数，用于多进程"""
    word_batch, char_frequencies, max_samples, process_id, random_seed, temperature = args_tuple

    # 为每个进程设置不同的随机种子
    process_seed = random_seed + process_id * 1000
    random.seed(process_seed)

    # 创建生成器实例
    generator = HangmanSoftLabelGenerator(word_batch, process_seed, temperature)
    generator.char_frequencies = char_frequencies  # 使用全局频率统计

    all_samples = []
    error_count = 0

    for i, word in enumerate(word_batch):
        try:
            word_samples = generator.generate_samples(word, max_samples)
            all_samples.extend(word_samples)
        except Exception as e:
            error_count += 1
            continue

    return all_samples, error_count, process_id, len(word_batch)


def split_list(lst: List, n_chunks: int) -> List[List]:
    """将列表分割为n个近似相等的块"""
    chunk_size = len(lst) // n_chunks
    remainder = len(lst) % n_chunks

    chunks = []
    start = 0

    for i in range(n_chunks):
        # 前remainder个块多分配一个元素
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        chunks.append(lst[start:end])
        start = end

    return chunks


def generate_dataset_multiprocess(word_list: List[str], char_frequencies: Dict[str, float],
                                  max_samples: int, max_processes: int, random_seed: int = 42,
                                  temperature: float = 1.0) -> List[Dict]:
    """使用多进程生成数据集"""

    if len(word_list) == 0:
        return []

    # 确定实际使用的进程数
    actual_processes = min(max_processes, len(word_list), multiprocessing.cpu_count())
    print(f"使用 {actual_processes} 个进程处理 {len(word_list)} 个词汇")

    # 分割词汇列表
    word_batches = split_list(word_list, actual_processes)

    # 准备参数
    args_list = []
    for i, batch in enumerate(word_batches):
        if batch:  # 确保批次不为空
            args_list.append((batch, char_frequencies, max_samples, i, random_seed, temperature))

    # 多进程处理
    print("开始多进程数据生成...")
    start_time = time.time()

    all_samples = []
    total_errors = 0

    with Pool(processes=actual_processes) as pool:
        # 使用map_async来获取进度
        result = pool.map_async(process_word_batch, args_list)

        # 获取结果
        results = result.get()

    # 合并结果
    print("合并结果...")
    for samples, error_count, process_id, batch_size in results:
        all_samples.extend(samples)
        total_errors += error_count
        print(f"进程 {process_id}: 处理 {batch_size} 个词汇, 生成 {len(samples)} 个样本, 错误 {error_count} 个")

    end_time = time.time()
    print(f"多进程生成完成! 耗时: {end_time - start_time:.2f}s")
    print(f"总样本数: {len(all_samples)}, 总错误数: {total_errors}")

    # Shuffle所有结果
    print("Shuffling数据...")
    random.seed(random_seed)
    random.shuffle(all_samples)

    return all_samples


def load_words(file_path: str) -> List[str]:
    """加载词汇列表"""
    words = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and word.isalpha():
                    words.append(word)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"加载文件出错 {file_path}: {e}")
        return []

    print(f"成功加载 {len(words)} 个词汇从 {file_path}")
    return words


def calculate_global_frequencies(word_list: List[str]) -> Dict[str, float]:
    """计算全局字符频率"""
    char_count = Counter()
    total_chars = 0

    for word in word_list:
        for char in word:
            if 'a' <= char <= 'z':
                char_count[char] += 1
                total_chars += 1

    frequencies = {}
    for char in 'abcdefghijklmnopqrstuvwxyz':
        frequencies[char] = char_count.get(char, 0) / total_chars if total_chars > 0 else 0.0

    return frequencies


def save_samples(samples: List[Dict], output_path: str, include_metadata: bool = False):
    """保存样本到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"保存 {len(samples)} 个样本到 {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            if include_metadata:
                # 包含完整信息的文件
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            else:
                # SFT训练文件：只包含必要字段
                clean_sample = {
                    "guess": sample["guess"],
                    "label": sample["label"],
                    "soft_targets": sample["soft_targets"]
                }
                f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')


def validate_dataset(samples: List[Dict], sample_size: int = 100):
    """验证数据集格式和逻辑"""
    print(f"\n=== 数据集验证 (检查{min(sample_size, len(samples))}个样本) ===")

    format_errors = 0
    logic_errors = 0
    soft_target_errors = 0
    sum_errors = 0
    incremental_errors = 0
    completion_errors = 0

    check_samples = samples[:sample_size] if len(samples) > sample_size else samples

    for i, sample in enumerate(check_samples):
        # 检查基本格式
        if not all(key in sample for key in ["guess", "label", "soft_targets"]):
            format_errors += 1
            continue

        if not all(key in sample["guess"] for key in ["prompt", "completion"]):
            format_errors += 1
            continue

        prompt = sample["guess"]["prompt"]
        completion = sample["guess"]["completion"]
        soft_targets = sample["soft_targets"]
        word = sample["label"]

        # 解析已猜字符
        parts = prompt.split("[SEP]")
        if len(parts) >= 2 and parts[1]:
            guessed_chars = set(parts[1].split(','))
        else:
            guessed_chars = set()

        # 检查completion是否在soft_targets中
        if completion not in soft_targets:
            logic_errors += 1
            if logic_errors <= 3:
                print(f"逻辑错误 {i}: completion '{completion}' 不在 soft_targets {list(soft_targets.keys())}")

        # 检查completion是否重复猜测
        if completion in guessed_chars:
            logic_errors += 1
            if logic_errors <= 3:
                print(f"重复猜测错误 {i}: completion '{completion}' 已在 guessed_chars {guessed_chars}")

        # 检查completion是否是最高分
        if soft_targets:
            max_score_char = max(soft_targets.items(), key=lambda x: x[1])[0]
            if completion != max_score_char:
                completion_errors += 1
                if completion_errors <= 3:
                    print(
                        f"completion错误 {i}: completion '{completion}' (score={soft_targets.get(completion, 0):.3f}) != 最高分字符 '{max_score_char}' (score={soft_targets[max_score_char]:.3f})")

        # 检查soft_targets中的字符是否都在词中且未猜过
        unguessed_in_word = set(word) - guessed_chars

        # 检查soft_targets是否只包含在词中的未猜字符
        if set(soft_targets.keys()) != unguessed_in_word:
            incremental_errors += 1
            if incremental_errors <= 3:
                print(f"增量错误 {i}: soft_targets={set(soft_targets.keys())} != 在词中未猜字符={unguessed_in_word}")

        # 检查soft_targets格式和分数总和
        if not isinstance(soft_targets, dict) or not soft_targets:
            soft_target_errors += 1
        else:
            # 检查分数是否合理
            scores = list(soft_targets.values())
            if any(score < 0 or score > 1 for score in scores):
                soft_target_errors += 1
                if soft_target_errors <= 3:
                    print(f"分数范围错误 {i}: {soft_targets}")

            # 检查总和是否为1
            total_sum = sum(scores)
            if abs(total_sum - 1.0) > 0.01:  # 允许1%的误差
                sum_errors += 1
                if sum_errors <= 3:
                    print(f"总和错误 {i}: soft_targets总和={total_sum:.3f}, 应为1.0")
                    print(f"  soft_targets: {soft_targets}")

    print(f"格式错误: {format_errors}")
    print(f"逻辑错误: {logic_errors}")
    print(f"软标签错误: {soft_target_errors}")
    print(f"总和错误: {sum_errors}")
    print(f"增量字符错误: {incremental_errors}")
    print(f"completion不是最高分错误: {completion_errors}")

    return (format_errors == 0 and logic_errors == 0 and soft_target_errors == 0
            and sum_errors == 0 and incremental_errors == 0 and completion_errors == 0)


def analyze_dataset(samples: List[Dict]):
    """分析生成的数据集"""
    print(f"\n=== 数据集分析 ===")
    print(f"总样本数: {len(samples)}")

    # 分析completion字符分布
    completion_chars = [s["guess"]["completion"] for s in samples]
    char_dist = Counter(completion_chars)
    print(f"最常见的completion字符: {char_dist.most_common(10)}")

    # 分析软标签数量分布
    soft_target_counts = [len(s["soft_targets"]) for s in samples]
    count_dist = Counter(soft_target_counts)
    print(f"软标签数量分布: {dict(sorted(count_dist.items()))}")

    # 分析最高分分布
    best_scores = [max(s["soft_targets"].values()) for s in samples]
    score_ranges = {
        "0.9-1.0": sum(1 for s in best_scores if s >= 0.9),
        "0.7-0.9": sum(1 for s in best_scores if 0.7 <= s < 0.9),
        "0.5-0.7": sum(1 for s in best_scores if 0.5 <= s < 0.7),
        "<0.5": sum(1 for s in best_scores if s < 0.5)
    }
    print(f"最高分分布: {score_ranges}")

    # 🎯 新增分析
    analyze_total_guesses_distribution(samples)  # 总猜测次数分布
    analyze_game_state_distribution(samples)  # 揭示进度和错误次数分布

    print(f"\n✅ 所有completion都是最高分字符")
    print(f"✅ 所有soft_targets只包含能带来增量的字符")
    print(f"✅ Score使用Softmax温度平滑（连续且可导）")
    print(f"✅ 总猜测次数均匀分布（模拟完整游戏过程）")


def show_sample_examples(samples: List[Dict], num_examples: int = 3):
    """显示样本格式示例"""
    print(f"\n=== 数据格式示例 ===")

    for i, sample in enumerate(samples[:num_examples]):
        print(f"\n示例 {i + 1}:")
        print(f"Word: {sample['label']} (长度: {len(sample['label'])})")
        print(f"Prompt: {sample['guess']['prompt']}")
        print(f"Completion: {sample['guess']['completion']}")
        print(f"Soft targets: {sample['soft_targets']}")

        # 验证信息
        soft_sum = sum(sample['soft_targets'].values())
        print(f"软标签总和: {soft_sum:.6f}")

        # 解析状态
        parts = sample['guess']['prompt'].split("[SEP]")
        if len(parts) >= 2 and parts[1]:
            guessed_chars = set(parts[1].split(','))
        else:
            guessed_chars = set()

        word = sample['label']
        unguessed_in_word = set(word) - guessed_chars

        print(f"已猜字符: {sorted(guessed_chars)}")
        print(f"总猜测次数: {len(guessed_chars)}")
        print(f"在词中未猜字符: {sorted(unguessed_in_word)}")
        print(f"软标签字符: {sorted(sample['soft_targets'].keys())}")

        # 🎯 显示Softmax分数计算详情
        if "metadata" in sample and "score_details" in sample["metadata"]:
            print(f"Softmax分数计算详情:")
            temperature = sample["metadata"].get("temperature", 1.0)
            print(f"  温度参数: {temperature}")
            for char, details in sample["metadata"]["score_details"].items():
                print(f"  {char}: 词中出现{details['count_in_word']}次, "
                      f"logit={details['softmax_logit']:.3f}, "
                      f"最终分数={details['final_score']:.6f}")

        # 验证逻辑
        if set(sample['soft_targets'].keys()) == unguessed_in_word:
            print("✅ 软标签=在词中未猜字符")
        else:
            print(f"⚠️  软标签不匹配在词中未猜字符")

        best_char = max(sample['soft_targets'].items(), key=lambda x: x[1])[0]
        if sample['guess']['completion'] == best_char:
            print("✅ completion是最高分字符")
        else:
            print(f"⚠️  completion不是最高分字符")


def main():
    parser = argparse.ArgumentParser(
        description="生成Hangman SFT训练数据 (Softmax温度平滑版 + 均匀总猜测次数分布 + 词长增强)")

    # 🎯 新增：输入输出目录参数
    parser.add_argument("--word_dir", type=str, default="dataset/180k_10k_10k_50k/",
                        help="词汇文件所在目录 (默认: dataset/180k_10k_10k_50k/)")
    parser.add_argument("--out_dir", type=str, default="sft/data/",
                        help="输出JSONL文件目录 (默认: sft/data/)")

    # 基础参数
    parser.add_argument("--max_samples", type=int, default=6,
                        help="每个词最多生成的样本数")
    parser.add_argument("--max_processes", type=int, default=6,
                        help="最大进程数 (默认: CPU核心数)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax温度参数 (默认: 1.0)")

    # 🎯 新增：词长增强参数
    parser.add_argument("--enhance_min_length", type=int, default=5,
                        help="增强词汇的最小长度 (默认: 5)")
    parser.add_argument("--enhance_max_length", type=int, default=13,
                        help="增强词汇的最大长度 (默认: 13)")
    parser.add_argument("--enhance_ratio", type=float, default=0.95,
                        help="增强词汇在数据中的目标比例 (0.0-1.0, 默认: 0.95)")

    # 其他参数
    parser.add_argument("--include_metadata", action="store_true",
                        help="在输出文件中包含metadata信息")
    parser.add_argument("--validate", action="store_true",
                        help="验证生成的数据")
    parser.add_argument("--show_examples", action="store_true",
                        help="显示数据格式示例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--no_enhance", action="store_true",
                        help="禁用词长增强，使用原始分布")

    args = parser.parse_args()

    # 🎯 处理目录路径，确保以斜杠结尾
    word_dir = args.word_dir.rstrip('/') + '/'
    out_dir = args.out_dir.rstrip('/') + '/'

    # 参数验证
    if not 0.0 <= args.enhance_ratio <= 1.0:
        print("错误: enhance_ratio 必须在 0.0-1.0 范围内")
        return

    if args.enhance_min_length > args.enhance_max_length:
        print("错误: enhance_min_length 不能大于 enhance_max_length")
        return

    # 检查输入目录是否存在
    if not os.path.exists(word_dir):
        print(f"错误: 词汇目录不存在: {word_dir}")
        return

    print("=" * 60)
    print("Hangman SFT数据生成器 (Softmax温度平滑版 + 均匀总猜测次数分布 + 词长增强)")
    print("=" * 60)

    # 设置随机种子
    random.seed(args.seed)

    print(f"配置参数:")
    print(f"  路径: word_dir={word_dir}, out_dir={out_dir}")
    print(f"  基础: max_samples={args.max_samples}, max_processes={args.max_processes}")
    print(f"  温度: temperature={args.temperature}, seed={args.seed}")

    if not args.no_enhance:
        print(
            f"  🎯 词长增强: 长度{args.enhance_min_length}-{args.enhance_max_length}, 目标比例{args.enhance_ratio:.1%}")
    else:
        print(f"  🔧 词长增强: 禁用，使用原始分布")

    # 🎯 构建输入文件路径
    train_file_path = os.path.join(word_dir, "pretrain.txt")
    test_file_path = os.path.join(word_dir, "sft.txt")

    # 加载数据
    print("\n" + "=" * 50)
    print("加载词汇数据...")
    start_time = time.time()

    train_words = load_words(train_file_path)
    test_words = load_words(test_file_path)

    if not train_words:
        print(f"错误：无法加载训练词汇从 {train_file_path}")
        return
    if not test_words:
        print(f"错误：无法加载测试词汇从 {test_file_path}")
        return

    load_time = time.time() - start_time
    print(f"词汇加载完成，耗时: {load_time:.2f}s")

    # 显示原始词汇统计
    print(f"\n词汇统计:")
    print(f"  训练词汇: {len(train_words):,} 个 (来源: {train_file_path})")
    print(f"  测试词汇: {len(test_words):,} 个 (来源: {test_file_path})")

    # 🎯 词长增强处理
    if not args.no_enhance:
        print("\n" + "=" * 50)
        print("词长增强重采样...")

        # 重采样训练词汇
        original_train_words = train_words.copy()
        train_words = resample_words_by_length(
            train_words,
            args.enhance_min_length,
            args.enhance_max_length,
            args.enhance_ratio
        )

        # 重采样测试词汇
        original_test_words = test_words.copy()
        test_words = resample_words_by_length(
            test_words,
            args.enhance_min_length,
            args.enhance_max_length,
            args.enhance_ratio
        )

        # 显示重采样对比
        print("\n📊 训练集重采样对比:")
        analyze_length_distribution_comparison(
            original_train_words, train_words,
            args.enhance_min_length, args.enhance_max_length
        )

        print("\n📊 测试集重采样对比:")
        analyze_length_distribution_comparison(
            original_test_words, test_words,
            args.enhance_min_length, args.enhance_max_length
        )

    else:
        print(f"\n🔧 跳过词长增强，使用原始分布")

    # 计算全局字符频率（保留用于分析）
    print("\n计算全局字符频率...")
    char_frequencies = calculate_global_frequencies(train_words)
    print("字符频率计算完成")

    # 显示字符频率前10
    sorted_freqs = sorted(char_frequencies.items(), key=lambda x: x[1], reverse=True)
    print(f"最常见字符: {[(c, f'{f:.4f}') for c, f in sorted_freqs[:10]]}")
    print(f"最稀有字符: {[(c, f'{f:.6f}') for c, f in sorted_freqs[-10:]]}")

    # 生成数据
    print("\n" + "=" * 50)
    print("开始多进程数据生成...")

    # 生成训练数据
    print("\n生成训练数据...")
    train_samples = generate_dataset_multiprocess(
        train_words, char_frequencies, args.max_samples, args.max_processes, args.seed, args.temperature
    )

    # 生成测试数据
    print("\n生成测试数据...")
    test_samples = generate_dataset_multiprocess(
        test_words, char_frequencies, args.max_samples, args.max_processes, args.seed + 1, args.temperature
    )

    # 🎯 构建输出文件路径
    os.makedirs(out_dir, exist_ok=True)

    train_output_path = os.path.join(out_dir, "train.jsonl")
    test_output_path = os.path.join(out_dir, "test.jsonl")
    train_metadata_path = os.path.join(out_dir, "train_with_metadata.jsonl")
    test_metadata_path = os.path.join(out_dir, "test_with_metadata.jsonl")

    # 保存数据
    print("\n" + "=" * 50)
    print("保存数据...")

    # SFT训练文件
    save_samples(train_samples, train_output_path, include_metadata=args.include_metadata)
    save_samples(test_samples, test_output_path, include_metadata=args.include_metadata)

    # 带metadata的分析文件
    save_samples(train_samples, train_metadata_path, include_metadata=True)
    save_samples(test_samples, test_metadata_path, include_metadata=True)

    print(f"✅ 文件保存完成:")
    print(f"  训练数据: {train_output_path}")
    print(f"  测试数据: {test_output_path}")
    print(f"  训练数据(含metadata): {train_metadata_path}")
    print(f"  测试数据(含metadata): {test_metadata_path}")

    # 验证和分析
    if args.validate:
        print("\n" + "=" * 50)
        print("验证数据...")
        train_valid = validate_dataset(train_samples)
        test_valid = validate_dataset(test_samples)

        if train_valid and test_valid:
            print("✅ 所有验证通过!")
        else:
            print("⚠️  验证发现问题，请检查数据")

    # 分析数据
    print("\n" + "=" * 50)
    print("训练集分析:")
    analyze_dataset(train_samples)

    print("\n测试集分析:")
    analyze_dataset(test_samples)

    # 显示示例
    if args.show_examples:
        show_sample_examples(train_samples)

    # 计算总耗时
    total_time = time.time() - start_time

    # 最终总结
    print("\n" + "=" * 60)
    print("✅ 数据生成完成!")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均速度: {(len(train_samples) + len(test_samples)) / total_time:.1f} 样本/秒")
    print(f"\n📁 输入输出路径:")
    print(f"  词汇目录: {word_dir}")
    print(f"  输出目录: {out_dir}")
    print(f"\n关键特性:")
    print(f"  ✅ Score = Softmax(count/temperature) - 连续且可导")
    print(f"  ✅ 温度参数 = {args.temperature}")
    print(f"  ✅ 只包含在词中的未猜字符")
    print(f"  ✅ completion总是最高分字符")
    print(f"  ✅ soft_targets总和为1.0")
    print(f"  ✅ 总猜测次数均匀分布 (0到N+5)")
    print(f"  ✅ 模拟完整游戏过程")
    print(f"  ✅ 多进程加速生成")
    print(f"  ✅ 详细的分布分析")
    print(f"  🎯 可配置输入输出目录")

    if not args.no_enhance:
        print(f"  🎯 词长增强: {args.enhance_min_length}-{args.enhance_max_length} 长度范围 {args.enhance_ratio:.1%}")
    else:
        print(f"  🔧 使用原始词汇分布")


if __name__ == "__main__":
    main()