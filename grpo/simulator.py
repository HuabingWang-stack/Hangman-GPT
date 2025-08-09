# simulator.py - Step-wise + Probability Distribution Hangman游戏模拟器（完整修复版）
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Any, Set, Optional
from collections import Counter, defaultdict


class StepwiseProbabilityHangmanSimulator:
    """基于Step-wise准确率和概率分布的Hangman游戏模拟器"""

    def __init__(self, model, tokenizer, device='cuda',
                 temperature: float = 1.0,
                 max_wrong_guesses: int = 6,
                 max_steps: int = 10,
                 record_probabilities: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_wrong_guesses = max_wrong_guesses
        self.max_steps = max_steps
        self.record_probabilities = record_probabilities

        # 🎯 预计算a-z字符映射 - 基于实际tokenizer结构
        self.a_z_chars = 'abcdefghijklmnopqrstuvwxyz'
        self.char_to_id = {}
        self.id_to_char = {}
        self._initialize_char_mappings()

        print(f"🎯 StepwiseProbabilityHangmanSimulator initialized:")
        print(f"   Temperature: {temperature}")
        print(f"   Max wrong guesses: {max_wrong_guesses}")
        print(f"   Max steps: {max_steps}")
        print(f"   Record probabilities: {record_probabilities}")
        print(f"   A-Z chars mapped: {len(self.char_to_id)}/26")

    def _initialize_char_mappings(self):
        """🔧 初始化字符映射 - 基于实际tokenizer结构"""
        try:
            # 🔧 修复：直接访问tokenizer的char_to_id字典
            if hasattr(self.tokenizer, 'char_to_id') and isinstance(self.tokenizer.char_to_id, dict):
                for c in self.a_z_chars:
                    if c in self.tokenizer.char_to_id:
                        token_id = self.tokenizer.char_to_id[c]
                        self.char_to_id[c] = token_id
                        self.id_to_char[token_id] = c
                print("✅ Using tokenizer.char_to_id (dict)")

                # 显示映射示例
                sample_mappings = {c: self.char_to_id[c] for c in ['a', 'z'] if c in self.char_to_id}
                print(f"   Sample mappings: {sample_mappings}")
                return
            else:
                print("⚠️ Tokenizer char_to_id not found or not a dict, using fallback mapping")
                self._create_fallback_mappings()

        except Exception as e:
            print(f"❌ Error initializing character mappings: {e}")
            self._create_fallback_mappings()

    def _create_fallback_mappings(self):
        """创建fallback字符映射"""
        print("🔧 Creating fallback character mappings...")
        # 根据实际tokenizer结构，a-z在位置4-29（跳过控制tokens）
        for i, c in enumerate(self.a_z_chars):
            token_id = 4 + i  # 跳过 <pad>, <bos>, <eos>, <unk>
            self.char_to_id[c] = token_id
            self.id_to_char[token_id] = c

        sample_mappings = {c: self.char_to_id[c] for c in ['a', 'z']}
        print(f"✅ Fallback mappings created: {sample_mappings}")

    def simulate_game(self, word: str, strategy: str = "greedy") -> Dict[str, Any]:
        """
        模拟单个Hangman游戏

        Args:
            word: 目标词汇
            strategy: 预测策略 ("greedy", "sampling", "mixed")

        Returns:
            游戏结果字典，包含step-wise统计
        """
        word = word.lower()
        guessed_letters = set()
        wrong_guesses = 0

        game_result = {
            "word": word,
            "strategy": strategy,
            "success": False,
            "failure_reason": "",
            "steps": [],
            "total_steps": 0,
            "correct_steps": 0,
            "wrong_guesses": 0,
            "step_accuracies": [],
            "step_prob_alignments": [],
            "step_kl_divergences": [],
            "step_confidences": [],
            "violations": [],
            "final_state": "",
            "game_stats": {}
        }

        # 初始化游戏状态
        current_state = ['_' if c.isalpha() else c for c in word]

        self.model.eval()

        for step_idx in range(self.max_steps):
            # 🔧 修复：使用正确的Hangman prompt格式
            prompt = self._build_hangman_prompt(current_state, guessed_letters, wrong_guesses)

            # 🎯 预测下一个字符
            prediction_result = self._predict_next_char(
                prompt, word, guessed_letters, strategy
            )

            if not prediction_result:
                game_result["failure_reason"] = "prediction_failed"
                break

            predicted_char = prediction_result["predicted_char"]
            model_distribution = prediction_result.get("model_distribution", {})
            ideal_distribution = prediction_result.get("ideal_distribution", {})
            kl_divergence = prediction_result.get("kl_divergence", 0.0)
            confidence = prediction_result.get("confidence", 0.0)
            prob_alignment = prediction_result.get("prob_alignment", 0.0)

            # 检查违规
            is_violation = predicted_char in guessed_letters
            if is_violation:
                game_result["violations"].append({
                    "step": step_idx,
                    "type": "repeat_guess",
                    "char": predicted_char
                })

            # 判断是否正确
            is_correct = predicted_char in word and not is_violation

            # 记录步骤详情
            step_detail = {
                "step_id": step_idx,
                "prompt": prompt,
                "predicted_char": predicted_char,
                "is_correct": is_correct,
                "is_violation": is_violation,
                "current_state": ''.join(current_state),
                "guessed_letters": list(guessed_letters),
                "wrong_guesses": wrong_guesses,
                "confidence": confidence,
                "kl_divergence": kl_divergence,
                "prob_alignment": prob_alignment,
            }

            # 🎯 添加概率分布信息
            if self.record_probabilities:
                step_detail.update({
                    "model_distribution": model_distribution,
                    "ideal_distribution": ideal_distribution,
                    "distribution_entropy": self._calculate_entropy(model_distribution),
                    "ideal_entropy": self._calculate_entropy(ideal_distribution)
                })

            game_result["steps"].append(step_detail)

            # 更新step-wise统计
            game_result["step_accuracies"].append(is_correct)
            game_result["step_prob_alignments"].append(prob_alignment)
            game_result["step_kl_divergences"].append(kl_divergence)
            game_result["step_confidences"].append(confidence)

            # 更新游戏状态
            guessed_letters.add(predicted_char)

            if predicted_char in word:
                # 猜对了
                for i, c in enumerate(word):
                    if c == predicted_char:
                        current_state[i] = c

                # 检查游戏是否完成
                if '_' not in current_state:
                    game_result["success"] = True
                    break
            else:
                # 猜错了
                wrong_guesses += 1
                if wrong_guesses >= self.max_wrong_guesses:
                    game_result["failure_reason"] = "too_many_wrong"
                    break

            # 违规也算失败
            if is_violation:
                game_result["failure_reason"] = "violation"
                break

        # 如果没有明确失败原因且未成功
        if not game_result["success"] and not game_result["failure_reason"]:
            game_result["failure_reason"] = "max_steps_reached"

        # 计算最终统计
        game_result["total_steps"] = len(game_result["steps"])
        game_result["correct_steps"] = sum(game_result["step_accuracies"])
        game_result["wrong_guesses"] = wrong_guesses
        game_result["final_state"] = ''.join(current_state)

        # 🎯 计算游戏级别的step-wise指标
        game_result["game_stats"] = {
            "step_accuracy": game_result["correct_steps"] / max(game_result["total_steps"], 1),
            "avg_prob_alignment": np.mean(game_result["step_prob_alignments"]) if game_result[
                "step_prob_alignments"] else 0.0,
            "avg_kl_divergence": np.mean(game_result["step_kl_divergences"]) if game_result[
                "step_kl_divergences"] else 0.0,
            "avg_confidence": np.mean(game_result["step_confidences"]) if game_result["step_confidences"] else 0.0,
            "violation_count": len(game_result["violations"]),
            "efficiency_score": self._calculate_efficiency_score(game_result)
        }

        return game_result

    def _build_hangman_prompt(self, current_state: List[str], guessed_letters: Set[str],
                              wrong_guesses: int) -> str:
        """🔧 构建标准Hangman prompt格式：word_state[SEP]guessed_letters[SEP]wrong/max"""
        # 第一部分：当前词状态
        word_state = ''.join(current_state)

        # 第二部分：已猜字母（按字母顺序）
        if guessed_letters:
            guessed_str = ','.join(sorted(guessed_letters))
        else:
            guessed_str = ''

        # 第三部分：错误次数
        wrong_str = f"{wrong_guesses}/6"

        # 🎯 标准Hangman格式
        prompt = f"{word_state}[SEP]{guessed_str}[SEP]{wrong_str}"

        return prompt

    def _predict_next_char(self, prompt: str, word: str, guessed_letters: Set[str],
                           strategy: str) -> Optional[Dict[str, Any]]:
        """🎯 预测下一个字符（支持多种策略）"""
        try:
            # 编码输入
            encoded = self._encode_prompt(prompt)
            if not encoded:
                return None

            input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

            with torch.no_grad():
                # 🔧 修复：处理ModelOutput对象
                outputs = self.model(x=input_ids)

                # 检查输出类型并提取logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    logits = outputs.last_hidden_state
                else:
                    logits = outputs

                # 确保logits形状正确
                if logits.dim() != 3:
                    print(f"❌ Unexpected logits shape: {logits.shape}")
                    return None

                last_logits = logits[0, -1, :]  # 最后位置的logits

                # 应用温度
                scaled_logits = last_logits / self.temperature

                # 提取a-z字符的logits
                a_z_indices = []
                valid_chars = []

                for c in self.a_z_chars:
                    if c in self.char_to_id:
                        a_z_indices.append(self.char_to_id[c])
                        valid_chars.append(c)

                if not a_z_indices:
                    return None

                a_z_logits = scaled_logits[a_z_indices]
                a_z_probs = F.softmax(a_z_logits, dim=-1)

                # 构建模型分布
                model_distribution = {}
                for i, char in enumerate(valid_chars):
                    model_distribution[char] = float(a_z_probs[i])

                # 为没有映射的字符填充极小值
                for c in self.a_z_chars:
                    if c not in model_distribution:
                        model_distribution[c] = 1e-8

                # 计算理想分布
                ideal_distribution = self._calculate_ideal_distribution(word, guessed_letters)

                # 计算概率对齐分数
                prob_alignment = self._calculate_probability_alignment(model_distribution, ideal_distribution)

                # 计算KL散度
                kl_divergence = self._calculate_kl_divergence(model_distribution, ideal_distribution)

                # 🎯 根据策略选择字符
                if strategy == "greedy":
                    pred_idx = torch.argmax(a_z_probs).item()
                elif strategy == "sampling":
                    pred_idx = torch.multinomial(a_z_probs, 1).item()
                elif strategy == "mixed":
                    if random.random() < 0.7:  # 70%贪心，30%采样
                        pred_idx = torch.argmax(a_z_probs).item()
                    else:
                        pred_idx = torch.multinomial(a_z_probs, 1).item()
                else:
                    pred_idx = torch.argmax(a_z_probs).item()

                predicted_char = valid_chars[pred_idx]
                confidence = float(a_z_probs[pred_idx])

                return {
                    "predicted_char": predicted_char,
                    "model_distribution": model_distribution,
                    "ideal_distribution": ideal_distribution,
                    "kl_divergence": kl_divergence,
                    "confidence": confidence,
                    "prob_alignment": prob_alignment,
                    "strategy_used": strategy
                }

        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None

    def _calculate_ideal_distribution(self, word: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """计算理想概率分布"""
        unguessed_chars = set(word) - guessed_letters

        if not unguessed_chars:
            return {c: 1.0 / 26 for c in self.a_z_chars}

        # 计算频率
        char_counts = Counter(c for c in word if c in unguessed_chars)
        total_unguessed = sum(char_counts.values())

        distribution = {}
        for c in self.a_z_chars:
            if c in char_counts:
                distribution[c] = char_counts[c] / total_unguessed * 0.8
            elif c in guessed_letters:
                distribution[c] = 0.01 / 26
            else:
                distribution[c] = 0.2 / 26

        # 归一化
        total_prob = sum(distribution.values())
        if total_prob > 0:
            distribution = {c: p / total_prob for c, p in distribution.items()}

        return distribution

    def _calculate_probability_alignment(self, model_dist: Dict[str, float],
                                         ideal_dist: Dict[str, float]) -> float:
        """计算概率分布对齐分数"""
        try:
            # 使用余弦相似度作为对齐分数
            model_probs = np.array([model_dist.get(c, 1e-8) for c in self.a_z_chars])
            ideal_probs = np.array([ideal_dist.get(c, 1e-8) for c in self.a_z_chars])

            # 归一化
            model_probs = model_probs / np.sum(model_probs)
            ideal_probs = ideal_probs / np.sum(ideal_probs)

            # 计算余弦相似度
            dot_product = np.dot(model_probs, ideal_probs)
            norm_model = np.linalg.norm(model_probs)
            norm_ideal = np.linalg.norm(ideal_probs)

            if norm_model > 0 and norm_ideal > 0:
                cosine_sim = dot_product / (norm_model * norm_ideal)
                return float(np.clip(cosine_sim, 0.0, 1.0))
            else:
                return 0.0

        except Exception as e:
            return 0.0

    def _calculate_kl_divergence(self, model_dist: Dict[str, float],
                                 ideal_dist: Dict[str, float]) -> float:
        """计算KL散度"""
        try:
            kl_div = 0.0
            for char in self.a_z_chars:
                p = ideal_dist.get(char, 1e-8)
                q = model_dist.get(char, 1e-8)

                p = max(p, 1e-8)
                q = max(q, 1e-8)

                kl_div += p * np.log(p / q)

            return float(kl_div)
        except:
            return 0.0

    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """计算分布的熵"""
        try:
            entropy = 0.0
            for char in self.a_z_chars:
                p = distribution.get(char, 1e-8)
                p = max(p, 1e-8)
                entropy -= p * np.log2(p)
            return float(entropy)
        except:
            return 0.0

    def _calculate_efficiency_score(self, game_result: Dict[str, Any]) -> float:
        """计算游戏效率分数"""
        if not game_result["success"]:
            return 0.0

        total_steps = game_result["total_steps"]
        word_length = len(set(game_result["word"]))  # 唯一字符数

        if total_steps == 0:
            return 0.0

        # 效率分数：越少步骤完成越好
        max_possible_steps = min(word_length, self.max_steps)
        efficiency = max_possible_steps / total_steps

        return min(efficiency, 1.0)

    def _encode_prompt(self, prompt: str) -> List[int]:
        """编码提示"""
        try:
            if hasattr(self.tokenizer, 'encode'):
                encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
                return encoded
            else:
                print(f"❌ Tokenizer does not have encode method")
                return []
        except Exception as e:
            print(f"❌ Error encoding prompt: {e}")
            return []

    # 其他方法保持相同，只是简化了不相关的部分...

    def simulate_batch(self, words: List[str],
                       games_per_word: int = 1,
                       strategy: str = "greedy") -> Dict[str, Any]:
        """批量模拟多个词汇的游戏"""
        all_results = []

        print(f"🎯 Simulating {len(words)} words × {games_per_word} games with strategy '{strategy}'...")

        for word_idx, word in enumerate(words):
            for game_idx in range(games_per_word):
                try:
                    result = self.simulate_game(word, strategy)
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error simulating '{word}' game {game_idx}: {e}")
                    continue

        # 计算批量统计
        batch_stats = self._calculate_batch_statistics(all_results)

        return {
            "individual_results": all_results,
            "batch_statistics": batch_stats,
            "simulation_config": {
                "strategy": strategy,
                "games_per_word": games_per_word,
                "total_words": len(words),
                "total_games": len(all_results),
                "temperature": self.temperature,
                "max_wrong_guesses": self.max_wrong_guesses,
                "max_steps": self.max_steps
            }
        }

    def _calculate_batch_statistics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🎯 计算批量统计，重点关注step-wise指标"""
        if not all_results:
            return {}

        # 基础统计
        total_games = len(all_results)
        successful_games = sum(1 for r in all_results if r["success"])

        # Step-wise统计
        all_step_accuracies = []
        all_prob_alignments = []
        all_kl_divergences = []
        all_confidences = []

        for result in all_results:
            game_stats = result["game_stats"]
            all_step_accuracies.append(game_stats["step_accuracy"])
            all_prob_alignments.append(game_stats["avg_prob_alignment"])
            all_kl_divergences.append(game_stats["avg_kl_divergence"])
            all_confidences.append(game_stats["avg_confidence"])

        # 计算综合统计
        stats = {
            # 🎯 Step-wise核心指标
            "step_accuracy": np.mean(all_step_accuracies),
            "step_accuracy_std": np.std(all_step_accuracies),
            "avg_prob_alignment": np.mean(all_prob_alignments),
            "avg_kl_divergence": np.mean(all_kl_divergences),
            "avg_confidence": np.mean(all_confidences),

            # 游戏级别指标
            "game_success_rate": successful_games / total_games,
            "total_games": total_games,
            "successful_games": successful_games,
        }

        return stats


# 向后兼容的包装器
class HangmanSimulator(StepwiseProbabilityHangmanSimulator):
    """向后兼容的模拟器"""

    def __init__(self, model, tokenizer, device='cuda', **kwargs):
        # 提取传统参数
        temperature = kwargs.get('temperature', 1.0)
        max_wrong_guesses = kwargs.get('max_wrong_guesses', 6)
        max_steps = kwargs.get('max_steps', 10)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=temperature,
            max_wrong_guesses=max_wrong_guesses,
            max_steps=max_steps,
            record_probabilities=True
        )

    def simulate(self, word: str) -> Dict[str, Any]:
        """向后兼容的模拟方法"""
        result = self.simulate_game(word, strategy="greedy")

        # 转换为旧格式
        return {
            "word": result["word"],
            "success": result["success"],
            "steps": result["total_steps"],
            "wrong_guesses": result["wrong_guesses"],
            "guesses": [step["predicted_char"] for step in result["steps"]],
            "final_state": result["final_state"],
            # 新增step-wise指标
            "step_accuracy": result["game_stats"]["step_accuracy"],
            "prob_alignment": result["game_stats"]["avg_prob_alignment"]
        }