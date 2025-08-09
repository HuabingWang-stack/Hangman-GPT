# evaluator.py - Step-wise准确率评估器（完整修复版）
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter


class StepWiseHangmanEvaluator:
    """基于Step-wise准确率的Hangman评估器"""

    def __init__(self, model, tokenizer, device='cuda',
                 temperature: float = 1.0,
                 max_steps: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_steps = max_steps

        # 🎯 预计算a-z字符映射 - 基于实际tokenizer结构
        self.a_z_chars = 'abcdefghijklmnopqrstuvwxyz'
        self.char_to_id = {}
        self.id_to_char = {}
        self._initialize_char_mappings()

        print(f"🎯 StepWiseHangmanEvaluator initialized:")
        print(f"   Device: {device}")
        print(f"   Temperature: {temperature}")
        print(f"   Max steps: {max_steps}")
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

    def evaluate(self, word_list: List[str], num_games_per_word: int = 1) -> Dict[str, Any]:
        """
        评估模型在单步准确率上的表现

        Returns:
            {
                'step_accuracy': float,  # 单步准确率
                'game_success_rate': float,  # 游戏胜率（向后兼容）
                'avg_steps_per_game': float,  # 平均步数
                'avg_kl_divergence': float,  # 平均KL散度
                'avg_prediction_confidence': float,  # 平均预测置信度
                'total_games': int,  # 总游戏数
                'total_steps': int,  # 总步数
                'correct_steps': int,  # 正确步数
                'detailed_stats': Dict,  # 详细统计
            }
        """
        self.model.eval()

        # 统计变量
        total_games = 0
        successful_games = 0
        total_steps = 0
        correct_steps = 0
        total_kl_divergence = 0.0
        total_confidence = 0.0
        step_count = 0

        # 详细统计
        word_length_stats = defaultdict(list)
        step_position_accuracy = defaultdict(list)
        kl_divergence_history = []
        confidence_history = []

        game_results = []

        print(f"🎯 Starting step-wise evaluation on {len(word_list)} words...")

        for word_idx, word in enumerate(word_list):
            word = word.lower()

            for game_idx in range(num_games_per_word):
                try:
                    game_result = self._evaluate_single_game(word, word_idx, game_idx)

                    if game_result:
                        game_results.append(game_result)
                        total_games += 1

                        # 游戏级别统计
                        if game_result['success']:
                            successful_games += 1

                        # Step级别统计
                        game_steps = game_result['total_steps']
                        game_correct = game_result['correct_steps']

                        total_steps += game_steps
                        correct_steps += game_correct

                        # KL散度和置信度
                        if game_result['avg_kl_divergence'] > 0:
                            total_kl_divergence += game_result['avg_kl_divergence']
                            kl_divergence_history.append(game_result['avg_kl_divergence'])

                        if game_result['avg_confidence'] > 0:
                            total_confidence += game_result['avg_confidence']
                            confidence_history.append(game_result['avg_confidence'])

                        step_count += 1

                        # 词长度统计
                        word_length_stats[len(word)].append(game_correct / max(game_steps, 1))

                        # 步骤位置准确率
                        for step_info in game_result['step_details']:
                            pos = step_info['step_position']
                            is_correct = step_info['is_correct']
                            step_position_accuracy[pos].append(1.0 if is_correct else 0.0)

                except Exception as e:
                    print(f"❌ Error evaluating word '{word}' game {game_idx}: {e}")
                    continue

            # 进度报告
            if (word_idx + 1) % 20 == 0:
                current_step_acc = correct_steps / max(total_steps, 1)
                current_game_acc = successful_games / max(total_games, 1)
                print(f"   Progress: {word_idx + 1}/{len(word_list)} words, "
                      f"step_acc={current_step_acc:.3f}, game_acc={current_game_acc:.3f}")

        # 计算最终指标
        step_accuracy = correct_steps / max(total_steps, 1)
        game_success_rate = successful_games / max(total_games, 1)
        avg_steps_per_game = total_steps / max(total_games, 1)
        avg_kl_divergence = total_kl_divergence / max(step_count, 1)
        avg_confidence = total_confidence / max(step_count, 1)

        # 详细统计
        detailed_stats = {
            'word_length_accuracy': {
                length: np.mean(accs) for length, accs in word_length_stats.items()
            },
            'step_position_accuracy': {
                pos: np.mean(accs) for pos, accs in step_position_accuracy.items()
            },
            'kl_divergence_std': np.std(kl_divergence_history) if kl_divergence_history else 0.0,
            'confidence_std': np.std(confidence_history) if confidence_history else 0.0,
            'total_violations': sum(gr.get('violations', 0) for gr in game_results),
        }

        results = {
            'step_accuracy': step_accuracy,
            'game_success_rate': game_success_rate,
            'avg_steps_per_game': avg_steps_per_game,
            'avg_kl_divergence': avg_kl_divergence,
            'avg_prediction_confidence': avg_confidence,
            'total_games': total_games,
            'total_steps': total_steps,
            'correct_steps': correct_steps,
            'successful_games': successful_games,
            'detailed_stats': detailed_stats,

            # 向后兼容字段
            'success_rate': game_success_rate,
            'avg_steps': avg_steps_per_game,
            'avg_reward': step_accuracy * 10 - 5,  # 简单的reward映射
        }

        print(f"\n🎯 Step-wise Evaluation Results:")
        print(f"   Step Accuracy: {step_accuracy:.1%} ({correct_steps}/{total_steps})")
        print(f"   Game Success Rate: {game_success_rate:.1%} ({successful_games}/{total_games})")
        print(f"   Avg Steps per Game: {avg_steps_per_game:.2f}")
        print(f"   Avg KL Divergence: {avg_kl_divergence:.4f}")
        print(f"   Avg Prediction Confidence: {avg_confidence:.3f}")

        return results

    def _evaluate_single_game(self, word: str, word_idx: int, game_idx: int) -> Dict[str, Any]:
        """评估单个游戏"""
        guessed_letters = set()
        wrong_guesses = 0
        max_wrong = 6

        game_result = {
            'word': word,
            'word_idx': word_idx,
            'game_idx': game_idx,
            'success': False,
            'total_steps': 0,
            'correct_steps': 0,
            'wrong_guesses': 0,
            'step_details': [],
            'avg_kl_divergence': 0.0,
            'avg_confidence': 0.0,
            'violations': 0,
            'failure_reason': ''
        }

        # 初始化游戏状态
        current_state = ['_' if c.isalpha() else c for c in word]
        kl_divergences = []
        confidences = []

        for step_idx in range(self.max_steps):
            # 🔧 修复：使用正确的Hangman prompt格式
            prompt = self._build_hangman_prompt(current_state, guessed_letters, wrong_guesses)

            # 🎯 预测并记录概率分布
            prediction_result = self._predict_with_probabilities(prompt, word, guessed_letters)

            if not prediction_result:
                game_result['failure_reason'] = 'prediction_failed'
                break

            predicted_char = prediction_result['predicted_char']
            model_dist = prediction_result.get('model_distribution', {})
            ideal_dist = prediction_result.get('ideal_distribution', {})
            kl_div = prediction_result.get('kl_divergence', 0.0)
            confidence = prediction_result.get('confidence', 0.0)

            # 检查违规
            is_violation = predicted_char in guessed_letters
            if is_violation:
                game_result['violations'] += 1

            # 判断是否正确
            is_correct = predicted_char in word and not is_violation

            # 记录step详情
            step_detail = {
                'step_position': step_idx,
                'predicted_char': predicted_char,
                'is_correct': is_correct,
                'is_violation': is_violation,
                'confidence': confidence,
                'kl_divergence': kl_div,
                'prompt': prompt,
                'current_state': ''.join(current_state),
                'model_distribution': model_dist,
                'ideal_distribution': ideal_dist
            }
            game_result['step_details'].append(step_detail)

            # 更新统计
            game_result['total_steps'] += 1
            if is_correct:
                game_result['correct_steps'] += 1

            if kl_div > 0:
                kl_divergences.append(kl_div)
            if confidence > 0:
                confidences.append(confidence)

            # 更新游戏状态
            guessed_letters.add(predicted_char)

            if predicted_char in word:
                # 猜对了
                for i, c in enumerate(word):
                    if c == predicted_char:
                        current_state[i] = c

                # 检查游戏是否完成
                if '_' not in current_state:
                    game_result['success'] = True
                    break
            else:
                # 猜错了
                wrong_guesses += 1
                game_result['wrong_guesses'] = wrong_guesses
                if wrong_guesses >= max_wrong:
                    game_result['failure_reason'] = 'too_many_wrong'
                    break

            # 违规也算失败
            if is_violation:
                game_result['failure_reason'] = 'violation'
                break

        # 如果没有明确失败原因且未成功
        if not game_result['success'] and not game_result['failure_reason']:
            game_result['failure_reason'] = 'max_steps_reached'

        # 计算平均指标
        game_result['avg_kl_divergence'] = np.mean(kl_divergences) if kl_divergences else 0.0
        game_result['avg_confidence'] = np.mean(confidences) if confidences else 0.0

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

    def _predict_with_probabilities(self, prompt: str, word: str,
                                    guessed_letters: Set[str]) -> Dict[str, Any]:
        """🎯 带概率分布的预测"""
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

                # 计算KL散度
                kl_divergence = self._calculate_kl_divergence(model_distribution, ideal_distribution)

                # 预测字符（贪心策略用于评估）
                pred_idx = torch.argmax(a_z_probs).item()
                predicted_char = valid_chars[pred_idx]
                confidence = float(a_z_probs[pred_idx])

                return {
                    'predicted_char': predicted_char,
                    'model_distribution': model_distribution,
                    'ideal_distribution': ideal_distribution,
                    'kl_divergence': kl_divergence,
                    'confidence': confidence
                }

        except Exception as e:
            print(f"❌ Error in probability prediction: {e}")
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

    def quick_evaluate(self, word_list: List[str], max_words: int = 50) -> Dict[str, Any]:
        """快速评估（用于训练期间）"""
        limited_words = word_list[:max_words] if len(word_list) > max_words else word_list
        return self.evaluate(limited_words, num_games_per_word=1)

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

                    # 🔧 修复：处理ModelOutput对象
                    outputs = model(x=input_ids, attn_mask=attention_mask)

                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs

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
                                a_z_indices = [self.tokenizer.char_to_id[c] for c in
                                               'abcdefghijklmnopqrstuvwxyz'
                                               if c in self.tokenizer.char_to_id]

                                # 创建mask
                                masked_logits = torch.full_like(pred_logits, float('-inf'))
                                if len(a_z_indices) > 0:
                                    masked_logits[a_z_indices] = pred_logits[a_z_indices]

                                pred_char_id = torch.argmax(masked_logits).item()
                                pred_char = self.tokenizer.decode([pred_char_id])

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


# 向后兼容的包装器
class HangmanEvaluator(StepWiseHangmanEvaluator):
    """向后兼容的评估器"""
    pass