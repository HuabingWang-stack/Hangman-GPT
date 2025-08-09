# grpo_trainer.py - 删除Probability Alignment后的清洁版本
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np


class StepwiseProbabilityWordClassifier:
    """基于step-wise准确率的词汇难度分类器（简化版）"""

    def __init__(self):
        self.performance_history = {}
        self.update_count = {}

    def classify_word(self, word: str, trajectories: List[Dict]) -> str:
        """基于step-wise准确率分类词汇难度"""
        # 计算当前step-wise指标
        all_step_accuracies = []

        for traj in trajectories:
            step_accuracies = traj.get("step_accuracies", [])
            all_step_accuracies.extend([1.0 if acc else 0.0 for acc in step_accuracies])

        current_step_accuracy = np.mean(all_step_accuracies) if all_step_accuracies else 0.0
        word_length = len(word)

        # 结合历史性能
        if word in self.performance_history:
            historical_score = self.performance_history[word]
            hist_weight = min(0.5, self.update_count[word] * 0.1)
            final_score = (1 - hist_weight) * current_step_accuracy + hist_weight * historical_score
        else:
            final_score = current_step_accuracy

        # 基于分数的分类逻辑
        if final_score >= 0.75:
            return "PROTECT"  # 高性能，需要保护
        elif final_score >= 0.45:
            return "IMPROVE"  # 中等性能，温和提升
        elif word_length <= 5:
            return "FOCUS"  # 短词低性能，重点攻克
        else:
            return "IMPROVE"  # 长词低性能，温和提升

    def update_history(self, word: str, step_accuracy: float):
        """更新词汇历史性能"""
        if word in self.performance_history:
            self.performance_history[word] = 0.8 * self.performance_history[word] + 0.2 * step_accuracy
            self.update_count[word] += 1
        else:
            self.performance_history[word] = step_accuracy
            self.update_count[word] = 1

    def get_stats(self) -> Dict[str, Any]:
        """获取分类器统计信息"""
        if not self.performance_history:
            return {"total_words": 0}

        categories = {}
        for word, score in self.performance_history.items():
            if score >= 0.75:
                cat = "PROTECT"
            elif score >= 0.45:
                cat = "IMPROVE"
            elif len(word) <= 5:
                cat = "FOCUS"
            else:
                cat = "IMPROVE"
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_words": len(self.performance_history),
            "protect_words": categories.get("PROTECT", 0),
            "focus_words": categories.get("FOCUS", 0),
            "improve_words": categories.get("IMPROVE", 0),
            "avg_score": np.mean(list(self.performance_history.values()))
        }


class AdaptiveKLScheduler:
    """自适应KL调度器"""

    def __init__(self, base_kl_coeff=0.08):
        self.base_kl_coeff = base_kl_coeff
        self.current_kl_coeff = base_kl_coeff
        self.recent_kl_losses = []
        self.recent_policy_losses = []

    def update_kl_coeff(self, epoch_stats):
        """根据训练情况自适应调整KL系数"""
        avg_kl_loss = epoch_stats.get('avg_kl_loss', 0.0)
        avg_policy_loss = epoch_stats.get('avg_policy_loss', 0.0)

        self.recent_kl_losses.append(avg_kl_loss)
        self.recent_policy_losses.append(avg_policy_loss)

        if len(self.recent_kl_losses) > 5:
            self.recent_kl_losses.pop(0)
            self.recent_policy_losses.pop(0)

        old_kl_coeff = self.current_kl_coeff

        if avg_kl_loss < 0.002:
            self.current_kl_coeff *= 0.85
            reason = "KL too small, reducing constraint"
        elif avg_policy_loss > 0.8:
            self.current_kl_coeff *= 1.25
            reason = "Policy loss too large, increasing constraint"
        else:
            reason = "No adjustment needed"

        self.current_kl_coeff = np.clip(self.current_kl_coeff, 0.01, 0.3)

        if abs(self.current_kl_coeff - old_kl_coeff) > 1e-6:
            print(f"🔧 KL coeff adjusted: {old_kl_coeff:.4f} → {self.current_kl_coeff:.4f} ({reason})")

        return self.current_kl_coeff


class CleanStepwisePPOTrainer:
    """清洁的Step-wise PPO训练器（删除冗余KL对齐）"""

    def __init__(self, model, reference_model, tokenizer, optimizer,
                 kl_coeff=0.08, clip_epsilon=0.15, device='cuda',
                 max_grad_norm=0.5, base_learning_rate=1e-4,
                 reward_calculator=None):

        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.kl_coeff = kl_coeff
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.base_learning_rate = base_learning_rate
        self.eps = 1e-8

        # 奖励计算器
        self.reward_calculator = reward_calculator

        # 词汇分类器和KL调度器
        self.difficulty_classifier = StepwiseProbabilityWordClassifier()
        self.kl_scheduler = AdaptiveKLScheduler(kl_coeff)

        # 字符映射
        self.a_z_chars = 'abcdefghijklmnopqrstuvwxyz'
        self.char_to_id = {}
        self._initialize_char_mappings()

        # 训练统计
        self.training_stats = {
            "protect_words_trained": 0,
            "focus_words_trained": 0,
            "improve_words_trained": 0,
            "total_updates": 0,
            "total_steps_trained": 0,
            "total_correct_steps": 0,
            "extreme_loss_count": 0,
            "ppo_computation_failures": 0
        }

        print(f"🎯 Clean Step-wise PPO Trainer initialized:")
        print(f"   A-Z chars mapped: {len(self.char_to_id)}/26")
        print(f"   KL coefficient: {kl_coeff}")
        print(f"   Clip epsilon: {clip_epsilon}")

    def _initialize_char_mappings(self):
        """初始化字符映射"""
        try:
            if hasattr(self.tokenizer, 'char_to_id') and isinstance(self.tokenizer.char_to_id, dict):
                for c in self.a_z_chars:
                    if c in self.tokenizer.char_to_id:
                        self.char_to_id[c] = self.tokenizer.char_to_id[c]
                print("✅ Using tokenizer.char_to_id (dict)")
                return
            else:
                print("⚠️ Tokenizer char_to_id not found, using fallback mapping")
                self._create_fallback_mappings()
        except Exception as e:
            print(f"❌ Error initializing character mappings: {e}")
            self._create_fallback_mappings()

    def _create_fallback_mappings(self):
        """创建fallback字符映射"""
        print("🔧 Creating fallback character mappings...")
        for i, c in enumerate(self.a_z_chars):
            self.char_to_id[c] = 4 + i  # 跳过控制tokens

    def train_step_clean_ppo(self, trajectory_groups_batch: List[Dict]) -> Dict[str, Any]:
        """清洁的Step-wise PPO训练步骤（删除冗余对齐）"""
        try:
            if not trajectory_groups_batch:
                return {"total_loss": 0.0, "backprop_success": False}

            print(f"🔄 Clean Step-wise PPO Training: Processing {len(trajectory_groups_batch)} trajectory groups")

            # 使用reward calculator计算奖励
            processed_groups = []
            for group in trajectory_groups_batch:
                if self.reward_calculator:
                    processed_group = self.reward_calculator.calculate_trajectory_group_rewards(group)
                    processed_groups.append(processed_group)
                else:
                    processed_groups.append(group)

            all_policy_losses = []
            all_kl_losses = []
            word_categories = {}
            category_stats = {"PROTECT": 0, "FOCUS": 0, "IMPROVE": 0}
            total_steps = 0
            total_correct_steps = 0
            valid_step_updates = 0

            for group_idx, processed_group in enumerate(processed_groups):
                trajectories = processed_group.get("trajectories", [])
                word = processed_group.get("word", trajectory_groups_batch[group_idx]["word"])

                # 分类词汇
                category = self.difficulty_classifier.classify_word(word, trajectories)
                word_categories[word] = category
                category_stats[category] += 1

                print(f"🎯 Processing group {group_idx + 1}: '{word}' (Category: {category}, {len(trajectories)} trajectories)")

                # 根据类别调整学习率
                self._adjust_learning_rate(category)

                # 计算step-wise advantages
                if self.reward_calculator:
                    advantages_list = self.reward_calculator.get_stepwise_advantages(
                        {"word": word, "trajectories": trajectories},
                        advantage_mode="combined"
                    )
                else:
                    advantages_list = self._fallback_advantages(trajectories)

                if not advantages_list:
                    print(f"   ⚠️ No valid advantages for '{word}', skipping")
                    continue

                # 根据类别设置KL权重
                kl_weight = self._get_kl_weight(category)

                # 计算step-wise policy losses
                group_policy_losses = []
                group_kl_losses = []
                group_steps = 0
                group_correct_steps = 0

                for traj_idx, (traj, step_advantages) in enumerate(zip(trajectories, advantages_list)):
                    step_accuracies = traj.get('step_accuracies', [])
                    steps = traj.get('steps', [])

                    group_steps += len(step_accuracies)
                    group_correct_steps += sum(step_accuracies)

                    # 对每一步计算PPO loss
                    for step_idx, (step, advantage) in enumerate(zip(steps, step_advantages)):
                        step_policy_loss, step_kl_loss = self._compute_step_ppo_loss(
                            step, advantage, traj_idx, step_idx, word, category
                        )

                        if step_policy_loss is not None:
                            group_policy_losses.append(step_policy_loss)
                            group_kl_losses.append(step_kl_loss * kl_weight)
                            valid_step_updates += 1
                        else:
                            self.training_stats["ppo_computation_failures"] += 1

                total_steps += group_steps
                total_correct_steps += group_correct_steps

                if group_policy_losses:
                    group_avg_policy = torch.stack(group_policy_losses).mean()
                    group_avg_kl = torch.stack(group_kl_losses).mean()

                    # 应用类别权重
                    category_weight = self._get_category_weight(category)
                    weighted_policy_loss = group_avg_policy * category_weight

                    # 检查loss异常
                    if torch.abs(weighted_policy_loss) > 3.0:
                        print(f"   ⚠️ Extreme loss detected for '{word}' ({category}): {weighted_policy_loss.item():.4f}, clamping")
                        weighted_policy_loss = torch.clamp(weighted_policy_loss, -3.0, 3.0)
                        self.training_stats["extreme_loss_count"] += 1

                    all_policy_losses.append(weighted_policy_loss)
                    all_kl_losses.append(group_avg_kl)

                    step_accuracy = group_correct_steps / max(group_steps, 1)
                    print(f"   ✅ Group '{word}' ({category}): {valid_step_updates} step updates, "
                          f"step_accuracy={step_accuracy:.1%}, "
                          f"policy_loss={group_avg_policy.item():.4f}, kl_loss={group_avg_kl.item():.4f}")
                else:
                    print(f"   ❌ No valid policy losses for '{word}' - all steps failed")

                # 更新历史性能
                if group_steps > 0:
                    current_step_accuracy = group_correct_steps / group_steps
                    self.difficulty_classifier.update_history(word, current_step_accuracy)

            if not all_policy_losses:
                print("❌ No valid step-wise PPO losses computed")
                return {
                    "total_loss": 0.0,
                    "backprop_success": False,
                    "valid_step_updates": 0,
                    "ppo_computation_failures": self.training_stats["ppo_computation_failures"]
                }

            # 合并losses - 只有Policy Loss + KL Loss
            total_policy_loss = torch.stack(all_policy_losses).mean()
            total_kl_loss = torch.stack(all_kl_losses).mean()

            # 最终损失：纯PPO
            current_kl_coeff = self.kl_scheduler.current_kl_coeff
            total_loss = total_policy_loss + current_kl_coeff * total_kl_loss

            # 安全检查
            if torch.isnan(total_loss) or torch.isinf(total_loss) or torch.abs(total_loss) > 15.0:
                print(f"❌ Invalid final loss: {total_loss}, skipping update")
                return {"total_loss": 0.0, "backprop_success": False}

            # 执行反向传播
            success = self._perform_safe_backprop(total_loss)

            # 更新训练统计
            self.training_stats["total_updates"] += 1
            self.training_stats["total_steps_trained"] += total_steps
            self.training_stats["total_correct_steps"] += total_correct_steps
            for category, count in category_stats.items():
                self.training_stats[f"{category.lower()}_words_trained"] += count

            overall_step_accuracy = total_correct_steps / max(total_steps, 1)

            return {
                "total_loss": total_loss,
                "policy_loss": total_policy_loss,
                "kl_loss": total_kl_loss,
                "current_kl_coeff": current_kl_coeff,
                "backprop_success": success,
                "total_steps": total_steps,
                "total_correct_steps": total_correct_steps,
                "overall_step_accuracy": overall_step_accuracy,
                "valid_step_updates": valid_step_updates,
                "word_categories": word_categories,
                "category_stats": category_stats,
                "extreme_loss_count": self.training_stats["extreme_loss_count"],
                "ppo_computation_failures": self.training_stats["ppo_computation_failures"]
            }

        except Exception as e:
            print(f"❌ Error in clean step-wise PPO training step: {e}")
            import traceback
            traceback.print_exc()
            return {"total_loss": 0.0, "backprop_success": False}

    def _compute_step_ppo_loss(self, step: Dict, advantage: float,
                               traj_idx: int, step_idx: int, word: str, category: str) -> Tuple:
        """计算单个步骤的PPO损失（纯PPO，无额外对齐）"""
        try:
            # 计算当前策略和参考策略的log概率
            current_log_prob = self._compute_step_log_prob(step, use_reference=False)
            reference_log_prob = self._compute_step_log_prob(step, use_reference=True)

            if current_log_prob is None or reference_log_prob is None:
                return None, None

            # 转换advantage为tensor
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(float(advantage), dtype=torch.float32, device=self.device)

            # 计算policy ratio
            log_ratio = current_log_prob - reference_log_prob
            log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
            ratio = torch.exp(log_ratio)

            # 根据类别调整clipping范围
            if category == "PROTECT":
                clip_epsilon = self.clip_epsilon * 0.7  # 更保守
            elif category == "FOCUS":
                clip_epsilon = self.clip_epsilon * 1.3  # 稍微放宽
            else:
                clip_epsilon = self.clip_epsilon

            # 标准PPO clipped policy loss
            policy_loss_1 = ratio * advantage
            policy_loss_2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            policy_loss = -torch.min(policy_loss_1, policy_loss_2)

            # 限制单步loss
            if category == "PROTECT":
                policy_loss = torch.clamp(policy_loss, -1.5, 1.5)
            elif category == "FOCUS":
                policy_loss = torch.clamp(policy_loss, -2.5, 2.5)
            else:
                policy_loss = torch.clamp(policy_loss, -2.0, 2.0)

            # 计算KL散度（只有Policy KL）
            kl_loss = self._compute_step_kl_divergence(step)
            if kl_loss is None:
                kl_loss = torch.tensor(0.0, device=self.device)

            # 调试信息
            if step_idx < 2 and traj_idx < 2:
                print(f"   Step {step_idx}: action='{step.get('action', 'unknown')}', "
                      f"advantage={advantage.item():.3f}, "
                      f"policy_loss={policy_loss.item():.4f}, "
                      f"kl_loss={kl_loss.item():.4f}, "
                      f"ratio={ratio.item():.3f}")

            return policy_loss, kl_loss

        except Exception as e:
            print(f"❌ Error computing step PPO loss for step {step_idx} in '{word}': {e}")
            return None, None

    def _compute_step_log_prob(self, step: Dict, use_reference: bool = False) -> torch.Tensor:
        """计算单个步骤的log概率"""
        try:
            model = self.reference_model if use_reference else self.model
            prompt = step["prompt"]
            action = step["action"]

            # 编码prompt
            encoded = self._encode_prompt(prompt)
            if not encoded:
                return None

            input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

            # 获取action token id
            action_token_id = self.char_to_id.get(action.lower(), None)
            if action_token_id is None:
                return None

            # 前向传播
            with torch.no_grad() if use_reference else torch.enable_grad():
                try:
                    outputs = model(x=input_ids)

                    # 检查输出类型并提取logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif hasattr(outputs, 'last_hidden_state'):
                        logits = outputs.last_hidden_state
                    else:
                        logits = outputs

                    if logits.dim() != 3:
                        return None

                    last_logits = logits[0, -1, :]
                    log_probs = F.log_softmax(last_logits, dim=-1)
                    action_log_prob = log_probs[action_token_id]

                    if torch.isnan(action_log_prob) or torch.isinf(action_log_prob):
                        return None

                    return action_log_prob

                except Exception as e:
                    return None

        except Exception as e:
            return None

    def _compute_step_kl_divergence(self, step: Dict) -> torch.Tensor:
        """计算单个步骤的KL散度（Policy KL only）"""
        try:
            prompt = step["prompt"]
            encoded = self._encode_prompt(prompt)
            if not encoded:
                return None

            input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

            try:
                # 当前策略
                current_outputs = self.model(x=input_ids)
                if hasattr(current_outputs, 'logits'):
                    current_logits = current_outputs.logits
                elif hasattr(current_outputs, 'last_hidden_state'):
                    current_logits = current_outputs.last_hidden_state
                else:
                    current_logits = current_outputs

                current_last_logits = current_logits[0, -1, :]
                current_log_probs = F.log_softmax(current_last_logits, dim=-1)

                # 参考策略
                with torch.no_grad():
                    ref_outputs = self.reference_model(x=input_ids)
                    if hasattr(ref_outputs, 'logits'):
                        ref_logits = ref_outputs.logits
                    elif hasattr(ref_outputs, 'last_hidden_state'):
                        ref_logits = ref_outputs.last_hidden_state
                    else:
                        ref_logits = ref_outputs

                    ref_last_logits = ref_logits[0, -1, :]
                    ref_probs = F.softmax(ref_last_logits, dim=-1)

                # 计算KL散度（只有Policy KL）
                step_kl = F.kl_div(current_log_probs, ref_probs, reduction='sum')

                if torch.isnan(step_kl) or torch.isinf(step_kl):
                    return None

                return step_kl

            except Exception as e:
                return None

        except Exception as e:
            return None

    def _encode_prompt(self, prompt: str) -> List[int]:
        """编码提示"""
        try:
            if hasattr(self.tokenizer, 'encode'):
                encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
                return encoded
            else:
                return []
        except Exception as e:
            return []

    def _fallback_advantages(self, trajectories: List[Dict]) -> List[List[float]]:
        """备用advantage计算"""
        trajectory_advantages = []
        for traj in trajectories:
            steps = traj.get('steps', [])
            step_accuracies = traj.get('step_accuracies', [True] * len(steps))

            advantages = []
            for is_correct in step_accuracies:
                advantage = 0.5 if is_correct else -0.5
                advantages.append(advantage)

            trajectory_advantages.append(advantages)

        return trajectory_advantages

    def _adjust_learning_rate(self, category: str):
        """根据类别调整学习率"""
        if category == "PROTECT":
            lr_multiplier = 0.6
        elif category == "FOCUS":
            lr_multiplier = 1.4
        else:
            lr_multiplier = 1.0

        new_lr = self.base_learning_rate * lr_multiplier
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _get_kl_weight(self, category: str) -> float:
        """获取KL权重"""
        current_kl_coeff = self.kl_scheduler.current_kl_coeff

        if category == "PROTECT":
            return current_kl_coeff * 1.5
        elif category == "FOCUS":
            return current_kl_coeff * 0.7
        else:
            return current_kl_coeff * 1.0

    def _get_category_weight(self, category: str) -> float:
        """获取类别权重"""
        if category == "FOCUS":
            return 1.5
        elif category == "PROTECT":
            return 0.7
        else:
            return 1.0

    def _perform_safe_backprop(self, total_loss) -> bool:
        """安全的反向传播"""
        try:
            if isinstance(total_loss, torch.Tensor) and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                self.optimizer.zero_grad()
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100.0:
                    print(f"❌ Extreme gradient norm detected: {grad_norm:.4f}, skipping update")
                    return False

                self.optimizer.step()
                print(f"✅ Clean Step-wise PPO Backprop successful, loss: {total_loss.item():.6f}, grad_norm: {grad_norm:.4f}")
                return True
            else:
                print(f"❌ Invalid loss for backprop: {total_loss}")
                return False

        except Exception as e:
            print(f"❌ Safe backprop failed: {e}")
            return False

    def update_epoch_stats(self, epoch_stats):
        """更新epoch统计信息并调整KL系数"""
        new_kl_coeff = self.kl_scheduler.update_kl_coeff(epoch_stats)
        return new_kl_coeff

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        classifier_stats = self.difficulty_classifier.get_stats()

        total_steps = self.training_stats["total_steps_trained"]
        total_correct = self.training_stats["total_correct_steps"]

        overall_step_accuracy = total_correct / max(total_steps, 1)

        return {
            **self.training_stats,
            **classifier_stats,
            "overall_step_accuracy": overall_step_accuracy,
            "current_kl_coeff": self.kl_scheduler.current_kl_coeff,
            "current_base_lr": self.base_learning_rate
        }

    def reset_training_stats(self):
        """重置训练统计"""
        self.training_stats = {
            "protect_words_trained": 0,
            "focus_words_trained": 0,
            "improve_words_trained": 0,
            "total_updates": 0,
            "total_steps_trained": 0,
            "total_correct_steps": 0,
            "extreme_loss_count": 0,
            "ppo_computation_failures": 0
        }


# 使用清洁版本的训练器
StepwisePPOTrainer = CleanStepwisePPOTrainer