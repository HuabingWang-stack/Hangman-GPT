# dataset.py
import random
from typing import List, Set


class HangmanDataset:
    """管理词库，提供训练/验证数据"""

    def __init__(self, pretrain_path: str, sft_path: str, grpo_path: str):
        self.train_words = self._load_words([pretrain_path, sft_path])
        self.eval_words = self._load_words([grpo_path])

        print(f"Loaded {len(self.train_words)} training words")
        print(f"Loaded {len(self.eval_words)} evaluation words")

    def _load_words(self, paths: List[str]) -> Set[str]:
        """从文件加载词汇"""
        words = set()
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word and word.isalpha():  # 只保留字母组成的词
                            words.add(word)
            except FileNotFoundError:
                print(f"Warning: File {path} not found")
        return words

    def get_train_batch(self, batch_size: int) -> List[str]:
        """返回一批训练用的words"""
        return random.sample(list(self.train_words), min(batch_size, len(self.train_words)))

    def get_eval_batch(self, batch_size: int) -> List[str]:
        """返回一批验证用的words"""
        return random.sample(list(self.eval_words), min(batch_size, len(self.eval_words)))

    def get_random_train_word(self) -> str:
        """随机获取一个训练词"""
        return random.choice(list(self.train_words))