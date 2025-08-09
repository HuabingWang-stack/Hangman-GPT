# custom_model.py - 清理版本
import torch
import torch.nn as nn
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.tokenizer import EnhancedCharacterTokenizer
from trainer.pretrain import GPT, ModelConfig, load_pretrained_model


class ModelOutput:
    def __init__(self, logits):
        self.logits = logits


def patch_gpt_forward():
    original_forward = GPT.forward

    def wrapped_forward(self, x, attn_mask=None):
        result = original_forward(self, x, attn_mask)

        if isinstance(result, torch.Tensor):
            return ModelOutput(result)
        elif isinstance(result, dict):
            if 'logits' in result:
                return ModelOutput(result['logits'])
            else:
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        return ModelOutput(value)
                return result
        elif hasattr(result, 'logits'):
            return result
        else:
            try:
                return ModelOutput(result)
            except:
                return result

    GPT.forward = wrapped_forward


def load_sft_model(model_path: str, tokenizer_path: str):
    try:
        tokenizer = EnhancedCharacterTokenizer(config_path=tokenizer_path)
    except Exception as e:
        raise

    patch_gpt_forward()

    pth_path = os.path.join(model_path, "model.pth")
    safetensors_path = os.path.join(model_path, "model.safetensors")

    try:
        if os.path.exists(safetensors_path):
            try:
                from safetensors import safe_open
            except ImportError:
                os.system("pip install safetensors")
                from safetensors import safe_open

            model = GPT(vocab_size=tokenizer.vocab_size)

            state_dict = {}
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            model.load_state_dict(state_dict)

        elif os.path.exists(pth_path):
            model, _, model_config = load_pretrained_model(pth_path, tokenizer_path)
        else:
            model = GPT(vocab_size=tokenizer.vocab_size)

    except Exception as e:
        model = GPT(vocab_size=tokenizer.vocab_size)

    return model, tokenizer


class TokenizerWrapper:
    def __init__(self, tokenizer: EnhancedCharacterTokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        self._char_to_id_dict = tokenizer.char_to_id
        self._id_to_char_dict = tokenizer.id_to_char

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        try:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except:
            return []

    def decode(self, token_ids: list, skip_special_tokens: bool = False) -> str:
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except:
            return ""

    def char_to_id(self, char: str) -> int:
        try:
            if isinstance(self._char_to_id_dict, dict):
                return self._char_to_id_dict.get(char, self._char_to_id_dict.get('<unk>', 1))
            return 1
        except:
            return 1

    def id_to_char(self, char_id: int) -> str:
        try:
            if isinstance(self._id_to_char_dict, dict):
                return self._id_to_char_dict.get(char_id, '<unk>')
            return '<unk>'
        except:
            return '<unk>'

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def save_config(self, save_path: str):
        try:
            return self.tokenizer.save_config(save_path)
        except:
            return None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def load_sft_model_with_wrappers(model_path: str, tokenizer_path: str):
    model, tokenizer = load_sft_model(model_path, tokenizer_path)
    wrapped_tokenizer = TokenizerWrapper(tokenizer)
    return model, wrapped_tokenizer


def save_model(model, tokenizer, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    model_file = os.path.join(save_path, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
    }, model_file)

    tokenizer_file = os.path.join(save_path, "tokenizer.json")
    tokenizer.save_config(tokenizer_file)