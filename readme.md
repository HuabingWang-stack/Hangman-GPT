# Hangman Game GPT - Character-Level GPT with Supervised Fine-Tuning

## 🎮 Project Overview

This project implements an GPT system for playing the Hangman word-guessing game using a character-level GPT model with supervised fine-tuning (SFT). The system learns to predict the next best character to guess based on the current game state, using a two-stage training approach: pretraining on word datasets followed by fine-tuning on game-specific data with soft targets.

## 🎯 Task Description

The Hangman GPT needs to:
- Analyze the current word state (e.g., "a_p_e")
- Consider already guessed characters
- Track the number of wrong guesses
- Predict the most likely character that appears in the word

The model outputs a probability distribution over characters a-z, with higher probabilities for characters more likely to be in the word.

## 📊 Dataset

The project uses a custom dataset with the following structure:
```dataset/225k_10k_5k_10k/
├── pretrain.txt    # 225,000 words for pretraining
├── sft.txt         # 10,000 words for SFT data generation
├── grpo.txt        # 5,000 words for future GRPO training
└── test.txt        # 10,000 words for evaluation
```


Total vocabulary: ~225,000 unique English words

## 🏗️ Model Architecture
Character-Level GPT Model:

Model dimension: 256
Layers: 8 transformer layers
Attention heads: 8
Max sequence length: 64
Vocabulary size: 42 (a-z + special tokens)
Total parameters: ~4M
Enhanced Character Tokenizer:

Supports lowercase letters (a-z)
Special characters: `_, ,, :, |, ` 
Control tokens: <pad>, <bos>, <eos>, <unk>
Custom tokens: `[SEP], [MASK], [CLS]`
🚀 Training Pipeline
## Step 1: Pretraining
Train a character-level GPT model on word data for next-character prediction:
```
python trainer/pretrain.py \
    --data dataset/225k_10k_5k_10k/pretrain.txt \
    --tokenizer scripts/tokenizer/tokenizer.json \
    --out checkpoints/pretrain.pth \
    --d-model 256 \
    --n-layer 8 \
    --n-head 8 \
    --max-seq-len 64 \
    --batch-size 512 \
    --lr 1e-3 \
    --epochs 70 \
    --experiment-name "hangman-pretrain"`
```

## Step 2: Generate SFT Training Data
Create supervised fine-tuning data from game simulations:
```
python sft/generate_guess.py \
    --max_samples 4 \
    --no_enhance \
    --word_dir dataset/225k_10k_5k_10k \
    --out_dir sft/data/final
```

### Generated Data Format:

`
{
    "guess": {
        "prompt": "a_p_e[SEP]a,e,p,t[SEP]1/6",
        "completion": "l"
    },
    "label": "apple",
    "soft_targets": {"l": 0.666667, "p": 0.333333}
}
`

## Step 3: Supervised Fine-Tuning
Fine-tune the pretrained model on Hangman-specific data:

```
python sft/sft_train.py \
    --pretrain_model checkpoints/final/pretrain.pth \
    --train_data sft/data/train.jsonl \
    --test_data sft/data/test.jsonl \
    --batch_size 2048 \
    --eval_batch_size 2048 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --num_epochs 20 \
    --eval_steps 500 \
    --logging_steps 50 \
    --dataloader_num_workers 6 \
    --max_seq_length 64 \
    --no_early_stopping \
    --no_load_best \
    --experiment_name "Hangman-SFT" \
    --output_dir sft/models/4xdataset-8layer/
```

## Step 4: Evaluation
Test the trained model on unseen words:

```
python test/sft_simulator.py \
    --num_games 2000 \
    --model_path sft/models/4xdataset-8layer/checkpoint-7400 \
    --test_set dataset/225k_10k_5k_10k/grpo.txt
```

## 📝 Output Structure

```

checkpoints/
├── pretrain.pth            # Pretrained model
└── final/                  # Best models

sft/
├── data/
│   ├── train.jsonl         # SFT training data
│   ├── test.jsonl          # SFT test data
│   └── *_with_metadata.jsonl  # Data with full metadata
└── models/
    └── 4xdataset-8layer/   # Fine-tuned models
        └── checkpoint-*/    # Training checkpoints

```

## 📈 Key Features

1. **Two-Stage Training**: Pretraining for language understanding + fine-tuning for game strategy
2. **Soft Targets**: Continuous probability distributions instead of hard labels
3. **Temperature Smoothing**: Adjustable distribution sharpness
4. **Comprehensive Logging**: Wandb integration with detailed metrics and samples
5. **Efficient Data Generation**: Multiprocess support for large-scale data creation
6. **Flexible Architecture**: Configurable model size and training parameters

## 🎯 Performance
Expected results on 5,000-word test set:
Win Rate: ~56%

## 🚀 Quick Start
Setup Environment:
```pip install -r requirements/txt```