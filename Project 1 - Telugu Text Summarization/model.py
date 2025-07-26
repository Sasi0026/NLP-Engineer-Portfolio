# =============================================================================
# STEP 1: IMPORT DEPENDENCIES
# =============================================================================

import os
import pandas as pd
import numpy as np
import nltk
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import gc

# =============================================================================
# STEP 2: ENVIRONMENT SETUP AND CONFIGURATION
# =============================================================================

def setup_environment():
    """Initialize reproducibility and optimize GPU memory"""
    set_seed(42)
    torch.cuda.empty_cache()
    gc.collect()
    print("✓ Environment setup completed")

def download_nltk_requirements():
    """Download required NLTK data"""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    print("✓ NLTK data downloaded")

# =============================================================================
# STEP 3: DATA LOADING AND PREPARATION
# =============================================================================

def load_datasets():
    """Load training, validation, and test datasets"""
    print("Loading datasets...")

    train_df = pd.read_json("/content/train.jsonl", lines=True)
    val_df = pd.read_json("/content/dev.jsonl", lines=True)
    test_df = pd.read_json("/content/test.jsonl", lines=True)

    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    """Filter and prepare data for training"""
    columns_to_use = ["cleaned_text", "summary"]

    train_df = train_df[columns_to_use].head(1500)
    val_df = val_df[columns_to_use].head(700)
    test_df = test_df[columns_to_use].head(100)

    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}")
    print(f"✓ Test samples: {len(test_df)}")

    return {
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    }

# =============================================================================
# STEP 4: MODEL AND TOKENIZER INITIALIZATION
# =============================================================================

def load_model_and_tokenizer():
    """Initialize mBART model and tokenizer for Telugu"""
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    print(f"Loading model: {model_checkpoint}")

    tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
    model = MBartForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer.src_lang = "te_IN"
    tokenizer.tgt_lang = "te_IN"

    print("✓ Model and tokenizer loaded successfully")
    return model, tokenizer

# =============================================================================
# STEP 5: DATA PREPROCESSING
# =============================================================================

def create_preprocess_function(tokenizer, max_input_length=512, max_target_length=128):
    """Create preprocessing function for tokenization"""

    def preprocess_function(examples):
        source_column = "cleaned_text"
        target_column = "summary"

        inputs = []
        targets = []

        for text, summary in zip(examples[source_column], examples[target_column]):
            if text and summary:
                clean_text = str(text).strip()
                clean_summary = str(summary).strip()

                if len(clean_text) > 10 and len(clean_summary) > 5:
                    inputs.append(clean_text)
                    targets.append(clean_summary)

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding=False
        )

        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function

def tokenize_datasets(raw_datasets, preprocess_function):
    """Apply tokenization to all datasets"""
    print("Tokenizing datasets...")

    tokenized_datasets = {}
    source_column = "cleaned_text"
    target_column = "summary"

    for split, dataset in raw_datasets.items():
        cols_to_remove = [col for col in dataset.column_names
                         if col not in [source_column, target_column]]

        tokenized_datasets[split] = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=cols_to_remove,
            batch_size=100
        )

    print("✓ Tokenization completed")
    return tokenized_datasets

# =============================================================================
# STEP 6: EVALUATION METRICS SETUP
# =============================================================================

def setup_evaluation_metrics(tokenizer):
    """Setup ROUGE evaluation metrics"""
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        try:
            result = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True,
                use_aggregator=True
            )

            result = {k: round(v, 4) for k, v in result.items()}
            avg_pred_length = np.mean([len(pred.split()) for pred in decoded_preds])
            result['avg_pred_length'] = round(avg_pred_length, 2)

            return result
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    return compute_metrics

# =============================================================================
# STEP 7: LORA CONFIGURATION AND MODEL ADAPTATION
# =============================================================================

def setup_lora_config():
    """Configure LoRA for parameter-efficient fine-tuning"""
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False
    )
    return peft_config

def apply_lora_to_model(model, peft_config):
    """Apply LoRA configuration to the model"""
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("✓ LoRA configuration applied")
    return model

def freeze_encoder_layers(model, num_layers_to_freeze=8):
    """Freeze initial encoder layers to reduce training time"""
    if hasattr(model.base_model.model, 'encoder'):
        encoder_layers = model.base_model.model.encoder.layers
        for i in range(min(num_layers_to_freeze, len(encoder_layers))):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
        print(f"✓ Frozen first {num_layers_to_freeze} encoder layers")

# =============================================================================
# STEP 8: TRAINING CONFIGURATION
# =============================================================================

def setup_training_arguments():
    """Configure training parameters"""
    training_args = Seq2SeqTrainingArguments(
        output_dir="./mbart-lora-telugu-optimized",
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=8,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        logging_dir="./logs",
        overwrite_output_dir=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        warmup_steps=100,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        push_to_hub=False,
        save_safetensors=True
    )
    return training_args

def setup_data_collator(tokenizer, model, max_input_length=512):
    """Configure data collator for dynamic padding"""
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=max_input_length,
        pad_to_multiple_of=8
    )
    return data_collator

# =============================================================================
# STEP 9: TRAINER INITIALIZATION AND TRAINING
# =============================================================================

def initialize_trainer(model, training_args, tokenized_datasets, tokenizer,
                      data_collator, compute_metrics):
    """Initialize the Seq2Seq trainer"""
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print("✓ Trainer initialized successfully")
    return trainer

def start_training(trainer):
    """Begin the training process"""
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training completed!")

# =============================================================================
# STEP 10: MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline"""
    print("=" * 60)
    print("mBART Telugu Summarization Model Training Pipeline")
    print("=" * 60)

    # Step 1: Environment Setup
    setup_environment()
    download_nltk_requirements()

    # Step 2: Data Loading
    train_df, val_df, test_df = load_datasets()
    raw_datasets = prepare_data(train_df, val_df, test_df)

    # Step 3: Model Initialization
    model, tokenizer = load_model_and_tokenizer()

    # Step 4: Data Preprocessing
    preprocess_function = create_preprocess_function(tokenizer)
    tokenized_datasets = tokenize_datasets(raw_datasets, preprocess_function)

    # Step 5: Evaluation Setup
    compute_metrics = setup_evaluation_metrics(tokenizer)

    # Step 6: LoRA Configuration
    peft_config = setup_lora_config()
    model = apply_lora_to_model(model, peft_config)
    freeze_encoder_layers(model)

    # Step 7: Training Configuration
    training_args = setup_training_arguments()
    data_collator = setup_data_collator(tokenizer, model)

    # Step 8: Trainer Setup and Training
    trainer = initialize_trainer(model, training_args, tokenized_datasets,
                               tokenizer, data_collator, compute_metrics)
    start_training(trainer)

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
