from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class ModelTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", output_dir="./model_output", num_samples=5000):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_environment()

    def setup_environment(self):
        """Configure training environment"""
        if not torch.cuda.is_available():
            raise RuntimeError("This script requires a GPU to run!")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    def load_dataset(self):
        """Load and preprocess dataset"""
        dataset = load_dataset("lara-martin/FIREBALL", trust_remote_code=True)
        dataset = dataset["train"].shuffle(seed=42).select(range(self.num_samples))
        splits = dataset.train_test_split(test_size=0.1, seed=42)
        logging.info(f"Training samples: {len(splits['train'])}")
        logging.info(f"Validation samples: {len(splits['test'])}")
        return splits

    def preprocess_function(self, examples, tokenizer):
        """Tokenize and preprocess examples"""
        def clean_utterances(utterances):
            if not utterances:
                return ""
            filtered = [u for u in utterances if not u.strip().startswith(("!i", "(")) and len(u.strip()) > 0]
            return " [SEP] ".join(filtered).strip().replace("  ", " ").lower()

        formatted_stories = [
            f"<|context|>{clean_utterances(before)}<|continue|>{clean_utterances(after)}<|end|>"
            for before, after in zip(examples["before_utterances"], examples["after_utterances"])
        ]
        return tokenizer(
            formatted_stories,
            truncation=True,
            max_length=512,  # Increased max length
            padding="max_length",
            return_tensors="pt"
        )

    def initialize_model(self):
        """Initialize and configure model with LoRA adapters"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=32,  # Increased rank for better representations
            lora_alpha=64,  # Increased LoRA alpha for better adaptation
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,  # Slightly higher dropout for regularization
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logging.info("LoRA adapters applied.")
        return model, tokenizer

    def get_training_args(self):
        """Configure training arguments with balanced optimizations"""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,  # Reduced for faster updates
            num_train_epochs=3,  # Slightly increased epochs
            learning_rate=3e-5,  # Adjusted learning rate
            warmup_ratio=0.1,
            fp16=True,
            gradient_checkpointing=True,
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            logging_steps=10,
            report_to="none",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            group_by_length=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def save_model(self, model, tokenizer):
        """Save the model and tokenizer"""
        os.makedirs(self.output_dir, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        logging.info(f"Model and tokenizer saved successfully to {self.output_dir}")

    def train(self):
        """Execute training pipeline"""
        dataset = self.load_dataset()
        model, tokenizer = self.initialize_model()

        tokenized_dataset = dataset.map(
            lambda x: self.preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=model,
            args=self.get_training_args(),
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )

        trainer.train()
        self.save_model(model, tokenizer)
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Training completed.")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
