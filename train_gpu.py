from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from typing import List, Dict
import os
import bitsandbytes as bnb
import warnings
from torch.amp import autocast

# Suppress specific warnings
warnings.filterwarnings('ignore', 
    message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization')
warnings.filterwarnings('ignore', 
    message='`torch.cpu.amp.autocast.*is deprecated')

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

# Check GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("This script requires a GPU to run!")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Set Triton cache to a local directory
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"  # Local temp directory
# Create directory if it doesn't exist
os.makedirs("/tmp/triton_cache", exist_ok=True)

# Load larger dataset since we have GPU
dataset = load_dataset("lara-martin/FIREBALL")
dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # Reduced to 10k examples
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Training on {len(dataset['train'])} examples")
print(f"Evaluating on {len(dataset['test'])} examples")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    trust_remote_code=True,
    use_fast=True  # Use fast tokenizer for GPU
)
tokenizer.pad_token = tokenizer.eos_token

def format_story(before: List[str], after: List[str]) -> str:
    """Format the story context and continuation."""
    context = " ".join(before[-3:])
    continuation = " ".join(after[:2])
    return f"Context: {context}\nContinue the story: {continuation}"

def preprocess_function(examples):
    """Prepare the dataset for training."""
    formatted_stories = [
        format_story(before, after)
        for before, after in zip(examples["before_utterances"], examples["after_utterances"])
    ]
    
    return tokenizer(
        formatted_stories,
        truncation=True,
        max_length=128,  # Reduced from 256
        padding="max_length",
        return_tensors="pt"
    )

# Preprocess the dataset
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4  # Increased for faster preprocessing
)

# Modify BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,  # Changed to False
)

# Modify model loading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float32,  # Changed to float32
    trust_remote_code=True,
)

# Print configuration info for verification
print("\nModel quantization info:")
print(f"8-bit Quantization: {getattr(model, 'is_loaded_in_8bit', False)}")
print(f"Model dtype: {model.dtype}")
print(f"Model device: {model.device}")
print(f"Quantization config: {model.config.quantization_config}")

# Modify LoRA config for 8-bit training
lora_config = LoraConfig(
    r=8,  # Reduced from 16
    lora_alpha=16,  # Reduced from 32
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
    init_lora_weights=True,
)

# Prepare model with consistent dtypes
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Print trainable parameters
trainable_params = 0
all_param = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params
print(
    f"trainable params: {trainable_params} || "
    f"all params: {all_param} || "
    f"trainable%: {100 * trainable_params / all_param}"
)

# Modify training arguments
training_args = TrainingArguments(
    output_dir="./gm_model_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-4,
    logging_steps=5,
    save_steps=250,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=250,
    warmup_steps=50,
    optim="adamw_torch",  # Changed from adamw_8bit
    report_to="none",
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_length=True,
    remove_unused_columns=True,
    max_grad_norm=1.0,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    dataloader_pin_memory=True,
    fp16=False,  # Disabled mixed precision
    bf16=False,
)

# Add dtype verification before training
print("\nVerifying model dtypes:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.dtype}")
        # Force dtype if needed
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)

# Modify the callback class
class CustomProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("\nStarting training on GPU...")
        print(f"Training on device: {next(model.parameters()).device}")
        
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0:  # Removed autocast
            print(f"\nStep {state.global_step}/{state.max_steps}")
            print(f"Current loss: {state.log_history[-1]['loss'] if state.log_history else 'N/A'}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    callbacks=[CustomProgressCallback],
)

try:
    print("\nStarting training...")
    training_results = trainer.train()
    
    print("\nTraining completed successfully!")
    print("\nTraining Statistics:")
    print(f"Total steps: {training_results.global_step}")
    print(f"Training time: {training_results.training_time:.2f} seconds")
    print(f"Final loss: {training_results.training_loss:.4f}")
    
    print("\nSaving model...")
    model.save_pretrained("./gm_model_output")
    print("Model saved successfully!")
    
    if training_results.metrics:
        print("\nFinal Metrics:")
        for key, value in training_results.metrics.items():
            print(f"{key}: {value:.4f}")

except Exception as e:
    print(f"\nTraining error: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nSaving checkpoint...")
    model.save_pretrained("./gm_model_output")

finally:
    print("\nCleaning up GPU memory...")
    torch.cuda.empty_cache()
    print("Training completed.")