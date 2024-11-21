import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

print("Using CPU for training")

class Args:
    model_name_or_path = "meta-llama/Llama-3.2-3B"
    cache_dir = "./cache/"
    model_revision = "main"
    use_fast_tokenizer = True
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1
    output_dir = "./gm_model_output"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    learning_rate = 2e-4
    weight_decay = 0.01
    save_steps = 50
    logging_steps = 5
    max_seq_length = 256
    warmup_ratio = 0.05
    evaluation_strategy = "steps"
    eval_steps = 50
    save_total_limit = 2
    gradient_accumulation_steps = 8
    num_train_examples = 1000

args = Args()

# Load a small subset of the dataset first
print("Loading dataset...")
dataset = load_dataset("lara-martin/FIREBALL", trust_remote_code=True)
train_dataset = dataset["train"].select(range(args.num_train_examples))
test_dataset = dataset["test"].select(range(args.num_train_examples // 10)) if "test" in dataset else \
    train_dataset.train_test_split(test_size=0.1)["test"]

def prepare_gm_training_data(example):
    # Filter out empty entries and ensure we have valid data
    if not example['after_utterances'] or not isinstance(example['after_utterances'], list):
        return None
        
    # Clean and format the utterances
    after_utterances = ' '.join(filter(None, example['after_utterances']))
    if not after_utterances.strip():
        return None
        
    # Prepare context
    context = f"Combat State: {example.get('combat_state_before', '')}\n"
    context += f"Current Actor: {example.get('current_actor', '')}\n"
    
    # Prepare input and target text
    input_text = (f"Context: {context}\n"
                 f"Player Action: {example.get('before_utterances', '')}\n"
                 f"Game Master Response:")
    
    return {
        "input_text": input_text,
        "target_text": after_utterances
    }

# Update dataset processing
print("Processing training dataset...")
train_dataset = train_dataset.map(
    prepare_gm_training_data,
    remove_columns=train_dataset.column_names,
    desc="Preparing training data"
).filter(lambda x: x is not None)

test_dataset = test_dataset.map(
    prepare_gm_training_data,
    remove_columns=test_dataset.column_names,
    desc="Preparing test data"
).filter(lambda x: x is not None)

# Initialize tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_fast=args.use_fast_tokenizer,
    revision=args.model_revision,
    use_auth_token=True  # You'll need to be logged in to Hugging Face
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add GM-specific special tokens
special_tokens_dict = {
    'additional_special_tokens': [
        '[CONTEXT]', '[SCENE]', '[PLAYER]', '[GM]'
    ]
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Load model with memory optimizations
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Free up memory
torch.cuda.empty_cache()
import gc
gc.collect()

# Resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# Configure LoRA
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
print("Applying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize_function(examples):
    # Ensure we have the required fields
    if not isinstance(examples, dict):
        print("Received data structure:", type(examples))
        raise ValueError("Expected dict input to tokenize_function")
    
    if 'input_text' not in examples or 'target_text' not in examples:
        print("Available keys:", examples.keys())
        raise ValueError("Missing required fields in tokenize_function")
    
    # Filter out None values
    valid_indices = [i for i, (inp, tgt) in enumerate(zip(examples['input_text'], examples['target_text']))
                    if inp is not None and tgt is not None]
    
    if not valid_indices:
        raise ValueError("No valid examples found after filtering")
    
    texts = [
        f"{examples['input_text'][i]}[GM]{examples['target_text'][i]}"
        for i in valid_indices
    ]
    
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_seq_length,
        return_tensors="pt"
    )

# Process datasets with batching enabled
print("Tokenizing dataset...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training data"
)
tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=test_dataset.column_names,
    desc="Tokenizing test data"
)

# Training arguments optimized for CPU
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    evaluation_strategy=args.evaluation_strategy,
    eval_steps=args.eval_steps,
    save_total_limit=args.save_total_limit,
    gradient_checkpointing=True,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_ratio=args.warmup_ratio,
    report_to="none",
    load_best_model_at_end=True,
    logging_dir=f"{args.output_dir}/logs",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the model
print("Starting training...")
trainer.train()

# Save the final model
print(f"Saving fine-tuned GM model to {args.output_dir}...")
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("GM model training complete.")