import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer,BertForMaskedLM,DistilBertForMaskedLM
from datasets import load_dataset
from transformers import (
    BertTokenizer, BertForMaskedLM, DistilBertForMaskedLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling



)
# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load Hugging Face dataset (WikiText)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# # Data collator for MLM (15% masking probability)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Initialize Teacher and Student models
teacher_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
student_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")



teacher_model = teacher_model.to(device)
student_model = student_model.to(device)


# # Freeze Teacher
teacher_model.eval()


# # Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha_ce=0.5, alpha_distill=0.5):
    # MLM Cross Entropy Loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    ce_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

    # KL Divergence Loss with Temperature Scaling
    student_soft_logits = student_logits / temperature
    teacher_soft_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_soft_logits, dim=-1)
    teacher_probs = F.softmax(teacher_soft_logits, dim=-1)

    kl_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction='batchmean'
    ) * (temperature ** 2)

    # Combined Loss
    total_loss = alpha_ce * ce_loss + alpha_distill * kl_loss
    return total_loss

# Custom Trainer to integrate distillation
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"].to(device)

        # Teacher forward (can use full inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                token_type_ids=inputs.get("token_type_ids", None)
            )
            teacher_logits = teacher_outputs.logits

        # Student forward (DistilBERT doesn't support token_type_ids)
        student_outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )
        student_logits = student_outputs.logits

        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=2.0, alpha_ce=0.5, alpha_distill=0.5
        )
        return (loss, student_outputs) if return_outputs else loss


# # Training arguments
training_args = TrainingArguments(
    output_dir="./distilbert-distilled",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=False,
    remove_unused_columns=False,
)

# Initialize trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save distilled model
trainer.save_model("./distilbert-distilled-final")
