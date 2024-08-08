import os
import time
import random
import argparse
import tensorboard
import torch

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from tokenizers.implementations import BaseTokenizer


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["WANDB_DISABLED"] = "true"

set_seed(42)
ckpt = "./checkpoints/progen2-base"
model = ProGenForCausalLM.from_pretrained(
    ckpt, 
    low_cpu_mem_usage=True,
    device_map={"":0},
)
# print(model)
# exit()
from transformers import DataCollatorForLanguageModeling

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
base_tokenizer = Tokenizer.from_file('tokenizer.json')
tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
tokenizer.pad_token = "<|pad|>"
tokenizer.bos_token = "<|bos|>"
tokenizer.eos_token = "<|eos|>"

from datasets import load_dataset
from datasets import Dataset, load_dataset, DatasetDict

from Bio import SeqIO

with open('tdt_fastas.csv', 'w') as out_fh:
    out_fh.write('ID,Seq\n')
    for x in SeqIO.parse('tdt.fa', 'fasta'):
        full_seq = str(x.seq)
        out_fh.write(f'{x.id},{full_seq}\n')

train_ds = load_dataset('csv', data_files = 'tdt_fastas.csv')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def preprocess_function(samples):
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    for i, protein_sequence in enumerate(samples['Seq']):
        # protein_sequence = protein_sequence.replace("*", "")
        tokenized_input = tokenizer(f"<|bos|>1{protein_sequence}2<|eos|>", padding="longest", truncation=True, max_length=1024)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
    return processed_samples

tokenized_ds = train_ds.map(
    preprocess_function,
    batched=True,
    num_proc=12,
)

train_testvalid = tokenized_ds['train'].train_test_split(test_size=0.2)

from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["qkv_proj", "out_proj"],
    # target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

from peft import get_peft_model

model = get_peft_model(model, config)

from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./progen_base_tdt_outputs",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=0.1,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=4,
    warmup_steps=1000,
    max_steps=3000, # only a demo
    logging_steps=100,
    eval_steps=100,
    logging_strategy="steps",
    save_total_limit = 3,
    report_to = "tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_testvalid["train"],
    eval_dataset=train_testvalid["test"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./progen_base_tdt_lora")