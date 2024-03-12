import os
import time
import random
import argparse
import tensorboard
import torch

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from models.progen.configuration_progen import ProGenConfig
from tokenizers.implementations import BaseTokenizer

# wget "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28IPR029059%29" -O ABH.fasta.gz
# gunzip ABH.fasta.gz

def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

set_seed(42)


from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
base_tokenizer = Tokenizer.from_file('tokenizer.json')
tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
tokenizer.pad_token = "<|pad|>"
tokenizer.bos_token = "<|bos|>"
tokenizer.eos_token = "<|eos|>"



config = ProGenConfig(
    vocab_size=len(tokenizer.get_vocab()),
    n_positions=1024,
    n_ctx=1024,
    n_embd=256,
    n_layer=3,
    n_head=16,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
    
model = ProGenForCausalLM(config)

# print(model)
# exit()

from datasets import load_dataset
from datasets import Dataset, load_dataset, DatasetDict

from Bio import SeqIO

with open('ABH.csv', 'w') as out_fh:
    out_fh.write('ID,Seq\n')
    for x in SeqIO.parse('ABH.fasta', 'fasta'):
        full_seq = str(x.seq)
        out_fh.write(f'{x.id},{full_seq}\n')

train_ds = load_dataset('csv', data_files = 'ABH.csv')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def preprocess_function(samples):
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    for protein_sequence in samples['Seq']:
        protein_sequence = protein_sequence.replace("*", "")
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

from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./progen_tiny_results",
    evaluation_strategy="steps",
    learning_rate=5e-4,
    weight_decay=0.1,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=32,
    warmup_steps=100,
    max_steps=200, # only a demo
    logging_steps=50,
    eval_steps=100,
    logging_strategy="steps",
    bf16=True,
    save_total_limit = 3,
    report_to = "tensorboard",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_testvalid["train"],
    eval_dataset=train_testvalid["test"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./progen_tiny")
tokenizer.save_pretrained("./progen_tiny")
