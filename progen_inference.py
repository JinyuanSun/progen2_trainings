import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.progen.modeling_progen import ProGenForCausalLM
model = ProGenForCausalLM.from_pretrained("./progen_tiny")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
tokenizer = AutoTokenizer.from_pretrained("./progen_tiny")

prompt = "<|bos|>1MDGVLWRVRTAALMAALLALAAWALVWASPSVEAQSNPYQRGPNPTRSALTADGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSLAWLGRRLASHGFVVLVIN"
max_length = 300
num_return_sequences = 1
top_p = 1
temp = 0.8

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs.to(device),
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        temperature=temp,
        do_sample=True,
    )
print(prompt)
print(tokenizer.decode(output[0], skip_special_tokens=False))