from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch

model_path="./saved_model"

tokenizer=AutoTokenizer.from_pretrained(model_path)

model=AutoModelForSequenceClassification.from_pretrained(model_path)

text="প্রতি ঘণ্টায় চাই হোম কোয়ারেন্টাইনদের ভিডিও!"

evidence="কর্ণাটক সরকারের নয়া সিদ্ধান্ত।"

inputs=tokenizer(
    evidence,
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.no_grad():

    outputs=model(**inputs)

prediction=torch.argmax(outputs.logits,dim=1).item()

labels={
    0:"SUPPORTS",
    1:"REFUTES",
    2:"NOT ENOUGH INFO"
}

print(labels[prediction])