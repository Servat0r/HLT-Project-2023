import numpy as np
import logging
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer

logging.root.setLevel(logging.INFO)
logging.basicConfig(filename='bert_example.log')

logging.info('Loading dataset ...')
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
logging.info('Dataset loaded, loading tokenizer ...')
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
logging.info('Tokenizer loaded')
training_args = TrainingArguments("test-trainer", evaluation_strategy='epoch')


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
logging.info('Loading model ...')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
logging.info('Model loaded')


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logging.info('Starting training ...')
trainer.train()
logging.info('Training ended!')
