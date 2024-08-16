#!/usr/bin/env python
# coding: utf-8
"""
   Copyright 2016-2024 Wannaphong Phatthiyaphaibun

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# In[1]:


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"


# In[2]:


import transformers
import sentencepiece as spm
transformers.__version__


# In[3]:


from datasets import load_dataset, load_metric, Dataset, DatasetDict

metric = load_metric("cer")


# In[4]:


import pandas as pd

#load data from csv
train_df = pd.read_csv('train.tsv',sep="\t")
val_df = pd.read_csv('val.tsv',sep="\t")
test_df = pd.read_csv('test.tsv',sep="\t")
train_df.tail()


# In[5]:


all_df= pd.read_csv("wiktionary-23-7-2022-clean.tsv",sep="\t",names=["grapheme","phoneme"])


# In[6]:


#convert to dictionary
j_train = {'translation':[]}
for i in train_df.itertuples():
    j_train['translation'] += [{'grapheme':i[1], 'phoneme':i[2]}]
#convert to dictionary
j_val = {'translation':[]}
for i in val_df.itertuples():
    j_val['translation'] += [{'grapheme':i[1], 'phoneme':i[2]}]
#convert to dictionary
j_test = {'translation':[]}
for i in test_df.itertuples():
    j_test['translation'] += [{'grapheme':i[1], 'phoneme':i[2]}]


# In[7]:


#load into dataset
dataset_pre = {"train":j_train,"validation":j_val,"test":j_test}
raw_datasets = DatasetDict()
# using your `Dict` object
for k,v in dataset_pre.items():
    raw_datasets[k] = Dataset.from_dict(v)


# In[8]:


raw_datasets


# In[9]:


import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks]['translation'])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


# In[10]:


show_random_elements(raw_datasets["train"])


# In[11]:


metric


# In[12]:


import sentencepiece as spm


# In[13]:


vocab_size = 5000 #choose what you want
user_defined_symbols = '<pad>'
vocab_special_size = vocab_size + len(user_defined_symbols)

#train tokenizer
def train_spm_tokenizer(train_fname, 
                        model_prefix,
                        vocab_special_size,
                        model_dir,
                        character_coverage=0.9995,
                        max_sentencepiece_length=16,
                        add_dummy_prefix='false',
                        model_type='unigram',
                        user_defined_symbols='<pad>'):
    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.train((f'--input={train_fname} '
                                   f'--model_prefix={model_prefix} '
                                   f'--vocab_size={vocab_special_size} '
                                   f'--character_coverage={character_coverage} '
                                   f'--max_sentencepiece_length={max_sentencepiece_length} '
                                   f'--add_dummy_prefix={add_dummy_prefix} '
                                   f'--model_type={model_type} '
                                   f'--user_defined_symbols={user_defined_symbols}'))
    get_ipython().system('mkdir $model_dir; mv $model_prefix* $model_dir')


# In[14]:


fake_preds = ["hello there general kenobi"]
fake_labels = [["hello there general kenobi"]]
metric.compute(predictions=fake_preds, references=fake_labels)


# In[15]:


train_df.to_csv('tokenizer_train.txt',header=None, index=None, sep='\n')


# In[16]:


#train both spm
#train_spm_tokenizer(train_fname='tokenizer_train.txt',
#                    model_prefix='both',
#                    vocab_special_size=vocab_special_size,
#                    model_dir='marian-mt'
#                   ) 


# In[17]:


#get_ipython().system('cp marian-mt/both.model marian-mt/source.spm')
#get_ipython().system('cp marian-mt/both.model marian-mt/target.spm')


# In[18]:


import json

#read vocab
with open('marian-mt/both.vocab','r') as f:
    vocab_lines = f.readlines()
vocab_lines[:10]
vocab_dict = {j.split('\t')[0]:i for i,j in enumerate(vocab_lines)}


# In[19]:


#save vocab
with open('marian-mt/vocab.json','w') as f:
    json.dump(vocab_dict,f)


# In[20]:


from transformers import MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained('marian-mt')


# In[21]:


tokenizer("สวัสดีครับ นี่ประโยคเดียว")


# In[22]:


max([len(tokenizer(i)['input_ids']) for i in all_df.grapheme.to_list()]) #':i[1], 'phoneme


# In[23]:


max([len(tokenizer(i)['input_ids']) for i in all_df.phoneme.to_list()])


# In[24]:


tokenizer.decode(tokenizer("สวัสดีครับ นี่ประโยคเดียว").input_ids)


# In[25]:


max_input_length = 20
max_target_length = 90
source_lang = "grapheme"
target_lang = "phoneme"

def preprocess_function(examples, tokenizer, prefix=''):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[26]:


preprocess_function(raw_datasets['train'][:2], tokenizer)


# In[27]:


from functools import partial

tokenized_datasets = raw_datasets.map(partial(preprocess_function, tokenizer=tokenizer), 
                                      batched=True,)


# In[28]:


fake_preds = ["hello there general kenobi"]
fake_labels = [["hello there general kenobi"]]
metric.compute(predictions=fake_preds, references=fake_labels)


# In[29]:


from transformers import (
    MarianMTModel, 
    MarianConfig, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
config = MarianConfig.from_pretrained('marian-mt')
model = MarianMTModel(config)


# In[30]:


batch_size = 4
args = Seq2SeqTrainingArguments(
    output_dir="models",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    #save_total_limit=1,
    num_train_epochs=200,
    predict_with_generate=True,
    fp16=True,
    overwrite_output_dir=True,
)


# In[31]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[32]:


import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"cer": result}#["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# In[33]:


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[34]:


trainer.train()


# In[ ]:





# In[35]:


tokenizer


# In[36]:


print(tokenizer.supported_language_codes)


# In[37]:


translated = model.generate(**tokenizer(["ต้นตาล"], return_tensors="pt", padding=True).to("cuda"))


# In[38]:


[tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# In[ ]:




