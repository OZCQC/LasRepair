"""
Used for seq2seq and mice based on dataset flights
"""
import os
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import GPUtil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim import AdamW, SGD, Adam
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import gc


class Seq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0)
        }

class IterativeGenerativeRepair():
    def __init__(self, dirty_path, clean_path, exclude_list=None, iteration_number=3, device='cuda:0'):
        # convert them into same dtype
        self.dirty_df = pd.read_csv(dirty_path).astype(str)
        self.clean_df = pd.read_csv(clean_path).astype(str)
        self.clean_df.columns = self.dirty_df.columns
        self.device = device
        self.iteration_number = iteration_number
        # exclude some unwanted columns like tuple_id
        if exclude_list:
            assert max(exclude_list) < len(self.clean_df.columns)
            self.clean_df.drop(self.clean_df.columns[exclude_list], axis=1, inplace=True)
            self.dirty_df.drop(self.dirty_df.columns[exclude_list], axis=1, inplace=True)
        
        self.error_df = (self.clean_df != self.dirty_df)
        self.res_df = self.clean_df.copy()  # store result sync
        self.temp_df = self.clean_df.copy() # copy res_df to process async

    
    def preprocess(self, target_index=0, iteration_time=1):
        train_data, test_data = [], []
        n_rows, n_cols = self.clean_df.shape
        column_names = list(self.clean_df.columns)
        
        target_column_name = column_names[target_index]
        input_column_name = column_names[:target_index]
        if target_index != n_cols-1:
            input_column_name += column_names[target_index+1:]
            
        if iteration_time!=1:
            # no need for masking then
            for row in range(n_rows):
                inputs = list(self.temp_df.iloc[row, :target_index].astype(str))
                # no index error
                if target_index != n_cols-1:
                    inputs += list(self.temp_df.iloc[row, target_index+1:].astype(str))
                input_content = ""
                for pair in zip(input_column_name, inputs):
                    input_content += pair[0] + ": " + pair[1] + "<extra_id_1>"
                post_fix = target_column_name + ": "
                input_content += post_fix

                # target_content = str(self.clean_df.iloc[row, -1]) 你无敌了
                target_content = str(self.clean_df.iloc[row, target_index])
                if self.error_df.iloc[row, target_index]:
                    test_data.append((input_content, target_content))
                else:
                    train_data.append((input_content, target_content))

        else:
            for row in range(n_rows):
                inputs = []
                for col in range(n_cols):
                    if col == target_index:
                        pass
                    else:
                        # ! column assigned incorrectly
                        if self.error_df.iloc[row, col]:  # if wrong then using mask
                            inputs.append("<extra_id_0>")
                        else:
                            inputs.append(str(self.dirty_df.iloc[row, col]))
                input_content = ""  # modify this initial value to implement prefix prompt
                # prefix = "some prompt"
                # input_content = prefix + input_content
                for pair in zip(input_column_name, inputs):
                    input_content += pair[0] + ": " + pair[1] + "<extra_id_1>"
                post_fix = target_column_name + ": "
                input_content += post_fix

                # split into test and train
                target_content = str(self.clean_df.iloc[row, target_index])
                if self.error_df.iloc[row, target_index]:
                    test_data.append((input_content, target_content))
                else:
                    train_data.append((input_content, target_content))
        
        return train_data, test_data


    # need modify
    def train_step(self, train_loader, epochs=3):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        
        model.train()
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}")
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # incase divide by 0
            print(f"Epoch {epoch+1} Train Loss: {total_loss/(len(train_loader)+1):.4f}")

    # need modify
    def test_step(self, test_loader, max_length=128):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        
        model.eval()
        total_loss = 0
        predictions = []
        targets = []
        print('start testing')
        with torch.no_grad():
            for batch in test_loader:
                # loss part
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
                # prediction part
                generated_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=max_length)
                # generated_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_new_tokens=128)
                preds = [tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g_id in generated_ids]
                predictions.extend(preds)
                # compare to targets
                targets.extend([tokenizer.decode(tgt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt in labels])
                
        print(f"Eval Loss: {total_loss/(len(test_loader) + 1):.4f}")
        return predictions, targets

    def run(self, epochs=3, batch_size=16):
        device = self.device
        seq2seq_model = 'google-t5/t5-large'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model).to(device)
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(seq2seq_model)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model).to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.batch_size = batch_size

        for iteration in range(self.iteration_number):
            # processing each column
            for ind in range(len(self.clean_df.columns)):
                # skip all true but useful cols
                if self.error_df.iloc[:, ind].sum() == 0:
                    continue
                # multi_version
                # if iteration == 0:
                #     # using pretrained t5-large for the first time
                #     self.model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model).to(device)
                #     self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
                # else:
                #     # load the previous model for specific col
                #     self.model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model).to(device)
                #     self.model.load_state_dict(torch.load(f"./temp_model/col_{ind}.bin"))
                #     self.model.to(device)
                #     self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
                
                train_data, test_data = self.preprocess(target_index=ind, iteration_time=iteration+1)
                train_dataset = Seq2SeqDataset(train_data, self.tokenizer)
                test_dataset = Seq2SeqDataset(test_data, self.tokenizer)
                # in case to fill the predictions to the df, do not shuffle
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)  # why need shuffle?
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
                print(f'---------------start fine tune col index {ind}---------------')
                self.train_step(train_loader, epochs=self.epochs)
                predictions, targets = self.test_step(test_loader)  # max_length 128
                print(f'---------------fine tune for col index {ind} done---------------')
                for i in range(5):
                    print(f"prediction: {predictions[i]}\t targets: {targets[i]}")
    
                # modify the data
                n_rows, n_cols = self.clean_df.shape
                pointer = 0
                for row in range(n_rows):
                    if self.error_df.iloc[row, ind]:
                        self.res_df.iloc[row, ind] = predictions[pointer]
                        pointer += 1
                self.temp_df = self.res_df.copy()
                # temp_model_path = f"./temp_model/col_{ind}.bin"
                # torch.save(
                #     self.model.state_dict(),
                #     temp_model_path
                # )
                # del self.model
                # torch.cuda.empty_cache()
                # gc.collect()
                
        print('Done.')
        

    def get_res(self):
        return self.res_df
        


if __name__=="__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
    print(torch.cuda.current_device())
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(dt.now())
    print("---------START---------")
    # # flights iteration=3, epochs=3, using t5-large, bug fixed
    # dirty_path = '../dataset/flight/dirty.csv'
    # clean_path = '../dataset/flight/clean.csv'
    # a = IterativeGenerativeRepair(dirty_path, clean_path, exclude_list=[0], device=device, iteration_number=5)
    dirty_path = '../dataset/hospital/dirty.csv'
    clean_path = '../dataset/hospital/clean.csv'
    a = IterativeGenerativeRepair(dirty_path, clean_path, exclude_list=[0, 16], device=device, iteration_number=4)
    # OOM, so batch size 4
    a.run(batch_size=8, epochs=3)
    r = a.get_res()
    true = a.clean_df
    new_error = (r != true)
    print(r.head())
    for i in range(r.shape[1]):
        print(a.error_df.iloc[:, i].sum(), new_error.iloc[:, i].sum())
    print(dt.now())
    
        
