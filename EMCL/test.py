"""
Used for seq2seq and mice based on dataset flights with multi-model approach
"""
import os
import sys
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
from EMCL.utils import F1_score
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from EMCL.custom_model import CustomT5Model
from dataset import Seq2SeqDataset


class MultiModelIterativeGenerativeRepair():
    def __init__(self, dirty_path, clean_path, exclude_list=None, iteration_number=3, gpu='cuda:0'):
        self.dirty_df = pd.read_csv(dirty_path)
        self.clean_df = pd.read_csv(clean_path)
        self.clean_df.columns = self.dirty_df.columns
        self.device = gpu
        self.iteration_number = iteration_number
        # exclude some unwanted columns like tuple_id
        if exclude_list:
            assert max(exclude_list) < len(self.clean_df.columns)
            self.clean_df.drop(self.clean_df.columns[exclude_list], axis=1, inplace=True)
            self.dirty_df.drop(self.dirty_df.columns[exclude_list], axis=1, inplace=True)

        self.error_df = (self.clean_df != self.dirty_df)
        self.res_df = self.clean_df.copy()  # store result sync
        self.temp_df = self.clean_df.copy()  # copy res_df to process async
        
        # Initialize model dictionary for each column
        self.models = {}
        self.optimizers = {}
        self.tokenizers = {}
        
        # Create checkpoint directory
        self.checkpoint_dir = './models_ckp'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def initialize_model(self):
        """Initialize a new model for a specific column"""
        seq2seq_model = 'google-t5/t5-large'
        tokenizer = AutoTokenizer.from_pretrained(seq2seq_model)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model)
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        
        # Apply LoRA to base model
        base_model = get_peft_model(base_model, peft_config)
        # Wrap with CL, comment out to do ablation study.
        model = CustomT5Model(base_model)
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        return model, optimizer, tokenizer

    def preprocess(self, target_index=0, iteration_time=1):
        train_data, test_data = [], []
        n_rows, n_cols = self.clean_df.shape
        column_names = list(self.clean_df.columns)

        target_column_name = column_names[target_index]
        input_column_name = column_names[:target_index]
        if target_index != n_cols - 1:
            input_column_name += column_names[target_index + 1:]

        dtypes = {col: str(self.clean_df[col].dtype) for col in column_names}

        for row in range(n_rows):
            # Get input values
            inputs = []
            for col in range(n_cols):
                if col == target_index:
                    continue
                # if iteration_time == 1 and self.error_df.iloc[row, col]:
                #     inputs.append("<extra_id_0>")
                else:
                    inputs.append(str(self.temp_df.iloc[row, col]))

            # too complex, rewrite it
            # Construct info part
            info_parts = []
            for col, val in zip(input_column_name, inputs):
                info_parts.append(f"(Attribute: {col}, Dtype: {dtypes[col]}, Value: {val}, Status: {self.error_df.iloc[row, column_names.index(col)]})")
            
            # Construct predict part
            target_dtype = dtypes[target_column_name]
            target_original = str(self.dirty_df.iloc[row, target_index])
            predict_part = f"(Attribute: {target_column_name}, Dtype: {target_dtype}, Origin: {target_original})"
            
            # Combine into final prompt
            input_content = f"{{Info: {', '.join(info_parts)}, Predict: {predict_part}}}"

            target_content = str(self.clean_df.iloc[row, target_index])
            if self.error_df.iloc[row, target_index]:
                test_data.append((input_content, target_content))
            else:
                train_data.append((input_content, target_content))

        return train_data, test_data

    def train_step(self, train_loader, epochs=3):
        model = self.model
        optimizer = self.optimizer
        device = self.device

        model.train()
        for epoch in range(epochs):
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
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
                
                # issues here after adding new layer
                generated_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=max_length)
                preds = [tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g_id in
                         generated_ids]
                predictions.extend(preds)
                targets.extend(
                    [tokenizer.decode(tgt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt in
                     labels])

        return predictions, targets

    def save_model_state(self, column_name):
        """Save model state to disk"""
        ckp_path = os.path.join(self.checkpoint_dir, f'{column_name}.pt')
        torch.save(self.model.state_dict(), ckp_path)

        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def load_model_state(self, column_name):
        """Load model state from disk"""
        ckp_path = os.path.join(self.checkpoint_dir, f'{column_name}.pt')
        if os.path.exists(ckp_path):
            # Initialize new model
            model, optimizer, tokenizer = self.initialize_model()
            # Load state
            model.load_state_dict(torch.load(ckp_path))
            # Update dictionaries
            self.model = model
            self.optimizer = optimizer
            self.tokenizer = tokenizer

    def run(self, epochs=3, batch_size=16):
        self.epochs = epochs
        self.batch_size = batch_size

        for iteration in range(self.iteration_number):
            print(f'---------------start iteration {iteration}---------------')
            # processing each column
            for ind in range(len(self.clean_df.columns)):
                column_name = self.clean_df.columns[ind]
                
                # skip columns without errors
                if self.error_df.iloc[:, ind].sum() == 0:
                    print(f'skip index {ind}')
                    continue
                
                # Initialize model for this column if not exists
                if iteration == 0:
                    print(f'Initializing model for column {column_name}')
                    self.model, self.optimizer, self.tokenizer = self.initialize_model()
                else:
                    print(f'Loading model for column {column_name} from iteration {iteration-1}')
                    self.load_model_state(column_name)
                
                # Process data and train
                train_data, test_data = self.preprocess(target_index=ind, iteration_time=iteration + 1)
                train_dataset = Seq2SeqDataset(train_data, self.tokenizer)
                test_dataset = Seq2SeqDataset(test_data, self.tokenizer)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                self.train_step(train_loader, epochs=self.epochs)
                predictions, targets = self.test_step(test_loader)
                print(f'fine tune for column {column_name} done')

                # Update results
                n_rows, n_cols = self.clean_df.shape
                pointer = 0
                for row in range(n_rows):
                    if self.error_df.iloc[row, ind]:
                        self.res_df.iloc[row, ind] = predictions[pointer]
                        pointer += 1
                self.temp_df = self.res_df.copy()
                
                # Save model state and clear from GPU
                print(f'Saving model state for column {column_name}')
                self.save_model_state(column_name)
            
            # detect errors after each iteration
            self.error_df = (self.res_df != self.clean_df)

        print('Done.')

    def get_res(self):
        return self.res_df


if __name__ == "__main__":
    # to run the code, use the following command like:
    # nohup python multi.py flight cuda:0 >> ./logs/flight_multi.txt 2>&1 &
    args = sys.argv[1:]
    experiment = args[0]
    gpu = args[1]
    if experiment == 'flight':
        dataset_name, exclude_list = 'flight', [0]
    elif experiment == 'beers':
        dataset_name, exclude_list = 'beers', [0]
    elif experiment == 'hospital':
        dataset_name, exclude_list = 'hospital', [0, 16]
    elif experiment == 'rayyan':
        dataset_name, exclude_list = 'rayyan', [0, 10]
    else:
        assert False, "No such dataset"

    dirty_path = './dataset/' + dataset_name + '/dirty.csv'
    clean_path = './dataset/' + dataset_name + '/clean.csv'

    a = MultiModelIterativeGenerativeRepair(dirty_path, clean_path, exclude_list=exclude_list, iteration_number=5, gpu=gpu)
    a.run(batch_size=16, epochs=3)
    r = a.get_res()
    r.to_csv('./result/' + dataset_name + '_repaired_multi.csv', index=False)
    true = a.clean_df
    dirty = a.dirty_df
    f1_score = F1_score(true, r, dirty)
    print(dataset_name)
    print(r.head())
    print(f"F1 score: {f1_score}") 