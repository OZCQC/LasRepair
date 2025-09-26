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
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import F1_score, group_by_modularity, all_wrong_corrector
from .custom_model import CustomT5Model
from .dataset import Seq2SeqDataset
from .confident_learning import uncertainty_matrix
import time
import argparse


class MultiModelIterativeGenerativeRepair():
    def __init__(self, dirty_path, clean_path, args:argparse.Namespace):
        """
        args: argparse.Namespace, see main function for more details
        """
        self.args = args
        self.dirty_df = pd.read_csv(dirty_path)
        self.clean_df = pd.read_csv(clean_path)
        self.clean_df.columns = self.dirty_df.columns
        self.clean_df = self.clean_df.astype(object)
        self.dirty_df = self.dirty_df.astype(object)

        self.device = args.gpu
        self.iteration_number = args.iteration_number
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.sample_prop = args.sample_prop
        self.do_group = args.do_group
        self.name = args.experiment
        self.group_path = f'./{self.name}_group.npy'  # store the group results

        self.error_df = (self.clean_df != self.dirty_df)
        self.clean_df, self.dirty_df, self.error_df = all_wrong_corrector(self.clean_df, self.dirty_df, self.error_df, prop=0.2)
        self.res_df = self.clean_df.copy()  # store result sync
        self.temp_df = self.clean_df.copy()  # copy res_df to process async
        self.weight_df = self.error_df.astype(float)  # full uncertainty matrix
        self.weight_used = np.ones(len(self.clean_df))  # row-wise weights for training
        self.group = []  # store group results, like [0, 0, 0, 1, 2, 4, 3]
        
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
        
        # lora version
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_2_SEQ_LM,
        #     inference_mode=False,
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        #     target_modules=["q", "v"]
        # )
        
        # # Apply LoRA to base model
        # base_model = get_peft_model(base_model, peft_config)
        # # Wrap with CL, comment out to do ablation study.
        # model = CustomT5Model(base_model)

        model = base_model
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        return model, optimizer, tokenizer

    def compute_weight_used(self):
        """Compute row-wise mean of weight_df and apply softmax to get weight_used"""
        # Compute row means across all columns
        row_means = self.weight_df.mean(axis=1).values  # Convert to numpy array
        
        # Apply softmax to get normalized weights
        exp_means = np.exp(row_means - np.max(row_means))  # Subtract max for numerical stability
        self.weight_used = exp_means / np.sum(exp_means)
        
    def preprocess(self, target_index=0, iteration_time=1):
        """
        preprocess the data
        divide the data into train and test
        prepare the prompt and the target for the model
        calculate the weight for the training data
        """
        train_data, test_data, train_weights = [], [], []
        n_rows, n_cols = self.clean_df.shape
        column_names = list(self.clean_df.columns)
        group = self.group[target_index]
        group_index = []
        for ind in range(n_cols):
            if self.group[ind] == group:
                group_index.append(ind)
    
        # generate prompt and target
        # prompt: col_name: col_value <extra_id_1> .... <extra_id_2> target: target_name
        if len(group_index) == 1:
            # use full data if there is only one column in the group
            group_index = [i for i in range(n_cols)]
        for row in range(n_rows):
            # Get input values
            inputs = []
            # no group version
            for col in group_index:
                if col == target_index:
                    continue
                # mask the error values
                if iteration_time == 1 and self.error_df.iloc[row, col]:
                    inputs.append(str(column_names[col]) + ": " + "<extra_id_0>")
                else:
                    inputs.append(str(column_names[col]) + ": " + str(self.temp_df.iloc[row, col]))

            # Construct prompt, simple, just for slm
            input_content = "<extra_id_1>".join(inputs)
            input_content = input_content + "<extra_id_2> target: " + str(column_names[target_index])
            target_content = str(self.clean_df.iloc[row, target_index])
            
            if self.error_df.iloc[row, target_index]:
                test_data.append((input_content, target_content))
            else:
                train_data.append((input_content, target_content))
                # Get weight for this training sample from weight_used (default to 1.0 for first iteration)
                weight = float(self.weight_used[row]) if iteration_time > 1 else 1.0
                train_weights.append(weight)

        return train_data, test_data, train_weights

    def train_step(self, train_loader, epochs=3):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        epochs = self.epochs

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Check if weights are provided
                if "weight" in batch:
                    weights = batch["weight"].to(device).float()
                    
                    # Get logits and compute weighted loss manually
                    outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits  # [batch, seq_len, vocab_size]
                    
                    # Compute cross-entropy loss per token
                    vocab_size = logits.size(-1)
                    loss_per_token = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,  # ?
                        reduction='none'
                    ).view(labels.size())  # [batch, seq_len]
                    
                    # Mask out padding tokens and compute per-sample loss
                    valid_mask = (labels != -100).float()
                    per_sample_loss = (loss_per_token * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                    
                    # Apply sample weights and compute final loss
                    """
                    这个地方的clamp好像有点问题
                    """
                    loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
                else:
                    # Standard unweighted loss
                    outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                    loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def test_step(self, test_loader, max_length=128, n=5):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        n_samples = n # number of samples to calculate the confident matrix

        model.eval()
        total_loss = 0
        predictions, targets, first_n_tokens, logits = [], [], [], []
        print('inference start')
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
                
                generated_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=max_length)
                preds = [tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g_id in
                         generated_ids]
                predictions.extend(preds)
                targets.extend(
                    [tokenizer.decode(tgt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt in
                     labels])

                first_pos_logits = outputs.logits[:, 0, :]  # [batch, vocab] - logits for first position
                
                # here is ids, not tokens!
                top_n_logits, top_n_ids = torch.topk(first_pos_logits, k=n_samples, dim=-1)

                first_n_tokens.extend(top_n_ids.cpu().tolist())  # List of [n] lists
                logits.extend(top_n_logits.cpu().tolist())  # List of [n] lists

        return predictions, targets, first_n_tokens, logits

    def save_model_state(self, column_name):
        """Save model state to disk"""
        ckp_path = os.path.join(self.checkpoint_dir, f'{column_name}.pt')
        torch.save(self.model.state_dict(), ckp_path)

        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def load_model_state(self, name):
        """Load model state from disk"""
        ckp_path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        if os.path.exists(ckp_path):
            # Initialize new model
            model, optimizer, tokenizer = self.initialize_model()
            # Load state
            model.load_state_dict(torch.load(ckp_path, weights_only=False))
            # Update dictionaries
            self.model = model
            self.optimizer = optimizer
            self.tokenizer = tokenizer

    def run(self, epochs=3, batch_size=16, sample_prop=0.5):
        """
        sample_prop: 1 for no sample, should be float
        """
        self.epochs = epochs
        self.batch_size = batch_size
        now_time = time.time()

        """
        maybe better to do it on the clean part of the dirty data.
        """
        if self.do_group:
            if os.path.exists(self.group_path):
                self.group = np.load(self.group_path)
            else:
                self.group = group_by_modularity(self.clean_df, sample_rows=40, dimensions=256, resolution=1)
                np.save(self.group_path, self.group)
        else:
            self.group = [0] * len(self.clean_df.columns)
        group_model = []

        # column version
        for iteration in range(self.iteration_number):
            print(f'---------------start iteration {iteration + 1}---------------')
            # processing each column
            for ind in range(len(self.clean_df.columns)):
                column_name = self.clean_df.columns[ind]
                group = self.group[ind]
                model_name = self.name + '_' + str(group)
                
                # skip columns without errors
                if self.error_df.iloc[:, ind].sum() == 0:
                    print(f'skip index {ind}')
                    continue
                
                # Column version, initialize model for this column if not exists
                # if iteration == 0:
                #     print(f'Initializing model for column {column_name}')
                #     self.model, self.optimizer, self.tokenizer = self.initialize_model()
                # else:
                #     print(f'Loading model for column {column_name} from iteration {iteration}')
                #     self.load_model_state(column_name)
                
                # group version, load model for this group if not exists
                if iteration == 0 and model_name not in group_model:
                    print(f'\n Initializing model for group {group}')
                    self.model, self.optimizer, self.tokenizer = self.initialize_model()
                    group_model.append(model_name)
                else:
                    print(f'Loading model for group {group}')
                    self.load_model_state(model_name)
                
                # Process data and train
                train_data, test_data, train_weights = self.preprocess(target_index=ind, iteration_time=iteration + 1)
                # sample here for efficiency
                if sample_prop < 1:
                    sample_indices = np.random.choice(len(train_data), int(len(train_data) * sample_prop), replace=False)
                    train_data = [train_data[i] for i in sample_indices]
                    train_weights = [train_weights[i] for i in sample_indices]

                train_dataset = Seq2SeqDataset(train_data, self.tokenizer, weights=train_weights)
                test_dataset = Seq2SeqDataset(test_data, self.tokenizer)
                # must not shuffle, or the order will be wrong
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                self.train_step(train_loader, epochs=self.epochs)
                print(f'fine tune for column {column_name} done, time taken: {(time.time() - now_time)/60} minutes')
                now_time = time.time()

                predictions, targets, first_n_tokens, logits = self.test_step(test_loader)
                print(f'inference for column {column_name} done, time taken: {(time.time() - now_time)/60} minutes')
                now_time = time.time()

                # Update results
                weights = uncertainty_matrix(first_n_tokens, logits)
                n_rows, n_cols = self.clean_df.shape
                pointer = 0
                for row in range(n_rows):
                    if self.error_df.iloc[row, ind]:
                        self.res_df.iloc[row, ind] = predictions[pointer]
                        # warning, weight_df is bool, while float is allocated.
                        self.weight_df.iloc[row, ind] = weights[pointer]
                        pointer += 1
                self.temp_df = self.res_df.copy()
                
                # Save model state and clear from GPU
                print(f'Saving model state for group {group}')
                self.save_model_state(model_name)
                
            
            # detect errors after each iteration
            self.error_df = (self.res_df != self.clean_df)
            
            # compute weight_used from weight_df after each iteration
            print(f'Computing weight_used after iteration {iteration}\n')
            self.compute_weight_used()

            now_time = time.time()

        """
        This line is kinda dangerous, delete model checkpoints.
        """
        os.system(f'rm -rf {self.checkpoint_dir}/{self.name}_*')
        print('Done.\n')

    def get_res(self):
        return self.res_df


if __name__ == "__main__":
    # to run the code, use the following command like:
    # nohup python multi.py flight cuda:0 >> ./logs/flight_multi.txt 2>&1 &
    args = argparse.ArgumentParser()
    args.add_argument("--experiment", type=str, default='flight')
    args.add_argument("--gpu", type=str, default='cuda:0')
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--iteration_number", type=int, default=3)
    args.add_argument("--sample_prop", type=float, default=0.5)
    args.add_argument("--do_group", type=bool, default=False)
    args = args.parse_args()

    # modify the path to run on your own dataset
    dirty_path = '/data1/qianc/EMCL/datasets/' + args.experiment + '/dirty.csv'
    clean_path = '/data1/qianc/EMCL/datasets/' + args.experiment + '/clean.csv'

    a = MultiModelIterativeGenerativeRepair(dirty_path, clean_path, args)
    a.run(batch_size=args.batch_size, epochs=args.epochs)
    r = a.get_res()
    r.to_csv('./result/' + args.experiment + '_repaired_multi.csv', index=False)
    true = a.clean_df
    dirty = a.dirty_df
    f1_score = F1_score(true, r, dirty)
    print(args.experiment)
    print(r.head())
    print(f"F1 score: {f1_score}") 