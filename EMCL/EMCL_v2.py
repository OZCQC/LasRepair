"""
Used for seq2seq and mice based on dataset flights with single model approach
v1 is original.py
v2: try to add a small budget for all_wrong_corrector
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
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import F1_score, all_wrong_corrector, error_drop_rate
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
        self.dirty_df = pd.read_csv(dirty_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '')
        self.clean_df = pd.read_csv(clean_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '')
        self.clean_df.columns = self.dirty_df.columns
        self.clean_df = self.clean_df.astype(str)
        self.dirty_df = self.dirty_df.astype(str)
        if self.clean_df.shape != self.dirty_df.shape:
            raise ValueError('clean_df and dirty_df have different shape, set to lower shape')

        self.device = args.gpu
        self.max_iteration = args.max_iteration
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.name = args.experiment
        self.use_weight = args.use_weight
        self.sample_prop = args.sample_prop
        self.temperature = args.temperature
        self.threshold = args.threshold
        self.budget = args.budget
        self.learning_rate = args.learning_rate

        self.error_df = (self.clean_df.ne(self.dirty_df))
        self.res_df = self.dirty_df.copy()  # store result sync
        self.temp_df = self.dirty_df.copy()  # copy res_df to process async
        self.weight_df = self.error_df.astype(float)  # full uncertainty matrix
        self.weight_used = np.ones(len(self.clean_df))  # row-wise weights for training
        self.flag = True
        
        # Initialize single model for all columns
        self.model = None
        self.optimizer = None
        self.tokenizer = None

    def initialize_model(self):
        """Initialize a single model for all columns"""
        seq2seq_model = 'google-t5/t5-large'
        tokenizer = AutoTokenizer.from_pretrained(seq2seq_model)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model)
        
        model = base_model
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        
        return model, optimizer, tokenizer

    def compute_weight_used(self):
        """Compute row-wise mean of weight_df and apply softmax to get weight_used"""
        # Compute row means across all columns
        row_means = self.weight_df.mean(axis=1).values  # Convert to numpy array
        # temperature used here. default is 1.0, lower temperature means more focus on the error values
        row_means = row_means / self.temperature
        
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
    
        # generate prompt and target
        # prompt: col_name: col_value <extra_id_1> .... <extra_id_2> target: target_name
        for row in range(n_rows):
            # Get input values
            inputs = []
            # Use all columns except target
            for col in range(n_cols):
                if col == target_index:
                    continue
                # mask the error values
                if iteration_time == 1 and self.error_df.iloc[row, col]:
                    inputs.append(str(column_names[col]) + ": " + "<extra_id_0>")
                else:
                    inputs.append(str(column_names[col]) + ": " + str(self.temp_df.iloc[row, col]))

            target_content = str(self.clean_df.iloc[row, target_index])

            # Construct prompt, simple, just for slm
            if self.error_df.iloc[row, target_index]:
                # if column_names[target_index] == 'article_jissue':
                #     example = f"EXAMPLE: (dirty: 4.0, correct: 4), (dirty: 2.0, correct: 2), (dirty: 9.0, correct: 9)"
                # elif column_names[target_index] == 'article_jcreated_at':
                #     example = f"EXAMPLE: (dirty: 1/1/71, correct: 1/1/71), (dirty: 1/1/72, correct: 1/1/72), (dirty: 1/1/73, correct: 1/1/73)"
                # elif column_names[target_index] == 'article_jvolumn':
                #     example = f"EXAMPLE: (dirty: 64, correct: 64), (dirty: 65, correct: 65), (dirty: 66, correct: 66)"
                input_content = "<extra_id_1>".join(inputs)
            # contain dirty information now, comment out to go back to original version.
                input_content = input_content + f"<extra_id_2> {str(column_names[target_index])}: " + str(self.temp_df.iloc[row, target_index])
                # prefix the task
                input_content = "correct column " + str(column_names[target_index]) + ": " + input_content
                target_content = ''
                test_data.append((input_content, target_content))
            else:
                input_content = "<extra_id_1>".join(inputs)
                input_content = input_content + "<extra_id_2>" + str(column_names[target_index]) + ": "
                train_data.append((input_content, target_content))
                # Get weight for this training sample from weight_used (default to 1.0 for first iteration)
                weight = float(self.weight_used[row]) if iteration_time > 1 and self.use_weight else 1.0
                train_weights.append(weight)
            
            if self.flag and column_names[target_index] == 'article_jissue':
                print(f"input_content: {input_content}, target_content: {target_content}")
                self.flag = False

        return train_data, test_data, train_weights

    # all in one. only will be used for little budget training. i.e. some columns with high error rate. col is string
    def budget_finetune(self, col, sample_indices, epochs=10):
        # dataset
        train_data = []
        for row in sample_indices:
            content = str(self.clean_df.iloc[row, col])
            content = f"{str(self.dirty_df.columns[col])}_dirty: {str(self.dirty_df.iloc[row, col])} <extra_id_2> target:"
            target = str(self.clean_df.iloc[row, col])
            train_data.append((content, target))

        train_dataset = Seq2SeqDataset(train_data, self.tokenizer, weights=None)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_step(train_loader)
        print(f"budget finetune for column {col}\n")


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
                if "weight" in batch and self.use_weight:
                    weights = batch["weight"].to(device).float()
                    
                    # Get logits and compute weighted loss manually
                    outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits  # [batch, seq_len, vocab_size]
                    
                    # Compute cross-entropy loss per token
                    vocab_size = logits.size(-1)
                    loss_per_token = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction='none'
                    ).view(labels.size())  # [batch, seq_len]
                    
                    # Mask out padding tokens and compute per-sample loss
                    valid_mask = (labels != -100).float()
                    per_sample_loss = (loss_per_token * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                    
                    # Apply sample weights and compute final loss
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
        n_samples = n  # number of samples to calculate the confident matrix

        model.eval()
        total_loss = 0
        predictions, targets, first_n_tokens, logits = [], [], [], []
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

                if self.use_weight:
                    first_pos_logits = outputs.logits[:, 0, :]  # [batch, vocab] - logits for first position
                    
                    # here is ids, not tokens!
                    top_n_logits, top_n_ids = torch.topk(first_pos_logits, k=n_samples, dim=-1)

                    first_n_tokens.extend(top_n_ids.cpu().tolist())  # List of [n] lists
                    logits.extend(top_n_logits.cpu().tolist())  # List of [n] lists

        return predictions, targets, first_n_tokens, logits

    def run(self):
        """
        Main training loop
        """
        now_time = time.time()
        start_time = time.time()
        self.last_f1 = 0

        # Initialize single model once
        print('Initializing model')
        self.model, self.optimizer, self.tokenizer = self.initialize_model()

        # use budget to fine tune columns with high error rate
        np.random.seed(114)
        error_count = self.error_df.sum(axis=0)
        for col in range(len(self.clean_df.columns)):
            if error_count[col] > self.threshold * self.clean_df.shape[0]:
                """
                不能保证一定选出来的是错误的，记得修正。（目前对要处理的数据集来说没问题）
                """
                sample_indices = np.random.choice(self.error_df.shape[0], self.budget, replace=False)
                self.budget_finetune(sample_indices=sample_indices, col=col, epochs=10 * self.epochs)
            else:
                continue

        # column version
        for iteration in range(self.max_iteration):
            print(f'---------------start iteration {iteration + 1}---------------')
            self.flag = True
            # processing each column in the order of error rate()
            for ind in range(len(self.clean_df.columns)):
                column_name = self.clean_df.columns[ind]
                
                # skip columns without errors
                if self.error_df.iloc[:, ind].sum() == 0:
                    print(f'skip index {ind}\n')
                    continue
                
                # Process data and train
                train_data, test_data, train_weights = self.preprocess(target_index=ind, iteration_time=iteration + 1)
                
                # Sample here for efficiency
                if self.sample_prop < 1:
                    sample_indices = np.random.choice(len(train_data), int(len(train_data) * self.sample_prop), replace=False)
                    train_data = [train_data[i] for i in sample_indices]
                    train_weights = [train_weights[i] for i in sample_indices]

                train_dataset = Seq2SeqDataset(train_data, self.tokenizer, weights=train_weights if self.use_weight else None)
                test_dataset = Seq2SeqDataset(test_data, self.tokenizer)
                # must not shuffle, or the order will be wrong
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                self.train_step(train_loader, epochs=self.epochs)
                print(f'fine tune for column {column_name}: {(time.time() - now_time)/60:.2f} minutes')
                now_time = time.time()

                predictions, targets, first_n_tokens, logits = self.test_step(test_loader)
                print(f'inference for column {column_name}: {(time.time() - now_time)/60:.2f} minutes\n')
                now_time = time.time()

                # Update results
                if self.use_weight:
                    weights = uncertainty_matrix(first_n_tokens, logits)
                n_rows, n_cols = self.clean_df.shape
                pointer = 0
                for row in range(n_rows):
                    if self.error_df.iloc[row, ind]:
                        self.res_df.iloc[row, ind] = predictions[pointer]
                        if self.use_weight:
                            self.weight_df.iloc[row, ind] = weights[pointer]
                        pointer += 1
                # the repaired data is immediately used for the next columns.
                self.temp_df = self.res_df.copy()

            # detect errors after each iteration
            self.error_df = (self.res_df.ne(self.clean_df))
            
            # compute weight_used from weight_df after each iteration
            if self.use_weight:
                print(f'Computing weight_used after iteration {iteration + 1}\n')
                self.compute_weight_used()

            now_time = time.time()

            f1_score = self.get_f1()
            print(f'F1 score: {f1_score}')
            if f1_score - self.last_f1 < 0.005:
                break
            else:
                self.last_f1 = f1_score

        print('Done.\n')
        print(f'Time taken: {((time.time() - start_time)/60):.2f} minutes')

    def get_res(self):
        return self.res_df

    def get_f1(self):
        f1_score = F1_score(self.clean_df.astype(str), self.res_df.astype(str), self.dirty_df.astype(str))
        return f1_score

    def get_edr(self):
        edr = error_drop_rate(self.clean_df, self.dirty_df, self.res_df)
        return edr


if __name__ == "__main__":
    # to run the code, use the following command like:
    # nohup python original.py --experiment flight --gpu cuda:0 >> ./logs/flight_original.txt 2>&1 &
    args = argparse.ArgumentParser()
    args.add_argument("--experiment", type=str, default='flight')
    args.add_argument("--gpu", type=str, default='cuda:0')
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--max_iteration", type=int, default=10)
    args.add_argument("--use_weight", type=bool, default=True)
    args.add_argument("--sample_prop", type=float, default=1.0)
    args.add_argument("--temperature", type=float, default=1.0)
    args.add_argument("--threshold", type=float, default=0.8)  # if error rate is greater than threshold, then use budget.
    args.add_argument("--budget", type=int, default=20)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args = args.parse_args()

    # modify the path to run on your own dataset
    dirty_path = '/data1/qianc/EMCL/datasets/' + args.experiment + '/dirty.csv'
    clean_path = '/data1/qianc/EMCL/datasets/' + args.experiment + '/clean.csv'
    print(f"arguments:\n weight: {args.use_weight}, sample_prop: {args.sample_prop}, budget: {args.budget}")
    print(f"batch_size: {args.batch_size}, epochs: {args.epochs}, max_iteration: {args.max_iteration}, temperature: {args.temperature}")

    a = MultiModelIterativeGenerativeRepair(dirty_path, clean_path, args)
    a.run()
    r = a.get_res()
    r.to_csv('./result/' + args.experiment + '_repaired_v2.csv', index=False)
    # r.to_csv('./result/' + args.experiment + '_test.csv', index=False)
    true = a.clean_df
    dirty = a.dirty_df
    f1_score = F1_score(true, r, dirty)
    print(args.experiment)
    print(r.head())
    print(f"F1 score: {f1_score}")
    print(f"running done at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")