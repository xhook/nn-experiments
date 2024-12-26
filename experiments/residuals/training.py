import os
from typing import Tuple, Dict, List, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from nnutils.metrics import MetricsLogger, CSVMetricsLogger


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        runs_dir: str = './runs',
        compile: bool = False,
    ):
        self.initial_model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.runs_dir = runs_dir
        self.compile = compile
        
        self.epoch: int | None = None
        self.model: nn.Model | None = None
        self.scaler = torch.amp.GradScaler()
        
    def accuracy(self, logits, labels):
        preds = torch.softmax(logits, dim=-1)
        preds = torch.argmax(preds, dim=1)
        acc = (preds == labels).sum().float() / len(labels)
        return acc

    @torch.no_grad()
    def test(self):
        total_loss = 0.0
        n = 0
        total_acc = 0.0
        for i, (inputs, labels) in enumerate(pbar := tqdm(self.testloader, smoothing=0.2)):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            total_acc += self.accuracy(outputs, labels).item()
            total_loss += self.criterion(outputs, labels).item()
            n += 1
            total_acc_avg = total_acc / (i + 1)
            total_loss_avg = total_loss / (i + 1)
            pbar.set_description(f'Validation | loss: {total_loss_avg:.5f}, acc: {total_acc_avg * 100:.2f}%')
        return total_loss_avg, total_acc_avg

    def save_checkpoint(self, run_name: str):
        path = os.path.join(self.runs_dir, run_name, 'checkpoint.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.initial_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }, path)
        
    def load_checkpoint(self, run_name: str):
        path = os.path.join('runs', run_name, f'checkpoint.pth')
        print(f"Loading checkpoint from {path}")
        state_dict = torch.load(path)
        model_state_dict = state_dict['model_state_dict']
        self.initial_model.load_state_dict(model_state_dict)
        self.initial_model.to(self.device)
        if self.compile:
            self.model = torch.compile(self.initial_model)
        else:
            self.model = self.initial_model
        if 'optimizer_state_dict' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if 'loss_scaler_state_dict' in state_dict:
            self.scaler.load_state_dict(state_dict['loss_scaler_state_dict'])
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
        
    def train(self, run_name: str):
        assert self.epoch is not None
        assert self.model is not None
        
        run_path = os.path.join(self.runs_dir, run_name)
        log_file_path = os.path.join(run_path, 'metrics.csv')
        logger = CSVMetricsLogger(log_file_path, ['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        
        while self.epoch <= self.epochs:
            self.model.train()
            running_train_loss = 0.0
            running_train_acc = 0.0
            print(f"[Epoch {self.epoch}/{self.epochs}]")
            for i, (inputs, labels) in enumerate(pbar := tqdm(self.trainloader, smoothing=0.05), 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_train_loss += loss.item()
                running_train_acc += self.accuracy(outputs, labels).item()
                running_train_loss_avg = running_train_loss / (i + 1)
                running_train_acc_avg = running_train_acc / (i + 1)
                pbar.set_description(f'Training   | loss: {running_train_loss_avg:.5f}, acc: {running_train_acc_avg * 100:.2f}%')
            self.model.eval()
            test_loss, test_acc = self.test()
            logger.log({'epoch': self.epoch, 'train_loss': running_train_loss_avg, 'test_loss': test_loss, 'train_acc': running_train_acc_avg, 'test_acc': test_acc})
            self.epoch += 1
            self.save_checkpoint(run_name)
        else:
            print("Training finished")

    def start(self, run_name: str):
        if self.epoch is not None:
            raise Exception('Cannot start new training with existing model state. Use `resume` or `continue` instead')
        self.epoch = 1
        run_path = os.path.join(self.runs_dir, run_name)
        if os.path.exists(run_path):
            raise Exception(f'Run already exists at {run_path}. Maybe you want to resume instead? If not, change the run name')
        os.makedirs(run_path)
        self.initial_model.to(self.device)
        if self.compile:
            print(f'torch.compile enabled')
            self.model = torch.compile(self.initial_model)
        else:
            self.model = self.initial_model
        print(f'Start training for run: {run_name}')
        self.train(run_name)
        
    def resume(self, run_name: str):
        run_path = os.path.join(self.runs_dir, run_name)
        if not os.path.exists(run_path):
            raise Exception(f'Run does not exist at {run_path}. Maybe you want to start a new one instead?')
        print('Loading checkpoint')
        self.load_checkpoint(run_name)
        print(f'Resuming training for run: {run_name}')
        self.train(run_name)
