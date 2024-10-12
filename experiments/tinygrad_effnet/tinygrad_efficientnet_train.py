import numpy as np
import torch
from tqdm import tqdm
from time import time
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple

import tinygrad as tg
from tg_efficientnet import EfficientNet

from imagenet import ImageNetDatasetAsync

efficient_net_models = {
    0: (224, 224),
    1: (240, 240),
    2: (260, 260),
    3: (300, 300),
    4: (380, 380),
    5: (456, 456),
    6: (528, 528),
    7: (600, 600),
}

# val_ds = ImageNetDatasetAsync('imagenet_val.csv')

def accuracy(preds: tg.Tensor, targets: tg.Tensor):
    pred_matches = preds.softmax(axis=1).argmax(axis=1) == targets
    return pred_matches.sum() / preds.shape[0]

def test_accuracy():
    inputs = torch.tensor([[2.0, 6.0], [8.0, 2.0], [4.0, 3.0]])
    targets = torch.tensor([1, 1, 0])
    acc = accuracy(inputs, targets)
    print(acc)
    assert acc == 2/3

def main():
    loss_scaler = 128.0 if tg.dtypes.default_float == tg.dtypes.float16 else 1.0
    print("loss scaler", loss_scaler)
    
    model_version = 0
    # model = efficient_net_models[model_version]['model_fn'](weights=None).cuda()
    model = EfficientNet(model_version)
    # model.train()
    optimizer = tg.nn.optim.AdamW(params=tg.nn.state.get_parameters(model), lr=0.001)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # scaler = torch.amp.GradScaler()
    # scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=500)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, 10000, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [500])
    
    @tg.TinyJit
    def train_step(img: tg.Tensor, label: tg.Tensor) -> Tuple[tg.Tensor, tg.Tensor]:
        optimizer.zero_grad()
        pred = model.forward(img)
        loss = pred.cast(tg.dtypes.float32).sparse_categorical_crossentropy(label, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        acc = accuracy(pred, label)
        return loss, acc

    train_ds = ImageNetDatasetAsync('imagenet_train.csv', limit=None, transform=transforms.Compose([
        transforms.RandomResizedCrop(efficient_net_models[model_version]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]))
    dl = torch.utils.data.DataLoader(train_ds, num_workers=16, shuffle=True, batch_size=8, drop_last=True)
    downloaded_size = 0
    start = time()
    with tg.Tensor.train():
        for epoch in range(3):
            for img, file_size, label in (pbar := tqdm(dl, smoothing=0.01)):
                img = tg.Tensor(img.numpy())
                label = tg.Tensor(label.numpy())
                loss, acc = train_step(img, label)
                # scaler.scale(loss).backward()
                # scheduler.step()
                # scaler.step(optimizer)
                # scaler.update()

                downloaded_size += torch.sum(file_size).item()
                end = time()
                desc = ', '.join([
                    f'loss: {loss.mean().numpy():2.5f}',
                    f'accuracy: {acc.numpy():2.5f}',
                    f'throughput: {downloaded_size / 1e6 / (end - start):2.3f} Mb/s',
                ])
                pbar.set_description(desc)
        
if __name__ == '__main__':
    main()
