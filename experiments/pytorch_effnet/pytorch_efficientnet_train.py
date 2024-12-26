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

from nnutils.imagenet import ImageNetDatasetAsync

efficient_net_models = {
    "v1b0": {'input_size':(224, 224), 'model_fn': efficientnet_b0},
    "v1b1": {'input_size':(240, 240), 'model_fn': efficientnet_b1},
    "v1b2": {'input_size':(260, 260), 'model_fn': efficientnet_b2},
    "v1b3": {'input_size':(300, 300), 'model_fn': efficientnet_b3},
    "v1b4": {'input_size':(380, 380), 'model_fn': efficientnet_b4},
    "v1b5": {'input_size':(456, 456), 'model_fn': efficientnet_b5},
    "v1b6": {'input_size':(528, 528), 'model_fn': efficientnet_b6},
    "v1b7": {'input_size':(600, 600), 'model_fn': efficientnet_b7},
    "v2s":  {'input_size': (384, 384), 'model_fn': efficientnet_v2_s},
    "v2m":  {'input_size': (480, 480), 'model_fn': efficientnet_v2_m},
    "v2l":  {'input_size': (480, 480), 'model_fn': efficientnet_v2_l},
}

# val_ds = ImageNetDatasetAsync('imagenet_val.csv')

def accuracy(preds, targets):
    pred_matches = torch.argmax(torch.softmax(preds, dim=1), dim=1) == targets
    return pred_matches.sum() / preds.shape[0]

def test_accuracy():
    inputs = torch.tensor([[2.0, 6.0], [8.0, 2.0], [4.0, 3.0]])
    targets = torch.tensor([1, 1, 0])
    acc = accuracy(inputs, targets)
    print(acc)
    assert acc == 2/3

def main():
    model_version = 'v1b0'
    model = efficient_net_models[model_version]['model_fn'](weights=None).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=500)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, 10000, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [500])

    train_ds = ImageNetDatasetAsync('imagenet_train.csv', limit=None, transform=transforms.Compose([
        transforms.RandomResizedCrop(efficient_net_models[model_version]['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]))
    dl = torch.utils.data.DataLoader(train_ds, num_workers = 16, shuffle = True, batch_size=32)
    downloaded_size = 0
    start = time()
    for epoch in range(10):
        for img, file_size, label in (pbar := tqdm(dl, smoothing=0.01)):
            img = img.cuda()
            label = label.cuda()
            model.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(img)
                loss = loss_fn(pred, label)
            scaler.scale(loss).backward()
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()

            downloaded_size += torch.sum(file_size).item()
            end = time()
            acc = accuracy(pred, label)            
            pbar.set_description(f'loss: {torch.mean(loss).item():2.5f}, accuracy: {acc:2.5f}, lr: {scheduler.get_last_lr()[0]:2.7f}, data throughput {downloaded_size / 1e6 / (end - start):2.3f} Mb/s')
        
if __name__ == '__main__':
    main()
