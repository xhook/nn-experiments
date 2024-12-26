from functools import partial

from nnutils.determinism import deterministic, seed_all

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore

from torchsummary import summary # type: ignore
from torchvision.transforms._presets import ImageClassification # type: ignore

from nnutils.models.resnet import resnet18, resnet50
from nnutils.imagenet import ImageNetDatasetAsync

from model import ResNet, GenResNet
from training import Trainer
from utils import count_params


# torch.autograd.set_detect_anomaly(True)

def load_cifar10(config, rng, worker_init_fn):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/data/datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['train_config']['bs'], drop_last=True,
                                            shuffle=True, num_workers=8,
                                            generator=rng, worker_init_fn=worker_init_fn)
    testset = torchvision.datasets.CIFAR10(root='/data/datasets', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['train_config']['bs'],
                                            shuffle=False, num_workers=8,
                                            generator=rng, worker_init_fn=worker_init_fn)
    return trainloader, testloader

def load_imagenet(config, rng, worker_init_fn):
    input_size = config['train_config']['input_size']
    resize_size = input_size[0]
    train_ds = ImageNetDatasetAsync(
        'data/imagenet_train.csv',
        # limit=1024 * 16,
        transform=transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandomResizedCrop(resize_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]),
        # partial(ImageClassification, 
        cache_dir='/data/datasets/cache/imagenet/train',
    )
    train_dl = torch.utils.data.DataLoader(train_ds, 
                                           num_workers = 16, 
                                           shuffle = True, 
                                           batch_size=config['train_config']['bs'], 
                                           generator=rng, 
                                           worker_init_fn=worker_init_fn,
                                        #    persistent_workers=True,
                                           )
    test_ds = ImageNetDatasetAsync(
        'data/imagenet_val.csv',
        # limit=1024 * 16, 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        cache_dir='/data/datasets/cache/imagenet/val',
    )
    test_dl = torch.utils.data.DataLoader(test_ds, 
                                          num_workers = 16,
                                          shuffle = False,
                                          batch_size=config['train_config']['bs'],
                                          generator=rng,
                                          worker_init_fn=worker_init_fn,
                                        #   persistent_workers=True,
                                          )
    return train_dl, test_dl

def run_experiment(name, config, resume):
    rng, worker_init_fn = deterministic(True)
    
    if config['train_config']['dataset'] == 'cifar10':
        trainloader, testloader = load_cifar10(config, rng, worker_init_fn)
    elif config['train_config']['dataset'] == 'imagenet':
        trainloader, testloader = load_imagenet(config, rng, worker_init_fn)

    model = config['model'](**config['model_config'])
    summary(model, input_size=(3,) + config['train_config']['input_size'], device='cpu')
    print(f'Model parameter count: {count_params(model)}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train_config']['lr'])
    
    trainer = Trainer(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config['train_config']['epochs'],
        device='cuda',
        runs_dir='./runs',
        compile=config['train_config']['compile'],
    )
    if resume:
        trainer.resume(name)
    else:
        trainer.start(name)

def run_experiment_in_subprocess(name, config, resume):
    from torch.multiprocessing import Process, set_start_method
    set_start_method('spawn')
    p = Process(target=run_experiment, args=(name, config, resume), )
    p.start()
    p.join()

def print_layer_con_weights(net: GenResNet):
    for r in net.layer_cons:
        for c in r:
            print(f"{c.item():1.2f} ", end='')
        print("")

if __name__ == '__main__':
    configs = {
        # 'resnet18_skip_after_nonlin_cifar10': {
        #     'model': resnet18, 
        #     'model_config': {'skip_after_nonlin': True, 'num_classes': 10}, 
        #     'train_config': {'dataset': 'cifar10', 'input_size': (32, 32), 'epochs': 100, 'bs': 64, 'lr': 0.001, 'compile': True}
        # },
        # 'resnet50_skip_after_nonlin_cifar10': {
        #     'model': resnet50, 
        #     'model_config': {'skip_after_nonlin': True, 'num_classes': 10}, 
        #     'train_config': {'dataset': 'cifar10', 'input_size': (32, 32), 'epochs': 100, 'bs': 64, 'lr': 0.001, 'compile': True}
        # },
        'resnet50_skip_after_nonlin_imagenet': {
            'model': resnet50, 
            'model_config': {'skip_after_nonlin': True, 'num_classes': 1000}, 
            'train_config': {'dataset': 'imagenet', 'input_size': (224, 224), 'epochs': 120, 'bs': 128, 'lr': 0.0003, 'compile': True}
        },

        # 'resnet18_classic_cifar10': {
        #     'model': resnet18, 
        #     'model_config': {'skip_after_nonlin': False, 'num_classes': 10},
        #     'train_config': {'dataset': 'cifar10', 'input_size': (32, 32), 'epochs': 100, 'bs': 64, 'lr': 0.001, 'compile': True}
        # },
        # 'resnet18_skip_after_nonlin_imagenet': {
        #     'model': resnet18, 
        #     'model_config': {'skip_after_nonlin': True, 'num_classes': 1000}, 
        #     'train_config': {'dataset': 'imagenet', 'input_size': (224, 224), 'epochs': 100, 'bs': 64, 'lr': 0.001, 'compile': True}
        # },
        # 'resnet18_classic_imagenet': {
        #     'model': resnet18, 
        #     'model_config': {'skip_after_nonlin': False, 'num_classes': 1000},
        #     'train_config': {'dataset': 'imagenet', 'input_size': (224, 224), 'epochs': 1, 'bs': 64, 'lr': 0.001, 'compile': True}
        # },
    }

    for name, config in configs.items():
        # create a subprocess for the experiment to enable deterministic behaviour when using torch.compile
        run_experiment_in_subprocess(name, config, True)
