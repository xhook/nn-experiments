import torch

from torchvision import transforms
from nnutils.imagenet import ImageNetDatasetAsync

def main():
    train_ds = ImageNetDatasetAsync(
        csv_file_path='data/imagenet_val.csv',
        dataset_path='http://nas.localdomain:5080/oophuwohkahghia7',
        limit=1024 * 16,
        # cache_dir='/data/datasets/cache/imagenet/train',
    )
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers = 4)

    for i, (x, y) in enumerate(train_dl):
        print(i, x.shape, y.shape)

if __name__ == '__main__':
    main()
