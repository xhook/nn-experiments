import torch

from tqdm import tqdm
from torchvision import transforms
from nnutils.imagenet import ImageNetDatasetAsync

def main():
    train_ds = ImageNetDatasetAsync(
        csv_file_path='data/imagenet_train.csv',
        dataset_path='http://nas.localdomain:5080/oophuwohkahghia7',
        # limit=1024 * 16,
        cache_dir='/Users/dzhukov/Datasets/cache/imagenet/train',
    )
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers = 16)

    for x, y in tqdm(train_dl):
        pass

if __name__ == '__main__':
    main()
