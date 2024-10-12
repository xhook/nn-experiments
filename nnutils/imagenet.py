import numpy as np
import torch
import pandas as pd
from PIL import Image
import io
import httpx
import asyncio

class ImageNetDatasetAsync(torch.utils.data.Dataset):
    def __init__(self, csv_file_path, transform=None, limit=None):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        if limit is not None:
            self.df = self.df.head(n=limit)
        self.transform = transform
        self.client = None  # Will initialize the httpx.AsyncClient later
        self.loop = None    # Will initialize an asyncio loop later

    def __len__(self):
        return len(self.df)

    async def download_file_async(self, url: str) -> bytes:
        """ Asynchronously download a file using httpx """
        if self.client is None:
            self.client = httpx.AsyncClient(http2=True)  # Initialize client lazily
        response = await self.client.get(url, timeout=httpx.Timeout(10))
        response.raise_for_status()
        return response.content

    def download_file(self, url: str) -> bytes:
        """ Wrapper to run async download in sync context """
        if self.loop is None:
            self.loop = asyncio.new_event_loop()  # Create a new event loop for the process
            asyncio.set_event_loop(self.loop)
            asyncio.get_event_loop
        return asyncio.get_event_loop().run_until_complete(self.download_file_async(url))
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        file_bytes = self.download_file(img_path)
        img  = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        # img  = np.array(img)
        if self.transform:
            img  = self.transform(img)
        return img, len(file_bytes), row['label']