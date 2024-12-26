import numpy as np
import torch
import pandas as pd
from PIL import Image
import io
import hashlib
import httpx
import asyncio
import os
import pathlib

class ImageNetDatasetAsync(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file_path: str,
                 dataset_path: str,
                 transform = None,
                 limit: int | None = None,
                 cache_dir: str | None = None,
                 http_basic_auth_user: str | None = None,
                 http_basic_auth_password: str | None = None,                 
                 ):
        self.csv_file_path = csv_file_path
        self.dataset_path = dataset_path
        self.df = pd.read_csv(csv_file_path)
        if limit is not None:
            self.df = self.df.head(n=limit)
        self.transform = transform
        self.client: httpx.AsyncClient | None = None  # Will initialize the httpx.AsyncClient later
        self.loop: asyncio.AbstractEventLoop | None  = None    # Will initialize an asyncio loop later
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            assert self.cache_dir.is_dir()
        self.auth: httpx.BasicAuth | None = httpx.BasicAuth(http_basic_auth_user, http_basic_auth_password) if http_basic_auth_user and http_basic_auth_password else None

    def __len__(self):
        return len(self.df)

    async def download_file_async(self, url: str) -> bytes:
        cache_file_path = self.cache_dir / hashlib.md5(url.encode()).hexdigest() if self.cache_dir else None
        if cache_file_path is not None and cache_file_path.is_file():
            with open(cache_file_path.absolute(), 'rb') as file:
                content = file.read()
        else:
            if self.client is None:
                self.client = httpx.AsyncClient(http2=True, headers={'Authorization': str(self.auth)} if self.auth else None)  # Initialize client lazily
            response = await self.client.get(url, timeout=httpx.Timeout(10))
            response.raise_for_status()
            content = response.content
            if cache_file_path is not None:
                with open(cache_file_path.absolute(), 'wb') as file:
                    file.write(content)
        return content

    def download_file(self, url: str) -> bytes:
        """ Wrapper to run async download in sync context """
        if self.loop is None:
            self.loop = asyncio.new_event_loop()  # Create a new event loop for the process
            asyncio.set_event_loop(self.loop)
            asyncio.get_event_loop
        return asyncio.get_event_loop().run_until_complete(self.download_file_async(url))
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f'{self.dataset_path}/{row['path']}'
        file_bytes = self.download_file(img_path)
        img  = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        if self.transform:
            img  = self.transform(img)
        else:
            img  = np.array(img)
        return img, row['label']