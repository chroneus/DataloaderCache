import numpy as np
import random
import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import glob
import psutil


def log(string: str):
    print(string)


class CachedDataset(Dataset):
    def __init__(self,
                 uncached_dataset: Dataset,
                 shared_dict: object = None,
                 preload: bool = False,
                 max_elements: int = -1
                 ):
        super()
        self.log("cached dataset instantiating")
        self.log(f"Free virtual memory available:{psutil.virtual_memory().free>>20} Mb")
        self.log(f"size of dataset {len(uncached_dataset)} items")
        self.uncached_dataset = uncached_dataset
        if shared_dict is None and not torch.distributed.is_initialized():
            self.shared_dict = {}
        else:
            self.shared_dict = shared_dict
        self.size = 0
        if max_elements == -1:
            self.max_elements = sys.maxsize
        else:
            self.max_elements = max_elements

        if preload:
            self.size = min(len(uncached_dataset), self.max_elements)
            for i in range(self.size):
                self.shared_dict[i] = uncached_dataset.__getitem__(i)
            self.log(f"Free virtual memory after preload :{psutil.virtual_memory().free>>20} Mb")

    def __len__(self):
        return len(self.uncached_dataset)


    def log(self,message:str):
        print(message)


    def __getitem__(self, idx):
        if idx in self.shared_dict:
            return self.shared_dict[idx]
        else:
            result = self.uncached_dataset.__getitem__(idx)
            if self.size < self.max_elements:
                self.shared_dict[idx] = result
            return result


class CacheImage():
    def __init__(self,
                 shared_dict: object = None,
                 preload_glob_path: str = None,
                 max_elements: int = -1,
                 preload_log_every: int =1000
                 ):

        if shared_dict is None and not torch.distributed.is_initialized():
                self.shared_dict = {}
        else:
            self.shared_dict = shared_dict
        if max_elements == -1:
            self.max_elements = sys.maxsize
        else:
            self.max_elements = max_elements


        if preload_glob_path is not None:
            paths = glob.glob(preload_glob_path)
            self.size = min(len(paths), self.max_elements)
            for i,img_path in enumerate(paths[:self.size]):                   
                with open(img_path, 'rb') as fh:
                    buf = BytesIO(fh.read())
                self.shared_dict[img_path] = buf
                if i%preload_log_every==0:
                    self.log(f"Free virtual memory after preload {i} images:{psutil.virtual_memory().free//1024//1024} Mb")

        self.size = len(self.shared_dict)



    def open(self,path):
        if self.shared_dict is None:
            return Image.open(path)

        if path in self.shared_dict:
            return Image.open(self.shared_dict[path])
        else:

            with open(path, 'rb') as fh:
                buf = BytesIO(fh.read())

            if self.size<self.max_elements:
                self.shared_dict[path] = buf
            size_dict = len(self.shared_dict)
        
        return Image.open(buf)

    
