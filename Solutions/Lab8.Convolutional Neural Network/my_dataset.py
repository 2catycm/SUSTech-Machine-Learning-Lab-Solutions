from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets import VisionDataset
from torchvision import datasets,transforms
import os
import numpy as np 
import pandas as pd 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
import torch
import gzip,os, pickle
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class MyCIFAR10(VisionDataset):
    """这次lab尝试手写数据集加载，方便以后加载未知数据集"""
    # 静态变量
    train_list = [f"data_batch_{i}" for i in range(1, 6)]
    test_list = ['test_batch']
    base_path = 'cifar-10-batches-py'
    def __init__(self, root, transform=None, target_transform=None, train=True) -> None:
        # 通过继承简化代码
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.data:Any = []
        self.targets = []
        self._load_data()
        self._load_meta()
    def _load_data(self) -> None:
        chosen_list = self.train_list if self.train else self.test_list
        for file in chosen_list:
            d = unpickle(os.path.join(os.path.join(self.root, self.base_path),file))
            self.data.append(d[b'data']) # 这里用apoend而不是extend，因为extend会把list中的元素一个个加入，而不是整个list
            self.targets.extend(d[b'labels']) # 追加另一个序列的多个值
        self.data = np.vstack(self.data) # 垂直方向拼接，变成50000*3072的矩阵
        self.data = self.data.reshape(-1, 3, 32, 32) # 这个格式其实挺好的，已经是CHW
        self.data = self.data.transpose((0, 2, 3, 1)) # 闲着没事干，转置成HWC格式，方便可视化
    def _load_meta(self) -> None:
        self.classes = unpickle(os.path.join(self.root, self.base_path, 'batches.meta'))[b'label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index] # 获得第index个样本,只有一个图片哦
        img = Image.fromarray(img)
        # pytorch真是傻透了，这种逻辑为什么不复用
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self) -> int:
        return len(self.data)
        
def get_cifar10():
    """获取cifar10数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # pytorch 这API真实是透了
    return (MyCIFAR10('datasets/',transform=transform,train=train) for train in [True,False])