import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class VolleyballBase(Dataset):
    def __int__(self,rootdir,split):
        #Main directories
        self.rootdir = rootdir
        self.split = split
        self.videos_dir = os.path.join(rootdir,"videos")

        #Splits
        self.train_ids = [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
        self.val_ids = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
        self.test_ids = [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

        #Pick a split
        if self.split == "train":
            self.video_ids = self.train_ids
        elif self.split == "val":
            self.video_ids = self.val_ids
        else:
            self.video_ids = self.test_ids

        #Define and enumerate classes
        self.classes = ['l_pass', 'r_pass', 'l_spike', 'r_spike', 'l_set', 'r_set', 'l_winpoint', 'r_winpoint']
        self.classes_to_idx = {cls: i for i,cls in enumerate(self.classes)}
        self.samples = self._load_annotations()

    def _load_annotations(self):
        pass