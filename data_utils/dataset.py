import os
from typing import List, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset

class VolleyballSceneBase(Dataset):
    def __init__(self,rootdir,split):
        # Main directories
        self.rootdir = rootdir
        self.split = split
        self.videos_dir = os.path.join(rootdir,"videos")

        # Splits
        self.train_ids = [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
        self.val_ids = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
        self.test_ids = [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

        # Pick a split
        if self.split == "train":
            self.video_ids = self.train_ids
        elif self.split == "val":
            self.video_ids = self.val_ids
        else:
            self.video_ids = self.test_ids

        # Define and enumerate classes
        self.classes = ['l_pass', 'r_pass', 'l_spike', 'r_spike', 'l_set', 'r_set', 'l_winpoint', 'r_winpoint']
        self.classes_to_idx = {cls: i for i,cls in enumerate(self.classes)}


    def _load_annotations(self,mode):
        samples = []
        for vid_id in self.video_ids:
            # Video path etc videos/1 and its annotation file
            vid_path = os.path.join(self.videos_dir,str(vid_id))
            annot_file = os.path.join(vid_path,'annotations.txt')
            if not os.path.exists(annot_file): continue

            # Open its annotation file
            with open(annot_file,"r") as f:
                for line in f:
                    parts = line.strip().split()  #'23613 l_spike' -> [23613,"l_spike"]
                    if len(parts) < 2: continue
                    clip_id = parts[0].split('.')[0]  #'9575.jpg' -> '9575'
                    clip_label = parts[1].replace('-', '_') #'l-pass' -> 'l_pass'
                    if clip_label not in self.classes_to_idx: continue

                    label_clip_idx = self.classes_to_idx[clip_label]  # Converts a class string to its index
                    clip_path = os.path.join(vid_path,clip_id)
                    if not os.path.isdir(clip_path): continue

                    # Retrieve the frames sorted in a list
                    all_frames = sorted([
                        os.path.join(clip_path, f)
                        for f in os.listdir(clip_path)
                        if f.endswith('.jpg')
                    ], key=lambda x: int(os.path.basename(x).split('.')[0]))

                    # Take the middle 9 frames
                    mid_idx = len(all_frames) // 2
                    start_idx = max(0, mid_idx - 4)
                    end_idx = min(len(all_frames), mid_idx + 5)
                    window_frames = all_frames[start_idx:end_idx]

                    if mode == "spatial":
                        # Append each in samples list
                        for frame_path in window_frames:
                            samples.append({
                                "image_path": frame_path,
                                "label": label_clip_idx,
                                "video_id": vid_id,
                                "clip_id": clip_id
                            })
                    elif mode == "temporal":
                        # Append all at once as a list to samples
                        samples.append({
                            "sequence": window_frames,
                            "label": label_clip_idx,
                            "video_id": vid_id,
                            "clip_id": clip_id
                        })
        return samples

class SpatialScene(VolleyballSceneBase):
    def __init__(self,root_dir,split,transform=None):
        super().__init__(root_dir, split)
        self.transform = transform
        self.samples: List[Dict[str, Any]] = self._load_annotations("spatial")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        label = sample["label"]

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img , label
    def __len__(self):
        return len(self.samples)

class TemporalScene(VolleyballSceneBase):
    def __init__(self,root_dir,split,transform=None):
        super().__init__(root_dir, split)
        self.transform = transform
        self.samples: List[Dict[str, Any]] = self._load_annotations("temporal")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = sample["sequence"]
        label = sample["label"]
        imgs = []
        for photo_path in sequence:
            img = Image.open(photo_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        return imgs , label