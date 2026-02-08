import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class VolleyballSceneDataset(Dataset):
    def __init__(self, root_dir, split, mode="scenecrops", transform=None, seq_len=9):

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.max_players = 12
        self.seq_len = seq_len

        self.videos_dir = os.path.join(root_dir, "videos")

        # Splits
        self.split_ids = {
            'train': [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }

        # Scene Labels (8 Classes)
        self.scene_classes = ['l-pass', 'r-pass', 'l-spike', 'r-spike',
                              'l-set', 'r-set', 'l-winpoint', 'r-winpoint']
        self.scene_to_idx = {cls: i for i, cls in enumerate(self.scene_classes)}

        # Action Labels (9 Classes)
        self.action_classes = ['blocking', 'digging', 'falling', 'jumping',
                               'moving', 'setting', 'spiking', 'standing', 'waiting']
        self.action_to_idx = {cls: i for i, cls in enumerate(self.action_classes)}

        self.samples = self._load_data()
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples (Mode: {mode})")

    def _load_data(self):
        samples = []
        target_vids = self.split_ids[self.split]

        for vid_id in target_vids:
            vid_path = os.path.join(self.videos_dir, str(vid_id))
            if not os.path.isdir(vid_path): continue

            annot_file = os.path.join(vid_path, 'annotations.txt')
            if not os.path.exists(annot_file): continue

            with open(annot_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue

                    filename = parts[0]
                    img_path = os.path.join(vid_path, filename)

                    # Scene Label
                    scene_label_str = parts[1]
                    if scene_label_str not in self.scene_to_idx: continue
                    scene_label = self.scene_to_idx[scene_label_str]

                    # Parse Players (x, y, w, h, action)
                    raw_data = parts[2:]
                    player_data = []

                    for i in range(0, len(raw_data), 5):
                        try:
                            # Coordinates
                            x, y, w, h = int(raw_data[i]), int(raw_data[i + 1]), int(raw_data[i + 2]), int(raw_data[i + 3])
                            # Action Label
                            action_str = raw_data[i + 4]
                            if action_str not in self.action_to_idx: continue
                            action_label = self.action_to_idx[action_str]

                            box = (x, y, x + w, y + h)  # X1 Y1 X2 Y2
                            player_data.append({
                                'box': box,
                                'action_label': action_label
                            })
                        except:
                            pass

                    # Action Training (Single Player)
                    if self.mode == "action_train":
                        for p in player_data:
                            samples.append({
                                'img_path': img_path,
                                'box': p['box'],
                                'label': p['action_label']
                            })

                    # Scene Inference (Group players)
                    elif self.mode == "scenecrops":
                        samples.append({
                            'img_path': img_path,
                            'players': player_data,
                            'label': scene_label
                        })

                    # Full Image
                    elif self.mode == "scenefull":
                        samples.append({
                            'img_path': img_path,
                            'label': scene_label
                        })

                    # Temporal Sequence (Full Image)
                    elif self.mode == "temporal":
                        # We have the middle frame (10535.jpg)
                        # We need to calculate neighbors: 10531...10539
                        try:
                            fid = int(filename.split('.')[0])  # 10535

                            # Calculate window range
                            half_window = self.seq_len // 2
                            start_fid = fid - half_window
                            end_fid = fid + half_window + 1  # +1 for range

                            frames = []
                            for i in range(start_fid, end_fid):
                                # Reconstruct path: videos/1/10531.jpg
                                frame_name = f"{i}.jpg"
                                frame_path = os.path.join(vid_path, frame_name)
                                frames.append(frame_path)

                            samples.append({
                                'frames': frames,  # List of seq_length (9) paths
                                'label': scene_label
                            })
                        except:
                            continue

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Action Training (1 Player and its action label)
        if self.mode == "action_train":
            img = Image.open(sample['img_path']).convert("RGB")
            crop = img.crop(sample['box'])
            if self.transform: crop = self.transform(crop)
            return crop, sample['label']  # Shape: [3, 224, 224]

        # Scene Inference (Stack of 12 Players and the scene label)
        elif self.mode == "scenecrops":
            img = Image.open(sample['img_path']).convert("RGB")
            crops = []
            for p in sample['players']:
                c = img.crop(p['box'])
                if self.transform: c = self.transform(c)
                crops.append(c)

            # Pad/Truncate to max_players (12)
            while len(crops) < self.max_players: crops.append(torch.zeros(3, 224, 224))
            return torch.stack(crops[:self.max_players]), sample['label']  # Shape: [12, 3, 224, 224]

        # Full Image
        elif self.mode == "scenefull":
            img = Image.open(sample['img_path']).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, sample['label'] # Shape: [3, 224, 224]

        # Temporal Sequence (Stack of 9 frames)
        elif self.mode == "temporal":
            imgs = []
            for path in sample['frames']:
                try:
                    img = Image.open(path).convert("RGB")
                    if self.transform: img = self.transform(img)
                    imgs.append(img)
                except:
                    # Safety: If a neighbor frame is missing, return black frame
                    imgs.append(torch.zeros(3, 224, 224))

            # Check length just in case
            if len(imgs) == 0: return torch.zeros(self.seq_len, 3, 224, 224), sample['label']

            return torch.stack(imgs), sample['label']  # Shape: [9, 3, 224, 224]

    def __len__(self):
        return len(self.samples)

class VolleyballPlayerDataset(Dataset):
    def __init__(self, root_dir, split, mode="spatial", transform=None, seq_len=9):

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len


        self.images_root = os.path.join(root_dir, "videos")

        self.annot_root = os.path.join(root_dir, "volleyball_tracking_annotation", "volleyball_tracking_annotation")

        self.classes = ['blocking', 'digging', 'falling', 'jumping',
                        'moving', 'setting', 'spiking', 'standing', 'waiting']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.split_ids = {
            'train': [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }

        self.samples = self._load_data()
        print(f"[{split.upper()}] Player Dataset ({mode}): {len(self.samples)} samples")

    def _load_data(self):
        samples = []
        target_vids = self.split_ids[self.split]

        for vid in target_vids:
            vid_annot_dir = os.path.join(self.annot_root, str(vid))
            if not os.path.isdir(vid_annot_dir): continue

            for clip in os.listdir(vid_annot_dir):
                clip_annot_dir = os.path.join(vid_annot_dir, clip)

                annot_file = os.path.join(clip_annot_dir, f"{clip}.txt")

                if not os.path.exists(annot_file):
                    continue

                clip_img_dir = os.path.join(self.images_root, str(vid), clip)
                if not os.path.isdir(clip_img_dir): continue

                track_data = {}

                with open(annot_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 10: continue

                        try:
                            pid = int(parts[0])
                            x1, y1, x2, y2 = map(int, parts[1:5])
                            fid = int(parts[5])
                            lost = int(parts[6])
                            action = parts[9]

                            # Filter
                            if lost == 1: continue
                            if action not in self.class_to_idx: continue

                            # Store info
                            if pid not in track_data: track_data[pid] = []

                            img_path = os.path.join(clip_img_dir, f"{fid}.jpg")

                            track_data[pid].append({
                                "path": img_path,
                                "bbox": (x1, y1, x2, y2),
                                "label": self.class_to_idx[action],
                                "fid": fid
                            })
                        except:
                            continue

                # Create Samples from Tracks
                for pid, frames in track_data.items():
                    frames.sort(key=lambda x: x['fid'])
                    if not frames: continue

                    if self.mode == "spatial":
                        mid = len(frames) // 2
                        samples.append(frames[mid])

                    elif self.mode == "temporal":
                        if len(frames) < self.seq_len: continue
                        mid = len(frames) // 2
                        start = max(0, mid - self.seq_len // 2)
                        end = min(len(frames), start + self.seq_len)
                        seq = frames[start:end]
                        if len(seq) == self.seq_len:
                            samples.append(seq)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        if self.mode == "spatial":
            return self._process_frame(data), data['label']

        elif self.mode == "temporal":
            frames = []
            label = data[len(data) // 2]['label']
            for frame_info in data:
                frames.append(self._process_frame(frame_info))
            return torch.stack(frames), label

    def _process_frame(self, frame_info):
        path = frame_info['path']
        bbox = frame_info['bbox']
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                crop = img.crop(bbox)
                if self.transform:
                    crop = self.transform(crop)
                return crop
        except:
            return torch.zeros(3, 224, 224)
