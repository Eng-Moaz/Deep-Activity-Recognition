import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


class VolleyballSceneDataset(Dataset):
    def __init__(self, root_dir, split, mode, seq_len, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.max_players = 12
        self.seq_len = seq_len

        # Paths
        self.videos_dir = "/kaggle/input/volleyball/volleyball_/videos"
        self.tracks_dir = "/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation"

        # Safety Check
        if not os.path.exists(self.tracks_dir):
            print(f"Tracking path not found at: {self.tracks_dir}")

        # Splits
        self.split_ids = {
            'train': [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }

        # Full Scene Classes (8 Classes)
        self.scene_classes = ['l_pass', 'r_pass', 'l_spike', 'r_spike', 'l_set', 'r_set', 'l_winpoint', 'r_winpoint']
        self.scene_to_idx = {cls: i for i, cls in enumerate(self.scene_classes)}

        # Player Actions Classes (8 Classes)
        self.action_classes = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing',
                               'waiting']
        self.action_to_idx = {cls: i for i, cls in enumerate(self.action_classes)}

        print(f"[{split.upper()}] Loading Data from:")
        print(f"   > Tracks: {self.tracks_dir}")

        self.samples = self._load_data()

        # Verification
        if len(self.samples) > 0:
            print(f"[{split.upper()}] Loaded {len(self.samples)} samples.")
        else:
            print(f"[{split.upper()}] Error , Loaded 0 samples.")

    def _load_data(self):
        samples = []
        target_vids = self.split_ids.get(self.split, [])

        for vid_id in target_vids:
            # Scene Labels
            scene_labels = {}
            vid_annot_path = os.path.join(self.videos_dir, str(vid_id), 'annotations.txt')

            if os.path.exists(vid_annot_path):
                with open(vid_annot_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            clip_id = parts[0].split('.')[0]
                            label_str = parts[1].replace('-', '_')
                            if label_str in self.scene_to_idx:
                                scene_labels[clip_id] = self.scene_to_idx[label_str]

            # Tracking Boxes
            vid_track_dir = os.path.join(self.tracks_dir, str(vid_id))
            if not os.path.isdir(vid_track_dir): continue

            for clip_id in os.listdir(vid_track_dir):
                if clip_id not in scene_labels: continue

                track_file = os.path.join(vid_track_dir, clip_id, f"{clip_id}.txt")
                if not os.path.exists(track_file): continue

                # Parse Tracking
                frames_data = defaultdict(list)
                with open(track_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()

                        try:
                            # [PlayerID, x1, y1, x2, y2, FrameID, Lost, ..., Action]
                            fid = int(parts[5])
                            lost = int(parts[6])
                            action_str = parts[9]
                            if lost == 1: continue

                            box = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                            action_label = self.action_to_idx.get(action_str, 0)

                            frames_data[fid].append({
                                'box': box,
                                'action_label': action_label
                            })
                        except:
                            pass

                # Generate Samples
                center_frame = int(clip_id)
                mid = self.seq_len // 2
                start_window = center_frame - mid
                end_window = center_frame + mid

                for fid in frames_data.keys():
                    # Window Logic
                    if self.split == 'train':
                        if not (start_window <= fid <= end_window): continue
                    else:
                        if fid != center_frame: continue

                    if len(frames_data[fid]) == 0: continue

                    img_path = os.path.join(self.videos_dir, str(vid_id), clip_id, f"{fid}.jpg")
                    if not os.path.exists(img_path): continue

                    # All players once as a batch
                    if self.mode == 'scenecrops':
                        player_boxes = [p['box'] for p in frames_data[fid]]
                        samples.append({
                            'img_path': img_path,
                            'players': player_boxes,
                            'label': scene_labels[clip_id]
                        })

                    # Each player on his own
                    elif self.mode == 'action_train':
                        for p in frames_data[fid]:
                            samples.append({
                                'img_path': img_path,
                                'box': p['box'],
                                'label': p['action_label']
                            })

                    # Full scene
                    elif self.mode == 'scenefull':
                        samples.append({
                            'img_path': img_path,
                            'label': scene_labels[clip_id]
                        })

                    elif self.mode == "scenefull_temporal":
                        img_paths = [img_path for img_path in range(start_window,end_window+1)]
                        samples.append({
                            'img_paths': img_paths,
                            'label': scene_labels[clip_id]
                        })

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            with Image.open(sample['img_path']) as img:
                img = img.convert("RGB")

                if self.mode == 'scenecrops':
                    crops = []
                    for box in sample['players']:
                        box = self._clamp_box(box, img.width, img.height)
                        if self._is_valid(box):
                            c = img.crop(box)
                            if self.transform: c = self.transform(c)
                            crops.append(c)
                    if len(crops) > self.max_players: crops = crops[:self.max_players]
                    while len(crops) < self.max_players: crops.append(torch.zeros(3, 224, 224))
                    return torch.stack(crops), sample['label']

                elif self.mode == 'action_train':
                    box = self._clamp_box(sample['box'], img.width, img.height)
                    c = img.crop(box) if self._is_valid(box) else torch.zeros(3, 224, 224)
                    if self.transform: c = self.transform(c)
                    return c, sample['label']

                elif self.mode == 'scenefull':
                    if self.transform: img = self.transform(img)
                    return img, sample['label']

        except:
            if self.mode == 'scenecrops': return torch.zeros(12, 3, 224, 224), 0
            return torch.zeros(3, 224, 224), 0

        if self.mode == "scenefull_temporal":
            imgs = [Image.open(img) for img in sample["img_paths"]]
            label = sample["label"]
            return imgs , label

    def _clamp_box(self, box, w, h):
        return (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))

    def _is_valid(self, box):
        return box[2] > box[0] and box[3] > box[1]

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

        self.classes = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing',
                        'waiting']
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
                if not os.path.exists(annot_file): continue

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
                            if lost == 1: continue
                            if action not in self.class_to_idx: continue

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
                if self.transform: crop = self.transform(crop)
                return crop
        except:
            return torch.zeros(3, 224, 224)