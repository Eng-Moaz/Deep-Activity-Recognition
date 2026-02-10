import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


# =============================================================================
# 1. SCENE DATASET (Stage 2 & Baselines)
# Focus: The whole court, or all 12 players at a specific time step.
# Modes: 'scenecrops' (Stage 2), 'scenefull' (Baseline 1), 'scenefull_temporal'
# =============================================================================
class VolleyballSceneDataset(Dataset):
    def __init__(self, root_dir, split, mode="scenecrops", transform=None, seq_len=9):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.max_players = 12
        self.seq_len = seq_len

        # [FIX] Hardcoded Paths
        self.videos_dir = "/kaggle/input/volleyball/volleyball_/videos"
        self.tracks_dir = "/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation"

        if not os.path.exists(self.tracks_dir):
            self.tracks_dir = "/kaggle/input/volleyball/volleyball_tracking_annotation"

        self.split_ids = {
            'train': [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }

        self.scene_classes = ['l_pass', 'r_pass', 'l_spike', 'r_spike', 'l_set', 'r_set', 'l_winpoint', 'r_winpoint']
        self.scene_to_idx = {cls: i for i, cls in enumerate(self.scene_classes)}

        print(f"[{split.upper()}] Loading SCENE Data (Mode: {mode})...")
        self.samples = self._load_data()
        print(f"[{split.upper()}] Loaded {len(self.samples)} scene samples.")

    def _load_data(self):
        samples = []
        target_vids = self.split_ids.get(self.split, [])

        for vid_id in target_vids:
            # 1. Get Scene Labels
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

            # 2. Get Tracking Data (Only needed for 'scenecrops')
            vid_track_dir = os.path.join(self.tracks_dir, str(vid_id))
            if not os.path.isdir(vid_track_dir): continue

            for clip_id in os.listdir(vid_track_dir):
                if clip_id not in scene_labels: continue

                # --- MODE: TEMPORAL SCENE (Full Images Sequence) ---
                if self.mode == 'scenefull_temporal':
                    center_frame = int(clip_id)
                    mid = self.seq_len // 2
                    start = center_frame - mid
                    end = center_frame + mid

                    paths = []
                    valid = True
                    for fid in range(start, end + 1):
                        p = os.path.join(self.videos_dir, str(vid_id), clip_id, f"{fid}.jpg")
                        if not os.path.exists(p):
                            valid = False
                            break
                        paths.append(p)

                    if valid:
                        samples.append({'img_paths': paths, 'label': scene_labels[clip_id]})
                    continue

                    # --- MODE: SPATIAL SCENE (Crops or Single Full Image) ---
                track_file = os.path.join(vid_track_dir, clip_id, f"{clip_id}.txt")
                if not os.path.exists(track_file): continue

                # Parse Tracking for this clip
                frames_data = defaultdict(list)
                with open(track_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        try:
                            fid = int(parts[5])
                            lost = int(parts[6])
                            if lost == 1: continue
                            box = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                            frames_data[fid].append(box)
                        except:
                            pass

                # Generate Samples (Center +/- 4 frames)
                center_frame = int(clip_id)
                start_win = center_frame - 4
                end_win = center_frame + 4

                for fid in frames_data.keys():
                    if self.split == 'train':
                        if not (start_win <= fid <= end_win): continue
                    else:
                        if fid != center_frame: continue

                    if not frames_data[fid]: continue

                    img_path = os.path.join(self.videos_dir, str(vid_id), clip_id, f"{fid}.jpg")
                    if not os.path.exists(img_path): continue

                    if self.mode == 'scenecrops':
                        samples.append({
                            'img_path': img_path,
                            'players': frames_data[fid],  # List of boxes
                            'label': scene_labels[clip_id]
                        })
                    elif self.mode == 'scenefull':
                        samples.append({
                            'img_path': img_path,
                            'label': scene_labels[clip_id]
                        })
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Temporal Mode
        if self.mode == 'scenefull_temporal':
            imgs = []
            try:
                for p in sample['img_paths']:
                    with Image.open(p) as img:
                        img = img.convert("RGB")
                        if self.transform: img = self.transform(img)
                        imgs.append(img)
                return torch.stack(imgs), sample['label']
            except:
                return torch.zeros(self.seq_len, 3, 224, 224), 0

        # Spatial Modes
        try:
            with Image.open(sample['img_path']) as img:
                img = img.convert("RGB")

                if self.mode == 'scenecrops':
                    crops = []
                    for box in sample['players']:
                        box = self._clamp(box, img.width, img.height)
                        if self._valid(box):
                            c = img.crop(box)
                            if self.transform: c = self.transform(c)
                            crops.append(c)
                    # Pad
                    if len(crops) > self.max_players: crops = crops[:self.max_players]
                    while len(crops) < self.max_players: crops.append(torch.zeros(3, 224, 224))
                    return torch.stack(crops), sample['label']

                elif self.mode == 'scenefull':
                    if self.transform: img = self.transform(img)
                    return img, sample['label']
        except:
            return torch.zeros(12, 3, 224, 224), 0

    def _clamp(self, box, w, h):
        return (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))

    def _valid(self, box):
        return box[2] > box[0] and box[3] > box[1]

    def __len__(self):
        return len(self.samples)


# =============================================================================
# 2. PLAYER DATASET (Stage 1 & LSTM)
# Focus: Individual players.
# Modes: 'action_train' (Stage 1), 'temporal' (LSTM on single player tracks)
# =============================================================================
class VolleyballPlayerDataset(Dataset):
    def __init__(self, root_dir, split, mode="action_train", transform=None, seq_len=9):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len

        # [FIX] Hardcoded Paths
        self.videos_dir = "/kaggle/input/volleyball/volleyball_/videos"
        self.tracks_dir = "/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation"

        if not os.path.exists(self.tracks_dir):
            self.tracks_dir = "/kaggle/input/volleyball/volleyball_tracking_annotation"

        self.split_ids = {
            'train': [1, 3, 6, 7, 10, 13, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 15, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }

        self.action_classes = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing',
                               'waiting']
        self.action_to_idx = {cls: i for i, cls in enumerate(self.action_classes)}

        print(f"[{split.upper()}] Loading PLAYER Data (Mode: {mode})...")
        self.samples = self._load_data()
        print(f"[{split.upper()}] Loaded {len(self.samples)} player samples.")

    def _load_data(self):
        samples = []
        target_vids = self.split_ids.get(self.split, [])

        for vid_id in target_vids:
            vid_track_dir = os.path.join(self.tracks_dir, str(vid_id))
            if not os.path.isdir(vid_track_dir): continue

            for clip_id in os.listdir(vid_track_dir):
                track_file = os.path.join(vid_track_dir, clip_id, f"{clip_id}.txt")
                if not os.path.exists(track_file): continue

                # We need to group data by PLAYER ID to form temporal sequences
                # player_tracks[pid] = [ {path, box, label, fid}, ... ]
                player_tracks = defaultdict(list)

                with open(track_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        try:
                            # [PID, x1, y1, x2, y2, FID, Lost, ..., Action]
                            pid = int(parts[0])
                            fid = int(parts[5])
                            lost = int(parts[6])
                            action_str = parts[9]

                            if lost == 1: continue
                            if action_str not in self.action_to_idx: continue

                            box = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                            action_label = self.action_to_idx[action_str]

                            img_path = os.path.join(self.videos_dir, str(vid_id), clip_id, f"{fid}.jpg")
                            if not os.path.exists(img_path): continue

                            player_tracks[pid].append({
                                'img_path': img_path,
                                'box': box,
                                'label': action_label,
                                'fid': fid
                            })
                        except:
                            pass

                # Now process tracks based on mode
                center_frame = int(clip_id)
                start_win = center_frame - 4
                end_win = center_frame + 4

                for pid, track in player_tracks.items():
                    # Sort by frame ID
                    track.sort(key=lambda x: x['fid'])

                    if self.mode == 'action_train':
                        # Add every valid frame as a sample
                        for frame in track:
                            # Window Logic (Train vs Val)
                            if self.split == 'train':
                                if not (start_win <= frame['fid'] <= end_win): continue
                            else:
                                if frame['fid'] != center_frame: continue

                            samples.append(frame)

                    elif self.mode == 'temporal':
                        # Create a sequence sample (e.g. 9 consecutive frames of Player 1)
                        # We try to center it around the middle
                        if len(track) < self.seq_len: continue

                        # Find middle index
                        mid_idx = len(track) // 2
                        start_idx = max(0, mid_idx - self.seq_len // 2)
                        end_idx = min(len(track), start_idx + self.seq_len)

                        seq = track[start_idx:end_idx]
                        if len(seq) == self.seq_len:
                            # Label is the label of the middle frame
                            label = seq[len(seq) // 2]['label']
                            samples.append({
                                'sequence': seq,
                                'label': label
                            })
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.mode == 'temporal':
            crops = []
            try:
                for frame in sample['sequence']:
                    with Image.open(frame['img_path']) as img:
                        img = img.convert("RGB")
                        box = self._clamp(frame['box'], img.width, img.height)
                        if self._valid(box):
                            c = img.crop(box)
                            if self.transform: c = self.transform(c)
                            crops.append(c)
                        else:
                            # Invalid box inside sequence
                            crops.append(torch.zeros(3, 224, 224))
                return torch.stack(crops), sample['label']
            except:
                return torch.zeros(self.seq_len, 3, 224, 224), 0

        # 'action_train' (Spatial)
        try:
            with Image.open(sample['img_path']) as img:
                img = img.convert("RGB")
                box = self._clamp(sample['box'], img.width, img.height)

                if self._valid(box):
                    c = img.crop(box)
                else:
                    return torch.zeros(3, 224, 224), 0

                if self.transform: c = self.transform(c)
                return c, sample['label']
        except:
            return torch.zeros(3, 224, 224), 0

    def _clamp(self, box, w, h):
        return (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))

    def _valid(self, box):
        return box[2] > box[0] and box[3] > box[1]

    def __len__(self):
        return len(self.samples)