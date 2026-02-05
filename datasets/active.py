import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from ipdb import set_trace as st
import joblib


# Training set: 57 subjects, Testing set: 23 subjects (total 80 subjects)
Cross_Subject = [
    1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    23, 24, 25, 26, 27, 28, 30, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 52, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 76, 77, 78, 79
]

def clip_normalize(clip, epsilon=1e-6):
    pc = clip.view(-1, 3) 
    centroid = pc.mean(dim=0)  
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    
    if m < epsilon:
        m = epsilon
    
    clip = (clip - centroid) / m
    return clip

class ACTIVESubject(Dataset):
    def __init__(self, root, meta, frames_per_clip=12, step_between_clips=1, num_points=768, train=True):
        super(ACTIVESubject, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        with open(meta, 'r') as f:
            for line in f:
                name, nframes_meta = line.strip().split()
                subject = int(name[1:4])
                nframes_meta = int(nframes_meta)

                if (train and subject in Cross_Subject) or (not train and subject not in Cross_Subject):
                    label = int(name[9:12]) - 1

                    video_file = os.path.join(root, name + '.pkl')
                    if not os.path.exists(video_file):
                        continue

                    if nframes_meta < 5:
                        continue
                    
                    try:
                        data_t = joblib.load(video_file) 
                        if data_t[0].shape[0] < 20:
                            continue

                    except Exception as e:
                        print(f"{e}")
                    nframes = nframes_meta

                    if nframes_meta < frames_per_clip:
                        self.index_map.append((index, 0))
                    else:
                        for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                            self.index_map.append((index, t))
                        
                    # st()
                    self.labels.append(label)
                    self.videos.append(video_file)
                    index += 1

        if len(self.labels) == 0:
            raise ValueError("No data found. Please check your dataset paths and meta file.")

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)


    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video_path = self.videos[index]
        try:
            video = joblib.load(video_path)
        except Exception as e:
            return None

        label = self.labels[index]

        total_frames = len(video)
        required_frames = self.frames_per_clip
        step = self.step_between_clips

        clip = []

        for i in range(required_frames):
            frame_idx = t + i * step
            if frame_idx < total_frames:
                p = torch.tensor(video[frame_idx]).float()
            else:
                if total_frames > 0:
                    if frame_idx - (total_frames - 1) <= 1:
                        p = torch.tensor(video[-1]).float()
                    elif t < total_frames:
                        p = torch.tensor(video[t]).float()
                    else:
                        p = torch.zeros((self.num_points, video[0].shape[1]))
                else:
                    p = torch.zeros((self.num_points, video[0].shape[1]))
            clip.append(p)

        for i in range(len(clip)):
            p = clip[i]
            if p.shape[0] >= self.num_points:
                r = torch.randperm(p.shape[0])[:self.num_points]
                clip[i] = p[r, :]
            elif p.shape[0] == 0:
                p = torch.zeros((self.num_points, p.shape[1]))
                clip[i] = p
            else:
                repeat, residue = divmod(self.num_points, p.shape[0])
                if p.shape[0] > 0:
                    r = torch.cat([
                        torch.arange(p.shape[0]).repeat(repeat),
                        torch.randint(0, p.shape[0], (residue,))
                    ])
                    clip[i] = p[r, :]
                else:
                    p = torch.zeros((self.num_points, p.shape[1]))
                    clip[i] = p

        clip = clip_normalize(torch.stack(clip))

        if self.train:
            scales = torch.FloatTensor(3).uniform_(0.9, 1.1)
            clip = clip * scales

        return clip.float(), label, index