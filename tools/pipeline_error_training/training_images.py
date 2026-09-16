"""Reuse exact immutable-frame quantiles across bounded training image readers."""
from collections import OrderedDict

import numpy as np

from .common import sha
from .crops import Images


class FrameStatistics:
    def __init__(self,max_entries=12800):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.computed = self.reused = 0

    def quantiles(self,key,frame_sha256,frame):
        if key in self.entries:
            expected,quantiles = self.entries[key]
            if frame_sha256!=expected:
                raise ValueError('Training image bytes changed after quantile calculation')
            self.entries.move_to_end(key)
            self.reused += 1
            return quantiles
        quantiles = tuple(np.quantile(frame,[.01,.99]))
        self.entries[key] = (frame_sha256,quantiles)
        self.computed += 1
        while len(self.entries)>self.max_entries:
            self.entries.popitem(last=False)
        return quantiles


class TrainingImages(Images):
    def __init__(self,path,statistics,max_frames=9):
        super().__init__(path,max_frames=max_frames)
        self.statistics = statistics
        self.statistics_key = str(self.path.resolve())

    def raw(self,t):
        if not 0<=t<self.shape[0]:
            return None
        if t in self.frames:
            self.frames.move_to_end(t)
            return self.frames[t]
        chunk = self.path/f'0/c/{t}/0/0/0'
        if not chunk.is_file() or not chunk.stat().st_size:
            raise FileNotFoundError(f'Missing image frame chunk: {chunk}')
        frame = np.asarray(self.array[t])
        self.read_hashes[t] = sha(chunk)
        self.quantiles[t] = self.statistics.quantiles((self.statistics_key,int(t)),self.read_hashes[t],frame)
        self.frames[t] = frame
        while len(self.frames)>self.max_frames:
            self.frames.popitem(last=False)
        return frame
