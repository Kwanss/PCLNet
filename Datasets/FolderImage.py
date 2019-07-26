from __future__ import division

import os
import os.path
import os.path
import random
import math
import glob

import torch
import torch.utils.data as data

from .dataset_utils.util_func import *
from .dataset_utils import frame_utils
from . import dataset_read


class FolderImage(data.Dataset):
    """
    """

    def __init__(self, args, is_train, root=None, loader=default_loader,
            replicates=1):
        self.args = args
        self.is_train = is_train
        self.train_size = args.train_size
        self.render_size = args.render_size
        self.real_size = None
        self.replicates = replicates
        self.snippet_len = args.snippet_len
        self.K = args.K
        self.replicates = replicates
        
        frames_list = [root]
        class_list = [ 0 ]
        frames_num = [len(glob.glob(os.path.join(root, 'frame*.jpg')))]

        self.loader = loader
        self.frames_list = frames_list
        self.class_list = class_list
        self.frames_num = frames_num

        self.real_size = frame_utils.read_gen(self.frames_list[0] + "/frame000001.jpg").shape[:2]

        if self.render_size == [-1, -1]:
            # choice the closest size
            f_h, f_w = self.real_size[:2]

            min_h, min_w = math.floor(f_h / 64) * 64, math.floor(f_w / 64) * 64
            max_h, max_w = math.ceil(f_h / 64) * 64, math.ceil(f_w / 64) * 64

            re_h = min_h if (abs(min_h - f_h) <= abs(max_h - f_h)) else max_h
            re_w = min_w if (abs(min_w - f_w) <= abs(max_w - f_w)) else max_w
            self.render_size = [re_h, re_w]
        assert [self.render_size[0] % 64, self.render_size[0] % 64] == [0, 0]

        # Cautious!
        args.render_size = self.render_size
        args.real_size = self.real_size

        trans_size = self.train_size if self.is_train else self.render_size
        self.transform = get_transform_flow(trans_size=trans_size, is_train=self.is_train,
                sparse=False, div_flow=self.args.div_flow, ct_type=args.ct_type)

    def __getitem__(self, index):

        index = index % len(self.frames_list)
        frames_path, class_idx, frames_num = self.frames_list[index], self.class_list[index], self.frames_num[index]

        # K_clip_idxs = get_sample_index(frames_num, self.K, self.snippet_len, stride=self.args.stride)
        K_clip_idxs = [[i, i+ 1] for i in range(frames_num-1)][:self.K]
        K_clip_img = []

        read_paths = []
        for clip_idxs in K_clip_idxs:
            clip_paths = [os.path.join(frames_path, 'frame%06d.jpg' % (im_idx + 1)) for im_idx in clip_idxs]
            read_paths.append(clip_paths)

            clip_img = [np.array(self.loader(p)) for p in clip_paths]  # (frame_num, H,W,C)

            input_transform, target_transform, com_transform = self.transform
            clip_img, _ = com_transform(clip_img, None)
            clip_img = [input_transform(im) for im in clip_img]
            clip_img = torch.stack(clip_img)
            K_clip_img.append(clip_img)
        K_clip_img = torch.stack(K_clip_img, 0)

        # (K, snippet_len, C,  H,W)
        return {'frames': K_clip_img, 'classes': class_idx, 'paths': read_paths}

    def __len__(self):
        return self.replicates * len(self.frames_list)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
