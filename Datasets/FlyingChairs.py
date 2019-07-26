from __future__ import division

import datetime
import math
import os
import os.path
from glob import glob
import random

import torch
import torch.utils.data as data
from os.path import join

from .dataset_utils.util_func import *
from .dataset_utils import frame_utils


class FlyingChairs(data.Dataset):
    def __init__(self, args, is_train, root='/path/to/FlyingChairs_release/data', replicates=1):
        self.args = args
        self.is_train = is_train
        self.train_size = args.train_size
        self.render_size = args.render_size
        self.real_size = None
        self.replicates = replicates

        images = sorted(glob(join(root, '*.ppm')))

        self.flow_list = sorted(glob(join(root, '*.flo')))

        assert (len(images) // 2 == len(self.flow_list))

        self.image_list = []
        for i in range(len(self.flow_list)):
            im1 = images[2 * i]
            im2 = images[2 * i + 1]
            self.image_list += [[im1, im2]]
        
        assert len(self.image_list) == len(self.flow_list)
        # shuffle
        inds = list(range(len(self.image_list)))
        random.shuffle(inds)
        self.image_list = [a for a in self.image_list]
        self.flow_list = [a for a in self.flow_list]

        self.real_size = frame_utils.read_gen(self.image_list[0][0]).shape[:2]

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
        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):
        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        input_transform, target_transform, com_transform = self.transform

        images, flow = com_transform(images, [flow])
        images = torch.stack((input_transform(images[0]), input_transform(images[1])), dim=0)  # Modified
        flow = target_transform(flow[0])

        return {'frames': images, 'flows': flow}

    def __len__(self):
        return len(self.image_list) * self.replicates
