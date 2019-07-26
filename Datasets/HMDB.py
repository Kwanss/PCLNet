from __future__ import division

import os
import os.path
import os.path

import torch
import torch.utils.data as data

from .dataset_utils.util_func import *
from . import dataset_read


class HMDBAsClips(data.Dataset):
    """
    """

    def __init__(self, args, is_train, root=None, crop_size=224, snippet_len=2, shuffle=False, K=1,
                 loader=default_loader, replicates=1, div_flow=1.0, val_full=False):
        self.args = args
        if self.args.inference:
            replicates = 1
            shuffle = False
            print("[>>> Inference mode:  replicates: 1, shuffle: False <<<]")
        train_set, validation_set = dataset_read.get_ucf101_info(root=root)
        video_info = train_set if is_train else validation_set
        frames_list, class_list, frames_num = video_info
        assert len(frames_list) == len(class_list) == len(frames_num)
        assert mode in ['train', 'validation']

        if shuffle:
            ind = list(range(len(frames_list)))
            random.shuffle(ind)
            frames_list = [frames_list[v] for v in ind]
            class_list = [class_list[v] for v in ind]
            frames_num = [frames_num[v] for v in ind]
        self.loader = loader
        self.frames_list = frames_list
        self.class_list = class_list
        self.frames_num = frames_num

        self.snippet_len = snippet_len
        self.crop_size = crop_size
        self.replicates = replicates
        self.mode = mode
        self.K = K
        self.transform = get_transform(trans_size=crop_size, is_train=(mode == 'train'), div_flow=div_flow, cr_rate=0.2,
                                       val_full=val_full)

    def __getitem__(self, index):

        index = index % len(self.frames_list)
        frames_path, class_idx, frames_num = self.frames_list[index], self.class_list[index], self.frames_num[index]

        if self.args.inference:
            K_clip_idxs = [list(range(begin, begin + self.snippet_len)) for begin in
                           range(0, frames_num, self.snippet_len)]
            tmp = K_clip_idxs[-1]
            tmp = [(ind if ind < self.snippet_len else self.snippet_len - 1) for ind in tmp]
            K_clip_idxs[-1] = tmp
        else:
            K_clip_idxs = get_sample_index(frames_num, self.K, self.snippet_len, stride=self.args.stride)
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
        return K_clip_img, class_idx, read_paths

    def __len__(self):
        return self.replicates * len(self.frames_list)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
