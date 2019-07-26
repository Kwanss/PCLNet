from __future__ import division

import datetime
import math
import os
import os.path
import os.path
from glob import glob
from os.path import *
import random

import torch
import torch.utils.data as data

from .dataset_utils import frame_utils
from .dataset_utils.util_func import *


class MpiSintel(data.Dataset):
    def __init__(self, args, is_train, root='', pass_name='clean', replicates=1):
        print("MPI pass:\t", pass_name)
        self.args = args
        self.is_train = is_train
        self.train_size = args.train_size
        self.render_size = args.render_size
        self.real_size = None
        self.replicates = replicates
        self.is_test = False
        assert len(args.train_size) == len(args.render_size) == 2

        flow_root = join(root, 'flow')
        image_root = join(root, pass_name)

        if os.path.exists(flow_root):
            file_list = sorted(glob(join(flow_root, '*/*.flo')))

            self.flow_list = []
            self.image_list = []

            for flo_path in file_list:
                fbase = flo_path[len(flow_root) + 1:]
                fprefix = fbase[:-8]
                fnum = int(fbase[-8:-4])

                img1 = join(image_root, fprefix + "%04d" % (fnum + 0) + '.png')
                img2 = join(image_root, fprefix + "%04d" % (fnum + 1) + '.png')

                self.image_list += [[img1, img2]]
                self.flow_list += [flo_path]
        else:
            # test
            self.is_test = True
            self.image_list = []
            self.flow_list = []

            f_dirs = os.listdir(image_root)
            for fd in f_dirs:
                frames_list = sorted(glob(join(image_root, fd, '*.png')))
                for i in range(len(frames_list) - 1):
                    self.image_list += [[frames_list[i], frames_list[i + 1]]]

        if self.args.split_mpi:
            prefix = self.image_list[0][0].split('training/')[0]
            if self.args.split_mpi_path is None:
                print("Use combined list, MPI")
                split_dir = './split_files/MPI_split'
                path_info = []
                for dt in ['clean', 'final']:
                    path_info += [p.strip().split() for p in open(os.path.join(split_dir, 
                        'mpi_train_%s_split%d.txt' % (dt, 1 if self.is_train else 2)))]
            else:
                path_info = [p.strip().split() for p in open(os.path.join(self.args.split_mpi_path,
                    'mpi_train_%s_split%d.txt' % (pass_name, 1 if self.is_train else 2)))]
            self.image_list = [[prefix + p1, prefix + p2] for p1, p2, _ in path_info]
            self.flow_list = [prefix + p for _, _, p in path_info]
            print("use MPI  split, num_img: %d num_flo: %d" % (len(self.image_list), len(self.flow_list)))

        # log path files
        save_path = args.save if args.save_flow_path is None else args.save_flow_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, 'input_path_img.log'), 'w') as f:
            for p1, p2 in self.image_list:
                f.write(p1 + ',' + p2 + '\n')
        with open(os.path.join(save_path, 'input_path_flo.log'), 'w') as f:
            for p in self.flow_list:
                f.write(p + '\n')
        print("input_path logged to: ", args.save)


        self.real_size = frame_utils.read_gen(self.image_list[0][0]).shape[:2]
        
        if self.render_size == [-1, -1]:
            
            # choice the closest size
            f_h, f_w = self.real_size[:2]

            min_h, min_w = math.floor(f_h / 64) * 64, math.floor(f_w / 64) * 64
            max_h, max_w = math.ceil(f_h / 64) * 64, math.ceil(f_w / 64) * 64

            re_h = min_h if (abs(min_h - f_h) <= abs(max_h - f_h)) else max_h
            re_w = min_w if (abs(min_w - f_w) <= abs(max_w - f_w)) else max_w
            self.render_size = [re_h, re_w]

            """
            # choice the largest size  
            self.render_size[0] = ( (self.real_size[0])//64 ) * 64
            self.render_size[1] = ( (self.real_size[1])//64 ) * 64
            """

        assert [self.render_size[0] % 64, self.render_size[0] % 64] == [0, 0]

        trans_size = self.train_size if self.is_train else self.render_size
        self.transform = get_transform_flow(trans_size=trans_size, is_train=self.is_train, sparse=False,
                                            div_flow=self.args.div_flow, ct_type=args.ct_type)
        if not self.is_test:
            assert (len(self.image_list) == len(self.flow_list))

        # Cautious!
        args.render_size = self.render_size
        args.real_size = self.real_size

    def __getitem__(self, index):

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        if not self.is_test:
            flow = frame_utils.read_gen(self.flow_list[index])
        else:
            # random
            flow = np.zeros_like(img1)[:, :, :2]

        images = [img1, img2]
        input_transform, target_transform, com_transform = self.transform

        images, flow = com_transform(images, [flow])
        images = torch.stack((input_transform(images[0]), input_transform(images[1])), dim=0) # Modified
        flow = target_transform(flow[0])

        return {'frames': images, 'flows': flow}

    def __len__(self):
        return len(self.image_list) * self.replicates


class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_train, root='', replicates=1):
        super(MpiSintelClean, self).__init__(args, is_train=is_train, root=root, pass_name='clean',
                                             replicates=replicates)


class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_train, root='', replicates=1):
        super(MpiSintelFinal, self).__init__(args, is_train=is_train, root=root, pass_name='final',
                                             replicates=replicates)


