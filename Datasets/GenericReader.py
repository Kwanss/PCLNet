from __future__ import division

import os
import os.path
import os.path
import random
from glob import glob
from os.path import *

import torch
import torch.utils.data as data

from .dataset_utils import frame_utils
from .dataset_utils.util_func import *


class ImagePaths(data.Dataset):
    def __init__(self, path_list, class_idx_list, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None):
        samples = make_dataset(path_list, class_idx_list, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files !\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class VideoAsFrames(ImagePaths):
    def __init__(self, video_dir_list, class_idxs_list, frames_num_list, shuffle=False, img_format='jpg',
                 transform=None, target_transform=None,
                 loader=default_loader):
        assert len(video_dir_list) == len(class_idxs_list) == len(frames_num_list)
        all_images = []
        all_class_idxs = []
        for i, vp in enumerate(video_dir_list):
            for f_ind in range(frames_num_list[i]):
                all_images.append(os.path.join(vp, "frame%06d.%s" % (f_ind + 1, img_format)))
                all_class_idxs.append(class_idxs_list[i])
        if shuffle:
            ind = list(range(len(all_images)))
            random.shuffle(ind)
            all_images = [all_images[k] for k in ind]
            all_class_idxs = [all_class_idxs[t] for t in ind]

        super(VideoAsFrames, self).__init__(all_images, all_class_idxs, loader, IMG_EXTENSIONS,
                                            transform=transform,
                                            target_transform=target_transform) 


class VideoAsTSN(data.Dataset):
    """
    Split the video as K segments, and then randomly choose one snippet from each segment,
    Noted that 'shuffle' here will only shuffle as video, frames in a snippet are in order.
    if merged is False, return (K, snippet_len, C, H, W), class_index
    else return (K * snippet_len, C, H, W)
    """

    def __init__(self, args, video_dir_list, class_idxs_list, frames_num_list, K=8, snippet_len=5,
                 shuffle=False, merged=True, img_format='jpg', transform=None, target_transform=None,
                 loader=default_loader, group_transform=False):
        assert len(video_dir_list) == len(class_idxs_list) == len(frames_num_list)
        self.args = args

        if shuffle:
            ind = list(range(len(video_dir_list)))
            random.shuffle(ind)
            video_dir_list = [video_dir_list[v] for v in ind]
            class_idxs_list = [class_idxs_list[v] for v in ind]
            frames_num_list = [frames_num_list[v] for v in ind]
        self.videos = video_dir_list
        self.class_idxs = class_idxs_list
        self.frame_num = frames_num_list
        self.group_transform = group_transform
        self.loader = loader
        self.merged = merged

        self.transform = transform
        self.target_transform = target_transform
        self.K = K
        self.snippet_len = snippet_len
        self.img_format = img_format

    def __getitem__(self, index):
        v_path, v_class, v_fnum = self.videos[index], self.class_idxs[index], self.frame_num[index]
        img_idxs = get_sample_index(v_fnum, self.K, self.snippet_len, stride=self.args.stride)

        if self.merged:
            loaded_imgs = []
            for snp_idx in img_idxs:
                for im_idx in snp_idx:
                    loaded_imgs.append(self.loader(
                        os.path.join(v_path, "frame%06d.%s" % (im_idx + 1, self.img_format))))

            if self.transform is not None:
                if self.group_transform:
                    loaded_imgs = self.transform(loaded_imgs)
                else:
                    loaded_imgs = [self.transform(I) for I in loaded_imgs]
                seg_imgs = torch.stack(loaded_imgs)
            else:
                seg_imgs = loaded_imgs

        else:
            seg_imgs = []
            for snp_idx in img_idxs:
                snp_img = []
                for im_idx in snp_idx:
                    snp_img.append(self.loader(
                        os.path.join(v_path, "frame%06d.%s" % (im_idx + 1, self.img_format))))
                if self.transform is not None:
                    if self.group_transform:
                        snp_img = self.transform(snp_img)
                    else:
                        snp_img = [self.transform(I) for I in snp_img]
                    seg_imgs.append(torch.stack(snp_img))
                else:
                    seg_imgs.append(snp_img)
            seg_imgs = torch.stack(seg_imgs)

        if self.target_transform is not None:
            v_class = self.target_transform(v_class)

        return seg_imgs, v_class

    def __len__(self):
        return len(self.videos)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str





class ImagesFromFolder(data.Dataset):
    def __init__(self, args, is_cropped, root='/path/to/frames/only/folder', iext='png', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        images = sorted(glob(join(root, '*.' + iext)))
        self.image_list = []
        for i in range(len(images) - 1):
            im1 = images[i]
            im2 = images[i + 1]
            self.image_list += [[im1, im2]]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))

        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))

        return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

    def __len__(self):
        return self.size * self.replicates