from __future__ import division

import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from . import co_transforms


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(image_paths, class_idxs, extensions):
    assert len(image_paths) == len(class_idxs)
    images = []
    for i, fp in enumerate(image_paths):
        if has_file_allowed_extension(fp, extensions):
            item = (fp, class_idxs[i])
            images.append(item)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_sample_index(f_n, K, snippet_len, stride=1):
    assert f_n > 0

    snippet_len = snippet_len * stride
    if f_n >= K * snippet_len:
        avg_len = f_n // K
        begin_idx = np.multiply(list(range(K)), avg_len) + np.random.randint(avg_len - snippet_len + 1)
        idx_list = [list(range(bi, bi + snippet_len)) for bi in begin_idx]
    elif f_n >= K:
        sp_ind = np.array_split(range(f_n), K)
        sp_ind = [list(sp) for sp in sp_ind]
        idx_list = [lst + [lst[-1]] * (snippet_len - len(lst)) for lst in sp_ind]
    else:
        idx_list = np.sort(np.random.randint(f_n, size=K * snippet_len))
        sp_ind = np.array_split(idx_list, K)
        idx_list = [list(sp) for sp in sp_ind]
    if stride > 1:
        idx_list = [ind[::stride] for ind in idx_list]
    return idx_list


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


################## Get transforms ############################


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]

"""
def get_transform(trans_size, is_train=True, sparse=False, div_flow=20.0, cr_rate=0.1, 
        val_full=False, fix_size=[256, 320], pre_scal_size=256):
    input_transform = transforms.Compose([
        co_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Changeed for vgg usage

    ])
    target_transform = transforms.Compose([
        co_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0], std=[div_flow, div_flow])
    ])
    if is_train:
        com_transform = co_transforms.Compose([
            co_transforms.SmallEdgeScale(pre_scal_size),
            co_transforms.RandomCropResize(trans_size, cr_rate),
            # co_transforms.RandomVerticalFlip(),
            co_transforms.RandomHorizontalFlip()
        ])
    else:
        com_transform = co_transforms.Compose([
            co_transforms.FixSizeScale(fix_size) if val_full else co_transforms.CenterCrop(trans_size)
        ])
    return input_transform, target_transform, com_transform
"""


def get_transform_flow(trans_size, is_train=True, sparse=False, div_flow=1.0, ct_type='1'):
    input_transform = transforms.Compose([
        co_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    """
    if ct_type == '1':
        input_transform = transforms.Compose([
            co_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    elif ct_type=='2':
        input_transform = transforms.Compose([
            co_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])
    """

    target_transform = transforms.Compose([
        co_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0], std=[div_flow, div_flow])
    ])

    if is_train:
        ct = []
        if ct_type == '1' and not sparse:
            ct.append(co_transforms.RandomCropResize(trans_size, 0.2))
        else:
            ct.append(co_transforms.RandomCrop(trans_size))
        ct.append(co_transforms.RandomHorizontalFlip())
        if ct_type != '3':
            ct.append(co_transforms.RandomVerticalFlip())
        com_transform = co_transforms.Compose(ct)
    else:
        ct = []
        if ct_type == '1' and not sparse:
            ct.append(co_transforms.FixSizeScale(trans_size))
        else:
            ct.append(co_transforms.CenterCrop(trans_size))
        com_transform = co_transforms.Compose(ct)
    return input_transform, target_transform, com_transform
