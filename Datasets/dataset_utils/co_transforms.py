from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input, target):
        return self.lambd(input, target)


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and should be the same size
    """

    def __init__(self, size):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets):
        """
        inputs: list of image
        targets: list of flow
        """
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if th > 0 and tw > 0:
            x = int(round((w - tw) / 2.))
            y = int(round((h - th) / 2.))
        else:
            if th == 0:
                y, th = 0, h
            elif th == -1:
                th = (h // 64) * 64
                y = (h - th) // 2.0

            if tw == 0:
                x, tw = 0, w
            elif tw == -1:
                tw = (w // 64) * 64
                x = (w - tw) // 2.0

        inputs = [A[y: y + th, x: x + tw] for A in inputs]
        if targets is not None:
            targets = [T[y: y + th, x: x + tw] for T in targets]
        return inputs, targets


class RandomZoomIn(object):
    """
    inputs: list of image
    targets: list of flow
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, ratios, order=2):
        # ZoomIn in range( 1, 1 + ratios[i]]
        if isinstance(ratios, numbers.Number):
            self.ratios = (float(ratios), float(ratios))
        else:
            self.ratios = ratios
        assert (0 < self.ratios[0] <= 1) and (0 < self.ratios[1] <= 1)
        self.order = order

    def __call__(self, inputs, targets):
        add_ratio_H, add_ratio_W = self.ratios
        ratio_H, ratio_W = 1 + add_ratio_H, 1 + add_ratio_W

        if ratio_H <= 1.0 and ratio_W <= 1.0:
            return inputs, targets
        inputs = [ndimage.interpolation.zoom(A, (ratio_H, ratio_W, 1), order=self.order) for A in inputs]
        if targets is not None:
            targets = [ndimage.interpolation.zoom(T, (ratio_H, ratio_W, 1), order=self.order) for T in targets]
            targets = [(np.dstack((T[:, :, 0] * ratio_W, T[:, :, 1] * ratio_H))) for T in targets]
        return inputs, targets


class SmallEdgeScale(object):
    """
    inputs: list of image
    targets: list of flow
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, targets):
        h, w, _ = inputs[0].shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs, targets
        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = [ndimage.interpolation.zoom(A, [ratio, ratio, 1], order=self.order) for A in inputs]
        if targets is not None:
            targets = [ndimage.interpolation.zoom(T, [ratio, ratio, 1], order=self.order) for T in targets]
            targets = [T * ratio for T in targets]
        return inputs, targets


class FixSizeScale(object):
    """
    inputs: list of image
    targets: list of flow
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, targets):
        h, w, _ = inputs[0].shape
        ratio_h = self.size[0] / h
        ratio_w = self.size[1] / w

        inputs = [ndimage.interpolation.zoom(A, [ratio_h, ratio_w, 1], order=self.order) for A in inputs]
        if targets is not None:
            targets = [ndimage.interpolation.zoom(T, [ratio_h, ratio_w, 1], order=self.order) for T in targets]
            targets = [np.dstack((T[:, :, 0] * ratio_w, T[:, :, 1] * ratio_h)) for T in targets]
        return inputs, targets


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, targets
        if th > 0 and tw > 0:
            x = random.randint(0, w - tw)
            y = random.randint(0, h - th)
        else:
            if th == 0:  # return origan size
                y, th = 0, h
            elif th == -1:  # return 64*
                y, th = 0, (h // 64) * 64

            if tw == 0:
                x, tw = 0, w
            elif tw == -1:
                x, tw = 0, (w // 64) * 64

        inputs = [A[y:y + th, x: x + tw] for A in inputs]
        if targets is not None:
            targets = [T[y:y + th, x: x + tw] for T in targets]
        return inputs, targets


class RandomCropResize(object):
    """
    Corp the image/ flow with random pre_ZoomIn or pre_ZoomOut, pre_crop with size 
    (range[(1-diff_ratio[0])size(0) , (1+diff_ratio[0])size(0)] , range[(1-diff_ratio[1])size(1) , (1+diff_ratio[1])size(1)]  and then resize to target_size
    Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, diff_ratio, order=2):
        self.order = order
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if isinstance(diff_ratio, numbers.Number):
            self.diff_ratio = (int(diff_ratio), int(diff_ratio))
        else:
            self.diff_ratio = diff_ratio
        assert (-1 < self.diff_ratio[0] < 1) and (-1 < self.diff_ratio[1] < 1)

    def __call__(self, inputs, targets):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, targets
        if th > 0 and tw > 0:
            ratio_H, ratio_W = self.diff_ratio
            resized_th = int(th + th * random.uniform(-ratio_H, ratio_H))
            resized_tw = int(tw + tw * random.uniform(-ratio_W, ratio_W))

            if resized_th < h:
                y = random.randint(0, h - resized_th)
            else:
                y, resized_th = 0, h
            if resized_tw < w:
                x = random.randint(0, w - resized_tw)
            else:
                x, resized_tw = 0, w

            # crop to [resized_th, resized_tw]
            inputs = [A[y:y + resized_th, x: x + resized_tw] for A in inputs]
            if targets is not None:
                targets = [T[y:y + resized_th, x: x + resized_tw] for T in targets]

            # resize to th, tw
            ratio_H, ratio_W = th / resized_th, tw / resized_tw
            inputs = [ndimage.interpolation.zoom(A, (ratio_H, ratio_W, 1), order=self.order) for A in inputs]

            if targets is not None:
                targets = [ndimage.interpolation.zoom(T, (ratio_H, ratio_W, 1), order=self.order) for T in targets]
                targets = [(np.dstack((T[:, :, 0] * ratio_W, T[:, :, 1] * ratio_H))) for T in targets]
        else:
            if th == 0:  # return origan size
                y, th = 0, h
            elif th == -1:  # return 64*
                y, th = 0, (h // 64) * 64

            if tw == 0:
                x, tw = 0, w
            elif tw == -1:
                x, tw = 0, (w // 64) * 64
            inputs = [A[y:y + th, x: x + tw] for A in inputs]
            if targets is not None:
                targets = [T[y:y + th, x: x + tw] for T in targets]
        return inputs, targets


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, targets):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(A)) for A in inputs]
            if targets is not None:
                targets = [np.copy(np.fliplr(T)) for T in targets]
                for T in targets:
                    T[:, :, 0] *= -1
        return inputs, targets


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, targets):
        if random.random() < 0.5:
            inputs = [np.copy(np.flipud(A)) for A in inputs]
            if targets is not None:
                targets = [np.copy(np.flipud(T)) for T in targets]
                for T in targets:
                    T[:, :, 1] *= -1
        return inputs, targets


class RandomRotate(object):
    # TODO
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs, target):
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180

        h, w, _ = target.shape

        def rotate_flow(i, j, k):
            return -k * (j - w / 2) * (diff * np.pi / 180) + (1 - k) * (i - h / 2) * (diff * np.pi / 180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        target_ = np.copy(target)
        target[:, :, 0] = np.cos(angle1_rad) * target_[:, :, 0] + np.sin(angle1_rad) * target_[:, :, 1]
        target[:, :, 1] = -np.sin(angle1_rad) * target_[:, :, 0] + np.cos(angle1_rad) * target_[:, :, 1]
        return inputs, target


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs, targets):
        h, w, _ = inputs[0].shape
        th, tw = self.translation

        out_inputs = []
        out_targets = []
        for i in range(0, len(inputs) - 1, 2):
            tw_i = random.randint(-tw, tw)
            th_i = random.randint(-th, th)
            if tw_i == 0 and th_i == 0:
                out_inputs.append(inputs[i])
                out_targets.append(targets[i])
                continue
            out_inputs.append(inputs[i])
            out_targets.append(targets[i])
            # TODO
        """
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th
        """
        return inputs, targets


class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, inputs, target):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        inputs[0] *= (1 + random_std)
        inputs[0] += random_mean

        inputs[1] *= (1 + random_std)
        inputs[1] += random_mean

        inputs[0] = inputs[0][:, :, random_order]
        inputs[1] = inputs[1][:, :, random_order]

        return inputs, target
