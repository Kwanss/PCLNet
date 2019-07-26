import numpy as np


def makeColorwheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), np.int32)  # R,G,B

    # RY
    col = 0
    colorwheel[:RY, 0] = 255
    colorwheel[:RY, 1] = (np.array(range(RY)) / float(RY) * 255.0).astype(np.int32)
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - (np.array(range(YG)) / float(YG) * 255.0).astype(np.int32)
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = (np.array(range(GC)) / float(GC) * 255.0).astype(np.int32)
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = (255 - (np.array(range(CB))) / float(CB) * 255.0).astype(np.int32)
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = (np.array(range(BM)) / float(BM) * 255.0).astype(np.int32)
    col += BM

    # MR
    colorwheel[col:col + MR, 2] = (255 - (np.array(range(MR))) / float(MR) * 255.0).astype(np.int32)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def computeColor(u, v):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    indNan = np.isnan(u) | np.isnan(v)
    u[indNan] = 0
    v[indNan] = 0

    colorwheel = makeColorwheel()
    ncols = np.shape(colorwheel)[0]

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 mapped to 0~(ncols-1) => col index

    k0 = fk.astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0

    f = fk - k0

    img = np.zeros((np.shape(u)[0], np.shape(u)[1], 3), np.uint8)
    for i in range(np.shape(colorwheel)[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])

        col[np.invert(idx)] = col[np.invert(idx)] * 0.75
        img[:, :, i] = ((255 * col) * np.invert(indNan)).astype(np.uint8)

    # print(np.shape(img))
    return img


def flowToColor(flo, norm_var=None):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    H, W, nBands = np.shape(flo)
    assert nBands == 2, "Currently only support 2 bands."
    u = flo[:, :, 0]
    v = flo[:, :, 1]

    maxu = -999
    maxv = -999
    minu = 999
    minv = 999
    maxrad = -1

    idxUnknown = ((np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH))
    if np.any(idxUnknown):
        u[idxUnknown] = 0
        v[idxUnknown] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    radius = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(maxrad, np.max(radius))

    if norm_var:
        maxrad = norm_var

    eps = 1e-16
    u = u / (maxrad + eps)
    v = v / (maxrad + eps)

    img = computeColor(u, v)
    return img


TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            data[np.isnan(data)] = 0
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
