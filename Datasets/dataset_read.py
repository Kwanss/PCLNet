import glob
import os


def get_jhmdb_info(root=None, split=1, flow_type='com'):
    root = '/data/shuosen/JHMDB' if root is None else root
    split_path = os.path.join(root, 'list_files', 'split_%d.txt' % split)

    class_info = [line.strip().split() for line in open(os.path.join(root, 'list_files', 'class_name.txt'))]
    class_idx = {n: int(idx) for n, idx in class_info}
    idx_class = {int(idx): n for n, idx in class_info}

    split_info = [line.strip().split() for line in open(split_path, 'r')]
    split_info = [(int(class_ind), v_name.split('.avi')[0], int(set_ind)) for class_ind, v_name, set_ind in split_info]

    frames_info = [(os.path.join(root, 'Rename_Images', idx_class[c_ind], vname), c_ind, set_ind) for
                   c_ind, vname, set_ind in split_info]
    flow_ann_info = [(os.path.join(root, 'puppet_flow_ann', idx_class[c_ind], vname, 'puppet_flow.mat'), set_ind) for
                     c_ind, vname, set_ind in split_info]
    flow_com_info = [(os.path.join(root, 'puppet_flow_com', idx_class[c_ind], vname, 'puppet_flow.mat'), set_ind) for
                     c_ind, vname, set_ind in split_info]
    mask_info = [(os.path.join(root, 'puppet_mask', idx_class[c_ind], vname, 'puppet_mask.mat'), set_ind) for
                 c_ind, vname, set_ind in split_info]

    frames_train = [(p, len(glob.glob(p + '/*.png')), c_ind) for p, c_ind, set_ind in frames_info if set_ind == 1]
    frames_test = [(p, len(glob.glob(p + '/*.png')), c_ind) for p, c_ind, set_ind in frames_info if set_ind == 2]

    flow_ann_train = [p for p, set_ind in flow_ann_info if set_ind == 1]
    flow_ann_test = [p for p, set_ind in flow_ann_info if set_ind == 2]

    flow_com_train = [p for p, set_ind in flow_com_info if set_ind == 1]
    flow_com_test = [p for p, set_ind in flow_com_info if set_ind == 2]

    mask_train = [p for p, set_ind in mask_info if set_ind == 1]
    mask_test = [p for p, set_ind in mask_info if set_ind == 2]

    train_info = [frames_train, flow_ann_train, flow_com_train, mask_train]
    test_info = [frames_test, flow_ann_test, flow_com_test, mask_test]
    return train_info, test_info


def get_flyingchairs_info(root=None):
    ID_list = [line.strip() for line in open(os.path.join(root, 'list_files', 'ID_list.txt'), 'r')]
    path_list = [(os.path.join(root, 'data', ID + '_img1.ppm'), os.path.join(root, 'data', ID + '_img2.ppm'),
                  os.path.join(root, 'data', ID + '_flow.flo')) for ID in ID_list]

    train_info = path_list[:20000]
    test_info = path_list[20000:]
    return train_info, test_info


def get_mpi_info(root=None, pass_name='clean', flow_type='flow'):
    # pass_name: albedo/ clean/final/
    # mode: training / test
    print('++ MPI flow type: ', pass_name)
    root = '/home/share/MPI-Sintel-complete/' if root is None else root

    mode = 'training'
    rgb_root = os.path.join(root, mode, pass_name)
    flow_root = os.path.join(root, mode, flow_type)

    class_ind = {n_ind.strip().split()[0]: int(n_ind.strip().split()[1]) for n_ind in
                 open(os.path.join(root, 'class_id.txt'), 'r')}
    class_names = os.listdir(rgb_root)

    train_info = []
    for cls in class_names:
        rgb_info = []
        flow_info = []
        fn = len(os.listdir(os.path.join(rgb_root, cls)))
        cls_idx = class_ind[cls]
        for n in range(fn):
            rgb_info.append(os.path.join(rgb_root, cls, "frame_%04d.png" % (n + 1)))
        for n in range(fn - 1):
            flow_info.append(os.path.join(flow_root, cls, "frame_%04d.flo" % (n + 1)))

        train_info.append([rgb_info, flow_info, cls_idx])

    test_root = os.path.join(root, 'test')

    class_names = os.listdir(os.path.join(test_root, 'clean'))
    test_info = []
    for cls in class_names:
        clean_info = []
        final_info = []
        fn_c = len(os.listdir(os.path.join(test_root, 'clean', cls)))
        fn_f = len(os.listdir(os.path.join(test_root, 'final', cls)))
        cls_idx = class_ind[cls]
        for n in range(fn_c):
            clean_info.append(os.path.join(test_root, 'clean', cls, "frame_%04d.png" % (n + 1)))
        for n in range(fn_f):
            final_info.append(os.path.join(test_root, 'final', cls, "frame_%04d.png" % (n + 1)))
        test_info.append([clean_info, final_info, cls_idx])

    return train_info, test_info


def get_ucf101_info(root=None, split=1, num=-1):
    root = '/home/share2/ucf-data/' if root is None else root
    # root = '/ssd/ucf-data/'
    class_txt = os.path.join(root, 'UCF-101-list/classInd.txt')
    idx_vname = [line.strip().split() for line in open(class_txt, 'r')]
    idx_to_class = dict(idx_vname)
    vname_idx = [row[::-1] for row in idx_vname]
    class_to_idx = dict(vname_idx)

    train_txt = os.path.join(root, 'UCF-101-list/trainlist%02d.txt' % split)
    test_txt = os.path.join(root, 'UCF-101-list/testlist%02d.txt' % split)

    frames_num_txt = os.path.join(root, 'UCF-101-list/video_framenum.txt')
    framenum_v = [line.strip().split(',') for line in open(frames_num_txt, 'r')]
    framenum_v = [(a, int(b)) for a, b in framenum_v]
    v_to_framenum = dict(framenum_v)

    root = os.path.join(root, 'jpegs_256')

    temp_info = [line.strip().split() for line in open(train_txt, 'r')]
    temp_info = [(p.strip().split('/')[1].split('.')[0], int(idx)) for p, idx in temp_info]
    train_info = [(os.path.join(root, v), idx - 1, v_to_framenum[v]) for v, idx in
                  temp_info]  # ! Remember idx: 0, 1, ... 100

    temp_info = [line.strip().split('/') for line in open(test_txt, 'r')]
    temp_info = [(class_n, v.split('.')[0]) for class_n, v in temp_info]
    test_info = [(os.path.join(root, vname), int(class_to_idx[class_name]) - 1, v_to_framenum[vname]) for
                 class_name, vname in temp_info]

    train_info = [k for k in zip(*train_info)]
    test_info = [k for k in zip(*test_info)]
    if num != -1:
        train_info = [lst[:num] for lst in train_info]
        test_info = [lst[:num] for lst in test_info]
    return train_info, test_info


def get_kinectics_mini_info(train_txt=None, val_txt=None, val_prefix=None, split=1, num=-1):
    train_txt = '/home/share/Kinetics_mini_200/kinectics_mini_200_train2.txt' if train_txt is None else train_txt
    val_txt = '/home/share/Kinetics_mini_200/kinectics_mini_200_val.txt' if val_txt is None else val_txt
    val_prefix = '/data1/Kinetics/' if val_prefix is None else val_prefix

    train_info = [l.strip().split() for l in open(train_txt)]
    val_info = [l.strip().split() for l in open(val_txt)]
    val_info = [[val_prefix + v[0]] + v[1:] for v in val_info]  # (path, begin_index, end_index, class_index)

    train_info = [[v[0], int(v[1]), int(v[2]), int(v[3])] for v in train_info]
    val_info = [[v[0], int(v[1]), int(v[2]), int(v[3])] for v in val_info]

    if num != -1:
        train_info = [lst[:num] for lst in train_info]
        test_info = [lst[:num] for lst in test_info]
    # (path, begin_index, end_index, class_index)
    return train_info, val_info


def get_hmdb_info(root=None, split=1, num=-1):
    root = '/home/share/yijun/HMDB-51' if root is None else root
    class_txt = os.path.join(root, 'class_id.txt')
    idx_vname = [line.strip().split()[::-1] for line in open(class_txt, 'r')]
    idx_to_class = dict(idx_vname)
    vname_idx = [row[::-1] for row in idx_vname]
    class_to_idx = dict(vname_idx)

    train_txt = os.path.join(root, 'split_lists/split%d_train.txt' % split)
    test_txt = os.path.join(root, 'split_lists/split%d_test.txt' % split)

    root = os.path.join(root, 'jpegs_256')

    temp_info = [line.strip().split() for line in open(train_txt, 'r')]
    train_info = [(os.path.join(root, v), int(idx), int(frame_num)) for v, idx, frame_num in
                  temp_info]  # ! Remember idx: 0, 1, ... 100

    temp_info = [line.strip().split() for line in open(test_txt, 'r')]
    test_info = [(os.path.join(root, v), int(idx), int(frame_num)) for v, idx, frame_num in
                 temp_info]  # ! Remember idx: 0, 1, ... 100

    train_info = [k for k in zip(*train_info)]
    test_info = [k for k in zip(*test_info)]
    if num != -1:
        train_info = [lst[:num] for lst in train_info]
        test_info = [lst[:num] for lst in test_info]
    return train_info, test_info


def dataset_info(dataset='ucf101', **kwargs):
    root = kwargs['root']
    if dataset == 'ucf101':
        return get_ucf101_info(root, **kwargs)
    if dataset == 'hmdb':
        return get_hmdb_info(root, **kwargs)
    if dataset == 'mpi':
        return get_mpi_info(root, **kwargs)
    if dataset == 'jhmdb':
        return get_jhmdb_info(root, **kwargs)
    print("Dataset not supported yet!")
    exit()


if __name__ == "__main__":
    train_data, test_data = get_jhmdb_info()
    print('done')
    for i, a in enumerate(train_data):
        for b in a:
            if i == 0:
                b = b[0]
            assert os.path.exists(b)
    for i, a in enumerate(test_data):
        for b in a:
            if i == 0:
                b = b[0]
            assert os.path.exists(b)
