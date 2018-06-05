import sys
import numpy as np
from os.path import join, exists
import os


def rand_index(folder_name, filename):
    with open(join(folder_name, filename), 'r') as f_:
        nline = len(f_.readlines())
    index = np.arange(nline)
    np.random.shuffle(index)
    np.save(join(folder_name, 'index'), index)


def split(folder_name, file_name, nsplit, split_id):
    index = np.load(join(folder_name, 'index.npy'))
    nline = len(index)
    chunk_size = np.int(np.ceil(nline * 1. / nsplit))
    with open(join(folder_name, file_name), 'r') as f_in:
        lines = f_in.readlines()
    start = chunk_size * split_id
    end = min(chunk_size * (split_id + 1), nline)
    folder_out = join(folder_name, str(split_id + 1))
    if not exists(folder_out):
        os.makedirs(folder_out)
    with open(join(folder_out, file_name), 'w') as f_out:
        for j in range(start, end):
            cur_line = lines[index[j]]
            f_out.write(cur_line)


def split_train_test(folder_name, file_name, train_ratio, split_id):
    index = np.load(join(folder_name, 'index.npy'))
    nline = len(index)
    ntrain = np.int(np.ceil(nline * train_ratio))
    with open(join(folder_name, file_name), 'r') as f_in:
        lines = f_in.readlines()
    folder_out = join(folder_name, str(split_id + 1))
    if not exists(folder_out):
        os.makedirs(folder_out)
    with open(join(folder_out, 'train.line'), 'w') as f_out:
        for i in range(ntrain):
            f_out.write(lines[index[i]])
    with open(join(folder_out, 'test.line'), 'w') as f_out:
        for i in range(ntrain, nline):
            f_out.write(lines[index[i]])


arg_folder = sys.argv[1]
arg_file = sys.argv[2]
task = int(sys.argv[3])
if task == 0:
    rand_index(arg_folder, arg_file)
elif task == 1:
    arg_split = int(sys.argv[4])
    arg_split_id = int(sys.argv[5])
    split(arg_folder, arg_file, arg_split, arg_split_id)
elif task == 2:
    arg_ratio = float(sys.argv[4])
    arg_split_id = int(sys.argv[5])
    split_train_test(arg_folder, arg_file, arg_ratio, arg_split_id)
