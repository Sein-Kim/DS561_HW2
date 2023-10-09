import numpy as np
import time
import math
import torch
import copy
import torch.nn as nn

def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))


def getBatch(data, batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]

    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        yield batch