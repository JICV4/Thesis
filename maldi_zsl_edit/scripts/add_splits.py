import pandas as pd
import numpy as np
import h5torch
import sys
from importlib.resources import files

file = sys.argv[1]


f = h5torch.File(file, "a")
f.visititems(print)

for repeat in range(10):
    data_path = files("maldi_zsl.utils").joinpath("split_%s.txt" % repeat)
    indicator = np.loadtxt(data_path, dtype='str')
    f.register(indicator.astype(bytes), axis = 0, name = "split_%s" % repeat, mode = "N-D", dtype_save="bytes", dtype_load="str")

f.close()