import json
import os.path as osp
from helper import run

data_dir = '../data/'
flist = list(json.load(open('../filelist.json')).keys())
for dataset in flist:
    
    run(osp.join(data_dir, dataset)+'.npz', "50 25 25")
    break