import csv, sys
import os,glob
import time
import argparse
import numpy as np
from numpy import *
import multiprocessing


run = 0

models = ["-w 12",
          "-w 12 4",
          "-w 12 8 4",
          "-w 16 12 8 4",
      ]


rseed = 1234
CV_split = range(5)

def my_job(arg):
    [i,j] = arg
    cmd = "python3 bnn_brGDGT.py %s -cv %s -r %s" % (arg[0], arg[1], rseed)
    if run:
        os.system(cmd)
    else:
        print(cmd)
        



list_args = []
for i in models:
    for j in CV_split:
        list_args.append([i,j])



# print(list_args)

if __name__ == '__main__': 
    pool = multiprocessing.Pool(len(list_args))
    pool.map(my_job, list_args)
    pool.close()
