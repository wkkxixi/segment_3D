import os
from os.path import join as pjoin
# import torch
# # import visdom
# import argparse
#
# from ptsemseg.models import get_model
# from scipy.ndimage.interpolation import zoom
#
# from ptsemseg.utils import *

flypath = '/home/heng/Desktop/Research/isbi/flyJanelia/'
model_path = '/home/heng/Desktop/Research/0920/segment_3D/runs/fcn3d_fly/20064/unet3d_flyJanelia_best_model.pkl'
# im_path = pjoin(flypath, 'images', im_name + '.tif')
with open(pjoin(flypath, 'datainfo', 'datainfo.txt')) as f:
    lines = f.readlines()
    count = 0
    for item in lines:
        if item.__contains__('.'):
            filename = item.split()[0]
            img_path = flypath + 'images/' + filename
            # img_path = pjoin(flypath, 'images', filename)
            # output_path = pjoin(flypath, 'pred', filename)
            output_path = flypath + 'pred_unet3d/' + filename


            cmd ='python3 test.py  --img_path ' + img_path + ' --out_path ' + output_path + ' --model_path ' + model_path
            print(str(count) + ': ' + cmd)
            count += 1
            os.system(cmd)






