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

flypath = '/home/heng/Desktop/Research/isbi/fly-dataset/utokyofly/'
model_path = '/home/heng/Desktop/Research/0920/segment_3D/runs/unet3d_regression_fly/68912/unet3dreg_flyDataset_model_0.pkl'
pred_folder = flypath + 'pred_68912/'
if not os.path.isdir(os.path.join(os.getcwd(), pred_folder)):
    os.mkdir(pred_folder)
else:
    print(pred_folder + ' already exists')
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
            output_path = pred_folder + filename # radius 1


            cmd ='python3 test.py  --img_path ' + img_path + ' --out_path ' + output_path + ' --model_path ' + model_path
            print(str(count) + ': ' + cmd)
            count += 1
            os.system(cmd)






