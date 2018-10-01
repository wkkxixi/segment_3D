import os
from os.path import join as pjoin
# import torch
# # import visdom
# import argparse
#
# from ptsemseg.models import get_model
# from scipy.ndimage.interpolation import zoom
#
from ptsemseg.utils import *
from os.path import join as pjoin
import fnmatch
import yaml



def compare_with_gt(folderpath):
    import csv
    content = 'File\tPrecision\tRecall\tF1'  # content format of the compareswc
    count = 0
    with open(folderpath + 'datainfo/datainfo.txt') as f:
        lines = f.readlines()  # read every line
        for item in lines:
            if item.__contains__('.'):  # escape the first line and recognize the path
                filename = item.split()[0]
                print(str(count) + ': ' + filename + ' is on processing')
                count += 1
                gt = folderpath + 'labels_v1/' + filename
                pred = folderpath + 'pred_unet3dreg_regression/' + filename
                gt = loadtiff3d(gt)
                print('gt max: {}  gt min: {}'.format(gt.max(), gt.min))
                pred = loadtiff3d(pred)
                print('pred max: {}  pred min: {}'.format(pred.max(), pred.min()))
                gt = (gt > 60).astype('int')
                pred = (pred > 60).astype('int')
                print(gt.shape)
                print(pred.shape)
                TP = np.sum(pred[gt == 1])
                FP = np.sum(pred[gt == 0])
                FN = np.sum(gt[pred == 0])
                print('TP: {}; FP: {}; FN: {}'.format(TP, FP, FN))
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = (2 * (recall * precision)) / (recall + precision)
                content = content + '\n' + filename + '\t%.2f\t%.2f\t%.2f' % (precision, recall, f1)

    lines = content.split('\n')

    with open(pjoin(folderpath, '40compare_unet3dreg_trainV3.csv'), "w") as csv_file:
        writer = csv.writer(csv_file)
        for line in lines:
            writer.writerow([line])



config_folder_path = '/home/heng/Desktop/Research/0920/segment_3D/configs'
log_file = '/home/heng/Desktop/Research/isbi/log.txt'
for f in os.listdir(config_folder_path):
    if fnmatch.fnmatch(f,'unet3d_regression_*.yml'):
        config_file_path = config_folder_path + '/' + f
        with open(config_file_path) as fp:
            cfg = yaml.load(fp)
        train_cmd = 'python3 train.py --config ' + config_file_path
        print(train_cmd)
        os.system(train_cmd)
        folder_path = cfg['data']['path']
        runID = None
        id = f.split('.yml')[0].split('_')[-1]
        with open(log_file) as log:
            lines = log.readlines()
            for item in lines:
                if item.__contains__(':'):
                    if id == item.split(':')[0]:
                        model_path = item.split(':')[-1]
                        runID = model_path.split('/')[-2]

                        pred_folder = pjoin(folder_path, 'pred_' + runID)
                        with open(pjoin(folder_path, 'meta.txt')) as meta:
                            items = meta.readlines()
                            for item in items:
                                if item.__contains__('.tif'):
                                    img_path = item
                                    img_set_name = img_path.split('/')[-2]
                                    img_name = img_path.split('/')[-1]
                                    out_path = pjoin(pred_folder, img_set_name + '_' +img_name)

                                    test_cmd = 'python3 test.py  --img_path ' + img_path + ' --out_path ' + out_path + ' --model_path ' + model_path





        flypath = '/home/heng/Desktop/Research/isbi/fly-dataset/' + f.split('.yml')[0].split('_')[-1]




config_file_path = '/home/heng/Desktop/Research/0920/segment_3D/configs/unet3d_regression_2.yml'
train_cmd = 'python3 train.py --config ' + config_file_path
dataset_folder_path = '/home/heng/Desktop/Research/isbi/fly-dataset'
for filename in os.listdir(dataset_folder_path):
    if fnmatch.fnmatch(filename, '*fly*'):
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






