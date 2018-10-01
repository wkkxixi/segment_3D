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



def compare_with_gt(folderpath, runID):
    import csv
    content = 'File\tPrecision\tRecall\tF1'  # content format of the compareswc
    count = 0
    meta_file = pjoin(folder_path, 'meta.txt')
    with open(meta_file) as f:
        lines = f.readlines()  # read every line
        for item in lines:
            if item.__contains__('.'):  # escape the first line and recognize the path
                filename = item.split('/')[-1]
                gt = item.replace('images', 'ground_truth')
                pred = item.replace('images', 'pred_'+runID)
                gt = loadtiff3d(gt)
                # print('gt max: {}  gt min: {}'.format(gt.max(), gt.min))
                pred = loadtiff3d(pred)
                # print('pred max: {}  pred min: {}'.format(pred.max(), pred.min()))
                gt = (gt > 40).astype('int')
                pred = (pred > 40).astype('int')
                # print(gt.shape)
                # print(pred.shape)
                TP = np.sum(pred[gt == 1])
                FP = np.sum(pred[gt == 0])
                FN = np.sum(gt[pred == 0])
                # print('TP: {}; FP: {}; FN: {}'.format(TP, FP, FN))
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = (2 * (recall * precision)) / (recall + precision)
                content = content + '\n' + filename + '\t%.2f\t%.2f\t%.2f' % (precision, recall, f1)

    lines = content.split('\n')

    with open(pjoin(folderpath, runID+'_compare.csv'), "w") as csv_file:
        writer = csv.writer(csv_file)
        for line in lines:
            writer.writerow([line])



config_folder_path = '/home/heng/Desktop/Research/0920/segment_3D/configs'
log_file = '/home/heng/Desktop/Research/isbi/log.txt'
counter = 0
for f in os.listdir(config_folder_path):
    if fnmatch.fnmatch(f,'unet3d_regression_*.yml'):
        config_file_path = config_folder_path + '/' + f
        print('{}: 1. Loading configuration => {}'.format(counter, config_file_path))

        with open(config_file_path) as fp:
            cfg = yaml.load(fp)
        print('{}: 2. Training...'.format(counter))
        train_cmd = 'python3 train.py --config ' + config_file_path
        os.system(train_cmd)

        folder_path = cfg['data']['path']
        runID = None
        id = cfg['id']
        with open(log_file) as log:
            lines = log.read().splitlines()
            for item in lines:
                if item.__contains__(':'):
                    if id == item.split(':')[0]:

                        model_path = item.split(':')[-1]
                        model_number = model_path.split('_')[-1].split('.pkl')[0]

                        if model_number != '4':
                            print('{}: Check...model number: {} ignored'.format(counter, model_number))
                            continue
                        runID = model_path.split('/')[-2]
                        print('{}: 3. Testing...runID: {}'.format(counter, runID))
                        with open(pjoin(folder_path, 'meta.txt')) as meta:
                            items = meta.read().splitlines()
                            for item in items:
                                if item.__contains__('.tif'):
                                    img_path = item
                                    out_path = img_path.replace('images', 'pred_'+runID)
                                    test_cmd = 'python3 test.py  --img_path ' + img_path + ' --out_path ' + out_path + ' --model_path ' + model_path
                                    os.system(test_cmd)
                        print('{}: 4. Comparing...'.format(counter))
                        compare_with_gt(folder_path, runID)

    counter += 1

