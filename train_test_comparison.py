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
    meta_file = pjoin(folderpath, 'meta_test.txt')
    with open(meta_file) as f:
        lines = f.read().splitlines()  # read every line
        for item in lines:
            if item.__contains__('.'):  # escape the first line and recognize the path
                filename = item.split('/')[-1]
                gt = item.replace('images', 'ground_truth_orginal')
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


config_folder_path = '/home/heng/Research/segment_3D/configs'
log_file = '/home/heng/Research/isbi/log_final_experiment_flyJanelia.txt'   # to change!!!!!

for f in os.listdir(config_folder_path):
    if fnmatch.fnmatch(f, 'final_1.yml'):
        continue
    # if fnmatch.fnmatch(f,'final_1.yml'): # to change!!!!!!
    elif fnmatch.fnmatch(f, 'final_6.yml'):
        config_file_path = config_folder_path + '/' + f
        with open(config_file_path) as fp:
            cfg = yaml.load(fp)
        id = cfg['id']
        print('{}: 1. Loading configuration => {}'.format(id, config_file_path))

        # if id == 4:
        #     pass
        # else:
        #     print('{}: 2. Training...'.format(id))
        #     train_cmd = 'python3 train.py --config ' + config_file_path
        #     os.system(train_cmd)

        # print('{}: 2. Training...'.format(id))
        # train_cmd = 'python3 train.py --config ' + config_file_path
        # os.system(train_cmd)

        folder_path = cfg['data']['path']
        runID = None

        print('configure file id: ' + str(id))
        with open(log_file) as log:
            lines = log.read().splitlines()
            for item in lines:
                if item.__contains__(':'):
                    if id == int(item.split(':')[0]):
                        print(item)
                        model_path = item.split(':')[-1]
                        model_number = model_path.split('_')[-1].split('.pkl')[0]

                        # if int(model_number) != 4: # to change!!!!!
                        #     print('{}: Check...model number: {} ignored'.format(id, model_number))
                        #     continue
                        runID = model_path.split('/')[-2]
                        print('{}: 3. Testing...runID: {}'.format(id, runID))
                        with open(pjoin(folder_path, 'meta_test.txt')) as meta:
                            items = meta.read().splitlines()
                            for item in items:
                                if item.__contains__('.tif'):
                                    img_path = item
                                    out_path = img_path.replace('test', 'pred_'+runID)
                                    task = cfg['training'].get('task', 'regression')
                                    test_cmd = 'python3 test.py  --img_path ' + img_path + ' --out_path ' + out_path + ' --model_path ' + model_path + ' --task ' + task
                                    os.system(test_cmd)
                        print('{}: 4. Comparing...'.format(id))
                        compare_with_gt(folder_path, runID)



