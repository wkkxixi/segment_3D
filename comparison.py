import time
from ptsemseg.utils import *
import csv

folderpath = '/home/heng/Desktop/Research/isbi/fly-dataset/flyJanelia/'
content = 'File\tPrecision\tRecall\tF1'#content format of the compareswc
count = 0
with open(folderpath+'datainfo/datainfo.txt') as f:
    lines = f.readlines()#read every line
    for item in lines:
        if item.__contains__('.'):#escape the first line and recognize the path
            filename=item.split()[0]
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

with open(folderpath + '40compare_unet3dreg_trainV3.csv', "w") as csv_file:
    writer = csv.writer(csv_file)
    for line in lines:
        writer.writerow([line])
