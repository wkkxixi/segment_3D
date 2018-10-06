DEBUG=True
def log(s):
    if DEBUG:
        print(s)


def display(string):
    print(string)
    logger.info(string)
'''
split dataset into train and validation randomly by ratio
'''
def init_data_split(root, split_ratio, compound_dataset = False):
    from glob import glob
    from random import shuffle
    from os.path import join as pjoin
    from ptsemseg.utils import dataset_meta
    ratio = split_ratio

    meta_path = pjoin(root, 'meta.txt')
    dataset_meta(root)
    img_paths = []
    with open(meta_path) as f:
        lines = f.read().splitlines()
        for item in lines:
            if item.__contains__('.tif'):
                #print(item[:-2])
                #item = item.replace('\n','')
                img_paths.append(item)

    # compound_dataset = False
    # if compound_dataset:
    #     for dir_name in os.listdir(root):
    #         if dir_name.__contains__('fly'):
    #             sub_dataset = pjoin(root, dir_name)
    #             sub_dataset_imgs = pjoin(sub_dataset, 'images')
    #             sub_dataset_img_paths = glob(sub_dataset_imgs + '/*.tif')
    #             ratio = split_ratio
    #             img_paths.extend([path.split('/')[-3] + '/' + path.split('/')[-1] for path in sub_dataset_img_paths])
    # else:
    #     img_paths = glob(pjoin(root, 'images') + '/*.tif')
    shuffle(img_paths)
    val_paths = img_paths[:int(ratio*(len(img_paths)))]
    log('length of val_paths is: {}'.format(len(val_paths)))
    log('length of img_paths is: {}'.format(len(img_paths)))
    shuffle(val_paths)
    return {'img_paths': img_paths, 'val_paths': val_paths}


def prep_class_val_weights(ratio):
    weight_foreback = torch.ones(2)
    weight_foreback[0] = 1 / (1 - ratio) # ?????????
    weight_foreback[1] = 1 / ratio
    if torch.cuda.is_available():
        weight_foreback = weight_foreback.cuda()
    display("Cross Entropy's Weight:{}".format(weight_foreback))
    return weight_foreback

def time_keeper():
    end = time.time()
    elapsed = end - start
    logger.info('The total time is: {}'.format(time_converter(elapsed)))

def time_converter(elapsed):
    hour = int(elapsed / 3600)
    left = elapsed % 3600
    minute = int(left / 60)
    seconds = left % 60
    return '{} h {} m {} s'.format(hour, minute, seconds)


import os
# import sys
import yaml
import torch
# import visdom
import argparse
# import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
from tensorboardX import SummaryWriter
import shutil


from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader
from ptsemseg.models import get_model
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import get_logger
# from torchvision.utils import make_grid
import random
import time


def train(cfg, writer, logger):
    # Setup dataset split before setting up the seed for random
    data_split_info = init_data_split(cfg['data']['path'], cfg['data'].get('split_ratio', 0), cfg['data'].get('compound', False))  # fly jenelia dataset

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Cross Entropy Weight
    if cfg['training']['loss']['name'] != 'regression_l1':
        weight = prep_class_val_weights(cfg['training']['cross_entropy_ratio'])
    else:
        weight = None
    log('Using loss : {}'.format(cfg['training']['loss']['name']))

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None) # if no augmentation => default None
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    patch_size = [para for axis, para in cfg['training']['patch_size'].items()]

    t_loader = data_loader(
        data_path,
        split=cfg['data']['train_split'],
        augmentations=data_aug,
        data_split_info=data_split_info,
        patch_size=patch_size,
        allow_empty_patch = cfg['training'].get('allow_empty_patch', True))

    v_loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        data_split_info=data_split_info,
        patch_size=patch_size)

    n_classes = t_loader.n_classes
    log('n_classes is: {}'.format(n_classes))
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=False)

    # valloader = data.DataLoader(v_loader,
    #                             batch_size=cfg['training']['batch_size'],
    #                             num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes) # a confusion matrix is created


    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))
    softmax_function = nn.Softmax(dim=1)

    model_count = 0
    start_iter = 0
    if cfg['training']['resume'] is not None:
        log('resume saved model')
        if os.path.isfile(cfg['training']['resume']):
            display(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            display(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            display("No checkpoint found at '{}'".format(cfg['training']['resume']))
            log('no saved model found')

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i_train_iter = start_iter

    display('Training from {}th iteration\n'.format(i_train_iter))
    while i_train_iter < cfg['training']['train_iters']:
        i_batch_idx = 0
        train_iter_start_time = time.time()

        # training
        for (images, labels) in trainloader:
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            # mean = images[0]


            optimizer.zero_grad()
            outputs = model(images)
            # log('TrainIter=> images.size():{} labels.size():{} | outputs.size():{}'.format(images.size(), labels.size(), outputs.size()))
            # log('image max: {} min: {} | label max: {} min: {} | output max: {} min: {}'.format(torch.max(images), torch.min(images), torch.max(labels), torch.min(labels), torch.max(outputs), torch.min(outputs)))
            # loss = loss_fn(input=outputs, target=labels, weight=weight, size_average=cfg['training']['loss']['size_average'])
            loss = nn.L1Loss()
            loss = loss(outputs, labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)
            print_per_batch_check = True if cfg['training']['print_interval_per_batch'] else i_batch_idx+1 == len(trainloader)
            if (i_train_iter + 1) % cfg['training']['print_interval'] == 0 and print_per_batch_check:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i_train_iter + 1,
                                           cfg['training']['train_iters'],
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                display(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i_train_iter + 1)
                time_meter.reset()
            i_batch_idx += 1
        time_for_one_iteration = time.time() - train_iter_start_time
        # time_converter(time_for_one_iteration)
        # display('EntireTime for {}th training iteration: {:.4f}   EntireTime/Image: {:.4f}'.format(i_train_iter + 1,
        #                           time_for_one_iteration, time_for_one_iteration / (len(trainloader)*cfg['training']['batch_size'])))
        display('EntireTime for {}th training iteration: {}  EntireTime/Image: {}'.format(i_train_iter+1, time_converter(time_for_one_iteration),
                                                                                          time_converter(time_for_one_iteration/(len(trainloader)*cfg['training']['batch_size']))))

        # validation
        validation_check = (i_train_iter + 1) % cfg['training']['val_interval'] == 0 or \
                           (i_train_iter + 1) == cfg['training']['train_iters']
        if not validation_check:
            print('no validation check')
        else:
            # model.eval()
            # with torch.no_grad():
            #     log('start tqdm...')
            #     # for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            #     i_val = 0
            #     # for i_val, (images_val, labels_val) in enumerate(valloader):
            #     for (images_val, labels_val) in valloader:
            #
            #         log('i_val: {} => inside for loop'.format(i_val))
            #         images_val = images_val.to(device)
            #         labels_val = labels_val.to(device)
            #
            #         outputs = model(images_val)
            #         log('ValIter=> images_val.size():{} labels_val.size():{} | outputs.size():{}'.format(
            #             images_val.size(),
            #             labels_val.size(),
            #             outputs.size()))
            #         val_loss = loss_fn(input=outputs, target=labels, weight=weight, size_average=cfg['training']['loss']['size_average'])
            #
            #         # pred = outputs.data.max(1)[1].cpu().numpy() # for classes 2
            #         pred = outputs.data.cpu().numpy()
            #         gt = labels_val.data.cpu().numpy()
            #         log('validation: gt shape: {}  pred shape: {}'.format(gt.shape, pred.shape))
            #         running_metrics_val.update(gt, pred)
            #         val_loss_meter.update(val_loss.item())
            #         i_val += 1
            #     log('outside for loop tqdm')
            # writer.add_scalar('loss/val_loss', val_loss_meter.avg, i_train_iter + 1)
            # logger.info("Iter %d Loss: %.4f" % (i_train_iter + 1, val_loss_meter.avg))
            # log('get score...')
            # '''
            # This CODE-BLOCK is used to calculate and update the evaluation matrcs
            # '''
            # score, class_iou = running_metrics_val.get_scores()
            # for k, v in score.items():
            #     print(k, v)
            #     logger.info('{}: {}'.format(k, v))
            #     writer.add_scalar('val_metrics/{}'.format(k), v, i_train_iter + 1)
            #
            # for k, v in class_iou.items():
            #     logger.info('{}: {}'.format(k, v))
            #     writer.add_scalar('val_metrics/cls_{}'.format(k), v, i_train_iter + 1)
            #
            # val_loss_meter.reset()
            # running_metrics_val.reset()

            '''
            This IF-CHECK is used to update the best model
            '''
            # if score["Mean IoU       : \t"] >= best_iou:
            #     best_iou = score["Mean IoU       : \t"]
            #     state = {
            #         "epoch": i_train_iter + 1,
            #         "model_state": model.state_dict(),
            #         "optimizer_state": optimizer.state_dict(),
            #         "scheduler_state": scheduler.state_dict(),
            #         "best_iou": best_iou,
            #     }
            #     save_path = os.path.join(writer.file_writer.get_logdir(),
            #                              "{}_{}_best_model.pkl".format(
            #                                  cfg['model']['arch'],
            #                                  cfg['data']['dataset']))
            #     torch.save(state, save_path)

            state = {
                "epoch": i_train_iter + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }

            save_path = os.path.join(os.getcwd(), writer.file_writer.get_logdir(),
                                     "{}_{}_model_{}.pkl".format(
                                         cfg['model']['arch'],
                                         cfg['data']['dataset'],
                                         model_count))
            print('save_path is: ' + save_path)
            with open('/home/heng/Research/isbi/log_res.txt', 'a') as f:
                id = cfg['id']
                f.write(str(id) + ':' + save_path + '\n')


            torch.save(state, save_path)
            model_count += 1

        i_train_iter += 1


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn3d_fly.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger)

    time_keeper()
