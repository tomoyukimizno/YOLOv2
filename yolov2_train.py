#!/usr/bin/env python
# coding=utf-8
import argparse
import glob
import numpy as np
import os
import random
import sys

import chainer
from chainer import serializers, optimizers, Variable, cuda
import cv2

from lib.image_generator import *
from lib.utils import *
from yolov2 import YOLOv2, YOLOv2Predictor

parser = argparse.ArgumentParser(description='')
# parser.add_argument('train', help='Path to training image-label list file')
# parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
# parser.add_argument('--epoch', '-E', type=int, default=10, help='Number of epochs to train')
parser.add_argument(
    '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU')
parser.add_argument('--initmodel', help='Initialize the model from given file')
# parser.add_argument(
# '--loaderjob', '-j', type=int, help='Number of parallel data loading processes')
parser.add_argument('--out', '-o', default='backup', help='Output directory')
# parser.add_argument('--resume', '-r', default='', help='Initialize the trainer from given file')
parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
# parser.add_argument(
# '--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
args = parser.parse_args()

# hyper parameters
train_sizes = [320, 352, 384, 416, 448]
backup_file = os.path.join(args.out, "backup.model")
batch_size = int(args.batchsize)
max_batches = 30000
learning_rate = 1e-5
learning_schedules = {"0": 1e-5, "500": 1e-4, "10000": 1e-5, "20000": 1e-6}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 10
n_boxes = 5

# load model
print("loading initial model...")
yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2)
serializers.load_npz(args.initmodel, model)

model.predictor.train = True
model.predictor.finetune = False
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
    model.to_gpu()
else:
    print("NEED GPU FOR TRAINING")
    sys.exit(0)

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
print("start training")
root = args.root
filename_list = list(map(lambda x: x.replace(root, ""), glob.glob(root + "\\*")))

for batch in range(max_batches):
    if str(batch) in learning_schedules:
        optimizer.lr = learning_schedules[str(batch)]
    if batch % 80 == 0:
        input_width = input_height = train_sizes[np.random.randint(len(train_sizes))]
    filename_shuffle_list = random.sample(filename_list, len(filename_list))
    # print(filename_shuffle_list)
    for i in range(len(filename_shuffle_list) // batch_size):
        # data load
        x = np.empty((batch_size, 3, input_height, input_height), dtype=np.float32)
        t = []
        for j, filename in enumerate(filename_shuffle_list[i:i + batch_size]):
            img = cv2.resize(cv2.imread(root + filename, 1), (input_width, input_height))
            img = np.asarray(img, dtype=np.float32) / 255
            img = img.transpose(2, 0, 1)
            x[j] = img
            with open(root.replace("type416", "bbox") + filename.split(".")[0], "r") as f:
                coodinates = list(map(lambda x: x.split(" "), f.read().strip().split("\n")))
                # print(coodinates)
            _t = [{
                "label": np.float64(coodinate[0]),
                "x": np.float64(coodinate[1]),
                "y": np.float64(coodinate[2]),
                "w": np.float64(coodinate[3]),
                "h": np.float64(coodinate[4]),
                "one_hot_label": np.zeros(
                    10, dtype=np.float64)
            } for coodinate in coodinates]
            for i in range(len(_t)):
                _t[i]["one_hot_label"][int(_t[i]["label"])] = 1
            t.append(_t)
        x = Variable(x)
        x.to_gpu()

        # forward
        model.zerograds()
        loss = model(x, t)
        print("batch: %d     input size: %dx%d     learning rate: %f    loss: %f" %
              (batch, input_height, input_width, optimizer.lr, loss.data))
        print("/////////////////////////////////////")

        # backward and optimize
        loss.backward()
        optimizer.update()

        # save model
        if (batch + 1) % 500 == 0:
            model_file = os.path.join(args.out, "%s.model" % (batch + 1))
            print("saving model to %s" % (model_file))
            serializers.save_npz(model_file, model)
            serializers.save_npz(backup_file, model)

print("saving model to %s/yolov2_final.model" % (args.out))
serializers.save_npz(os.path.join(args.out, "yolov2_final.model"), model)

model.to_cpu()
serializers.save_npz(os.path.join(args.out, "yolov2_final_cpu.model"), model)
