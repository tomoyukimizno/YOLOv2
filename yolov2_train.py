#!/usr/bin/env python
# coding=utf-8
import argparse
import numpy as np
import os
import random

import chainer
# from chainer import functions as F
from chainer import serializers
from chainer import training
from chainer.dataset.convert import concat_examples
from chainer.training import extensions

from darknet19 import Darknet19
from yolov2 import YOLOv2


def my_converter(batch, device=None):
    return concat_examples(batch, device=device, padding=-1)


class YoloDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, rand=True):
        self.imgs = chainer.datasets.ImageDataset(path, root)
        with open(path) as f:
            self.path_bboxes = list(
                map(lambda x: os.path.join(root, "bbox", x.split(".")[0]),
                    f.read().strip().split("\n")))
            # print(self.path_bboxes)
        # self.rand = rand
        # self.train_sizes = [320, 352, 384, 416, 448]

    def __len__(self):
        return len(self.imgs)

    def get_example(self, i):
        image = self.imgs[i]
        # ここに書くのはおかしい(Epoch単位で固定が必要)
        # 多分 updater/trainer に書くのが良い
        """
        # print(image.shape)
        c, h, w = image.shape
        if self.rand:
            randsize = random.sample(self.train_sizes, 1)[0]
            # print(randsize)
            image = F.expand_dims(image, axis=0)
            # print(image.shape)
            image = F.resize_images(image, (randsize, randsize))
            # print(image.shape)
            image = image[0]
        # """
        image *= (1.0 / 255.0)
        data = np.loadtxt(self.path_bboxes[i], delimiter=" ", dtype=np.float32)
        return image, data  # data を同じ行数にする必要あり、今は適当な値
        # return image, chainer.Variable(data)  # data を同じ行数にする必要あり、今は適当な値
        # return image, int(label), int(center_x), int(center_y), int(width), int(height)


def copy_layer(src, dst, max_num):
    for i in range(1, max_num + 1):
        src_layer = eval("src.dark%d" % i)
        dst_layer = eval("dst.dark%d" % i)
        # copy conv
        dst_layer.c = src_layer.c
        # copy bn
        dst_layer.n = src_layer.n
        # copy bias
        dst_layer.b = src_layer.b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel', help='Initialize the model from given file')
    parser.add_argument(
        '--loaderjob', '-j', type=int, help='Number of parallel data loading processes')
    parser.add_argument('--out', '-o', default='backup', help='Output directory')
    parser.add_argument(
        '--resume', '-r', default='', help='Initialize the trainer from given file')
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument(
        '--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    args = parser.parse_args()

    # hyper parameters
    backup_file = os.path.join(args.out, "backup.model")
    lr_decay_power = 4  # learning_rate の減衰のさせ方がわかっていない
    n_classes = 10
    n_boxes = 5

    # load model
    print("loading initial model...")
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        # cupy.cuda.Device(args.gpu).use()

    model = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
    if args.initmodel:
        print('Load model from', args.initmodel)
        darknet = Darknet19()
        serializers.load_npz(args.initmodel, darknet)  # load saved model
        copy_layer(darknet, model, 18)

    if args.gpu >= 0:
        model.to_gpu()

    train = YoloDataset(args.train, args.root)
    val = YoloDataset(args.val, args.root)
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.batchsize, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=1e-5, momentum=0.9)
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005), 'hook_dec')

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=my_converter)
    # updater = yolov2_updater.YOLOUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = 500, 'epoch'
    log_interval = 100, 'epoch'

    learning_rate = extensions.LinearShift("lr", (1e-5, 1e-4), (500 - 2, 500 - 1))
    learning_rate = extensions.LinearShift("lr", (1e-4, 1e-5), (10000 - 2, 10000 - 1))
    learning_rate = extensions.LinearShift("lr", (1e-5, 1e-6), (20000 - 2, 20000 - 1))
    learning_rate.trigger = 1, 'epoch'
    trainer.extend(learning_rate)

    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'), trigger=val_interval)

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'main/loss_x', 'main/loss_y', 'main/loss_w',
            'main/loss_h', 'main/loss_conf', 'lr'
        ]),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_x', 'main/loss_y', 'main/loss_w', 'main/loss_h', 'main/loss_conf'],
                'epoch',
                file_name='loss.png'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    print("saving model to %s/yolov2_final.model" % (args.out))
    serializers.save_npz(os.path.join(args.out, "yolov2_final.model"), model)

    model.to_cpu()
    serializers.save_npz(os.path.join(args.out, "yolov2_final_cpu.model"), model)
