#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import random

import chainer
from chainer import training, serializers
from chainer.training import extensions

from darknet19 import Darknet19, Darknet19Predictor


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


if __name__ == "__main__":
    # hyper parameters
    backup_path = "backup.yolo_data.yolo_lee"
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0005

    parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel', help='Initialize the model from given file')
    parser.add_argument(
        '--loaderjob', '-j', type=int, help='Number of parallel data loading processes')
    parser.add_argument(
        '--resume', '-r', default='', help='Initialize the trainer from given file')
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument('--insize', '-s', default='224', help='Model image size')
    parser.add_argument(
        '--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    args = parser.parse_args()

    # load model
    model = Darknet19Predictor(Darknet19())
    backup_file = os.path.join(backup_path, "backup.model")
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    model.predictor.train = True
    model.insize = int(args.insize)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    train = PreprocessedDataset(args.train, args.root, model.insize)
    val = PreprocessedDataset(args.val, args.root, model.insize, False)
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    optimizer = chainer.optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), backup_path)

    val_interval = 250, 'epoch'
    log_interval = 100, 'epoch'

    learning_rate = extensions.LinearShift("lr", (0.001, 0), (0, int(args.epoch)))
    learning_rate.trigger = 1, 'epoch'
    trainer.extend(learning_rate)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'), trigger=val_interval)

    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy',
            'validation/main/accuracy', 'lr'
        ]),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()
    model.to_cpu()
    print("saving model to %s/darknet19_final.model" % (backup_path))
    serializers.save_hdf5(os.path.join(backup_path, "darknet19_448_final.model.dhf5"), model)
