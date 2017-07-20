#!/usr/bin/env python
from __future__ import print_function
import argparse
import random
import os

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

import darknet19


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
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
        else:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image *= (1.0 / 255.0)
        return image, label


def main():
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
    parser.add_argument('--out', '-o', default='result', help='Output directory')
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument('--insize', '-s', default='224', help='Input image height, width')
    parser.add_argument(
        '--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Initialize the model to train
    model = darknet19.Darknet19()
    model.insize = int(args.insize)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Load the datasets
    train = PreprocessedDataset(args.train, args.root, model.insize)
    val = PreprocessedDataset(args.val, args.root, model.insize, False)

    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005), 'hook_dec')

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = 25, 'epoch'
    log_interval = 1, 'epoch'

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
        serializers.load_npz(args.resume, trainer)
    trainer.run()
    model.to_cpu()
    serializers.save_npz(
        os.path.join(args.out, "darknet19_%d_final.model.npz" % model.insize), model)


if __name__ == '__main__':
    main()
