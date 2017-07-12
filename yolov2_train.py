#!/usr/bin/env python
from __future__ import print_function
import argparse
import random
import os
import pickle

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

import yolov2


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.ImageDataset(path, root)
        # with open(path, "r") as f:
        with open("train.data\\label_bbox", "r") as f:
            self.path = f.read().strip().split("\n")

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        """
        label_path = os.path.join("tmp\\label", self.path[i].split("\\")[-1].split(".")[0])
        with open(label_path, "r") as f:
            labels_bboxes = f.read().strip().split("\n")
        for label_bbox in labels_bboxes:
            label, top, left, bottom, right = labels_bboxes[0].split(" ")
            print(top)
            print(left)
            print(bottom)
            print(right)
            yield image, label, top, left, bottom, right
        """
        # print(self.path[i])
        label, top, left, bottom, right = self.path[i].split(" ")
        return image, int(label), int(top), int(left), int(bottom), int(right)
        # return image, label, center_y, center_x, height, width のどちらか


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


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
    parser.add_argument(
        '--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Initialize the model to train

    model = yolov2.YOLOv2(n_classes=10, n_boxes=5)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets
    train = PreprocessedDataset(args.train, args.root)
    val = PreprocessedDataset(args.val, args.root)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    # optimizer.lr = learning_rate * (
    # 1 - batch / max_batches)**lr_decay_power  # Polynomial decay learning rate

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

    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu), trigger=val_interval)
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
    serializers.save_npz(os.path.join(args.out, "model.npz"), model)
    pickle.dump(model, open(os.path.join(args.out, "test.model.pickle"), 'wb'), -1)


if __name__ == '__main__':
    main()
