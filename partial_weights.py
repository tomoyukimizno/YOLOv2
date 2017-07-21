#!/usr/bin/env python
# coding=utf-8
import argparse
from chainer import serializers
from darknet19 import Darknet19
from yolov2 import YOLOv2

n_classes = 10
n_boxes = 5
partial_layer = 18


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


# load model
print("loading original model...")
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--infile', '-i', default='backup/darknet19_448_final.model', help='Input weight file')
parser.add_argument('--out', '-o', default='backup/partial.model', help='Output weight file')
args = parser.parse_args()

model = Darknet19()
serializers.load_npz(args.infile, model)  # load saved model

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
copy_layer(model, yolov2, partial_layer)

print("saving model to %s" % (args.out))
serializers.save_npz(args.out, yolov2)
