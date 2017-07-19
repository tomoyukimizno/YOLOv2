#!/usr/bin/env python
# coding=utf-8
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
        dst_layer.c.W = src_layer.c.W
        dst_layer.c.b = src_layer.c.b
        # copy bn
        dst_layer.n.N = src_layer.n.N
        dst_layer.n.avg_var = src_layer.n.avg_var
        dst_layer.n.avg_mean = src_layer.n.avg_mean
        dst_layer.n.gamma = src_layer.n.gamma
        dst_layer.n.eps = src_layer.n.eps
        # copy bias
        dst_layer.b.b = src_layer.b.b


# load model
print("loading original model...")
input_weight_file = "./backup/darknet19_448_final.model"
output_weight_file = "./backup/partial.model"

model = Darknet19()
serializers.load_npz(input_weight_file, model)  # load saved model

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
copy_layer(model, yolov2, partial_layer)

print("saving model to %s" % (output_weight_file))
serializers.save_npz("%s" % (output_weight_file), yolov2)
