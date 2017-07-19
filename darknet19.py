#!/usr/bin/env python
# coding=utf-8

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from lib.utils import *
from lib.functions import *


def darknetConv2D(in_channel, out_channel, ksize=3, pad=1):
    return chainer.Chain(
        c=L.Convolution2D(
            in_channel, out_channel, ksize=ksize, stride=1, pad=pad, nobias=True),
        n=L.BatchNormalization(
            out_channel, use_beta=False),
        b=L.Bias(shape=(out_channel, )), )


def CRP(c, h, stride=2, pooling=False):
    # convolution -> leakyReLU -> MaxPooling
    h = c.b(c.n(c.c(h), test=True))
    h = F.leaky_relu(h, slope=0.1)
    if pooling:
        h = F.max_pooling_2d(h, ksize=2, stride=stride, pad=0)
    return h


class Darknet19(chainer.Chain):
    """
    Darknet19
    - It takes (224, 224, 3) or (448, 448, 4) sized image as input
    """

    def __init__(self):
        super(Darknet19, self).__init__(
            # common layers for both pretrained layers and yolov2
            dark1=darknetConv2D(3, 32),
            dark2=darknetConv2D(None, 64),
            dark3=darknetConv2D(None, 128),
            dark4=darknetConv2D(
                None, 64, ksize=1),
            dark5=darknetConv2D(None, 128),
            dark6=darknetConv2D(None, 256),
            dark7=darknetConv2D(
                None, 128, ksize=1),
            dark8=darknetConv2D(None, 256),
            dark9=darknetConv2D(None, 512),
            dark10=darknetConv2D(
                None, 256, ksize=1),
            dark11=darknetConv2D(None, 512),
            dark12=darknetConv2D(
                None, 256, ksize=1),
            dark13=darknetConv2D(None, 512),
            dark14=darknetConv2D(None, 1024),
            dark15=darknetConv2D(
                None, 512, ksize=1),
            dark16=darknetConv2D(None, 1024),
            dark17=darknetConv2D(
                None, 512, ksize=1),
            dark18=darknetConv2D(None, 1024),

            # new layer
            conv19=L.Convolution2D(
                1024, 10, ksize=1, stride=1, pad=1), )
        self.train = True

    def __call__(self, x, t):
        # common layer
        h = CRP(self.dark1, x, pooling=True)
        h = CRP(self.dark2, h, pooling=True)
        h = CRP(self.dark3, h)
        h = CRP(self.dark4, h)
        h = CRP(self.dark5, h, pooling=True)
        h = CRP(self.dark6, h)
        h = CRP(self.dark7, h)
        h = CRP(self.dark8, h, pooling=True)
        h = CRP(self.dark9, h)
        h = CRP(self.dark10, h)
        h = CRP(self.dark11, h)
        h = CRP(self.dark12, h)
        h = CRP(self.dark13, h, pooling=True)
        h = CRP(self.dark14, h)
        h = CRP(self.dark15, h)
        h = CRP(self.dark16, h)
        h = CRP(self.dark17, h)
        h = CRP(self.dark18, h)

        # new layer
        h = self.conv19(h)
        h = F.reshape(
            F.average_pooling_2d(
                h, h.data.shape[-1], stride=1, pad=0), (x.data.shape[0], -1))

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predictor(self, x):
        # common layer
        h = CRP(self.dark1, x, pooling=True)
        h = CRP(self.dark2, h, pooling=True)
        h = CRP(self.dark3, h)
        h = CRP(self.dark4, h)
        h = CRP(self.dark5, h, pooling=True)
        h = CRP(self.dark6, h)
        h = CRP(self.dark7, h)
        h = CRP(self.dark8, h, pooling=True)
        h = CRP(self.dark9, h)
        h = CRP(self.dark10, h)
        h = CRP(self.dark11, h)
        h = CRP(self.dark12, h)
        h = CRP(self.dark13, h, pooling=True)
        h = CRP(self.dark14, h)
        h = CRP(self.dark15, h)
        h = CRP(self.dark16, h)
        h = CRP(self.dark17, h)
        h = CRP(self.dark18, h)

        # new layer
        h = self.conv19(h)
        h = F.reshape(
            F.average_pooling_2d(
                h, h.data.shape[-1], stride=1, pad=0), (x.data.shape[0], -1))

        return F.softmax(h)
