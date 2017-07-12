#!/usr/bin/env python

import chainer
import chainer.links as L
import chainer.functions as F
from lib.utils import *
from lib.functions import *


class Darknet19(chainer.Chain):
    """
    Darknet19
    - It takes (224, 224, 3) or (448, 448, 4) sized image as input
    """
    insize = None

    def __init__(self):

        super(Darknet19, self).__init__(
            # common layers for both pretrained layers and yolov2 #####
            conv1=L.Convolution2D(
                3, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1=L.BatchNormalization(
                32, use_beta=False),
            bias1=L.Bias(shape=(32,)),
            conv2=L.Convolution2D(
                32, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2=L.BatchNormalization(
                64, use_beta=False),
            bias2=L.Bias(shape=(64,)),
            conv3=L.Convolution2D(
                64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3=L.BatchNormalization(
                128, use_beta=False),
            bias3=L.Bias(shape=(128,)),
            conv4=L.Convolution2D(
                128, 64, ksize=1, stride=1, pad=0, nobias=True),
            bn4=L.BatchNormalization(
                64, use_beta=False),
            bias4=L.Bias(shape=(64,)),
            conv5=L.Convolution2D(
                64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn5=L.BatchNormalization(
                128, use_beta=False),
            bias5=L.Bias(shape=(128,)),
            conv6=L.Convolution2D(
                128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn6=L.BatchNormalization(
                256, use_beta=False),
            bias6=L.Bias(shape=(256,)),
            conv7=L.Convolution2D(
                256, 128, ksize=1, stride=1, pad=0, nobias=True),
            bn7=L.BatchNormalization(
                128, use_beta=False),
            bias7=L.Bias(shape=(128,)),
            conv8=L.Convolution2D(
                128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn8=L.BatchNormalization(
                256, use_beta=False),
            bias8=L.Bias(shape=(256,)),
            conv9=L.Convolution2D(
                256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn9=L.BatchNormalization(
                512, use_beta=False),
            bias9=L.Bias(shape=(512,)),
            conv10=L.Convolution2D(
                512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn10=L.BatchNormalization(
                256, use_beta=False),
            bias10=L.Bias(shape=(256,)),
            conv11=L.Convolution2D(
                256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn11=L.BatchNormalization(
                512, use_beta=False),
            bias11=L.Bias(shape=(512,)),
            conv12=L.Convolution2D(
                512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn12=L.BatchNormalization(
                256, use_beta=False),
            bias12=L.Bias(shape=(256,)),
            conv13=L.Convolution2D(
                256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn13=L.BatchNormalization(
                512, use_beta=False),
            bias13=L.Bias(shape=(512,)),
            conv14=L.Convolution2D(
                512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn14=L.BatchNormalization(
                1024, use_beta=False),
            bias14=L.Bias(shape=(1024,)),
            conv15=L.Convolution2D(
                1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn15=L.BatchNormalization(
                512, use_beta=False),
            bias15=L.Bias(shape=(512,)),
            conv16=L.Convolution2D(
                512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn16=L.BatchNormalization(
                1024, use_beta=False),
            bias16=L.Bias(shape=(1024,)),
            conv17=L.Convolution2D(
                1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn17=L.BatchNormalization(
                512, use_beta=False),
            bias17=L.Bias(shape=(512,)),
            conv18=L.Convolution2D(
                512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn18=L.BatchNormalization(
                1024, use_beta=False),
            bias18=L.Bias(shape=(1024,)),

            # new layer
            conv19=L.Convolution2D(
                1024, 10, ksize=1, stride=1, pad=0),)
        self.train = True
        # self.finetune = False

    def __call__(self, x, t):

        # common layer
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), test=not self.train)), slope=0.1)

        # new layer
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape
        h = F.reshape(h, (x.data.shape[0], -1))
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)

        return loss

    def predictor(self, x):
        # common layer
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), test=not self.train)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), test=not self.train)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), test=not self.train)), slope=0.1)

        # new layer
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape
        h = F.reshape(h, (x.data.shape[0], -1))
        return F.softmax(h)
