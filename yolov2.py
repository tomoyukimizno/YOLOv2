import numpy as np
import cupy
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


def darknetConv2D(in_channel, out_channel, ksize=3, pad=1):
    return chainer.Chain(
        c=L.Convolution2D(
            in_channel, out_channel, ksize=ksize, stride=1, pad=pad, nobias=True),
        n=L.BatchNormalization(
            out_channel, use_beta=False),
        b=L.Bias(shape=(out_channel, )), )


def CRP(c, h, train, stride=2, pooling=False):
    # convolution -> leakyReLU -> MaxPooling
    h = c.b(c.n(c.c(h), test=not train))
    h = F.leaky_relu(h, slope=0.1)
    if pooling:
        h = F.max_pooling_2d(h, ksize=2, stride=stride, pad=0)
    return h


def multi_overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    return F.minimum(x1 + len1_half, x2 + len2_half) - F.maximum(x1 - len1_half, x2 - len2_half)


def multi_box_intersection(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    w = multi_overlap(a_x, a_w, b_x, b_w)
    h = multi_overlap(a_y, a_h, b_y, b_h)
    zeros = Variable(cupy.zeros(w.shape, dtype=cupy.float32))

    return F.maximum(w, zeros) * F.maximum(h, zeros)


def multi_box_iou(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    i = multi_box_intersection(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h)
    return i / (a_w * a_h + b_w * b_h - i)


class YOLOv2(chainer.Chain):
    """
    YOLOv2
    - It takes (416, 416, 3) sized image as input
    """

    def __init__(self, n_classes, n_boxes):
        super(YOLOv2, self).__init__(
            # common layers for both pretrained layers and yolov2 #
            dark1=darknetConv2D(3, 32),
            dark2=darknetConv2D(None, 64),
            dark3=darknetConv2D(None, 128),
            dark4=darknetConv2D(
                None, 64, ksize=1, pad=0),
            dark5=darknetConv2D(None, 128),
            dark6=darknetConv2D(None, 256),
            dark7=darknetConv2D(
                None, 128, ksize=1, pad=0),
            dark8=darknetConv2D(None, 256),
            dark9=darknetConv2D(None, 512),
            dark10=darknetConv2D(
                None, 256, ksize=1, pad=0),
            dark11=darknetConv2D(None, 512),
            dark12=darknetConv2D(
                None, 256, ksize=1, pad=0),
            dark13=darknetConv2D(None, 512),
            dark14=darknetConv2D(None, 1024),
            dark15=darknetConv2D(
                None, 512, ksize=1, pad=0),
            dark16=darknetConv2D(None, 1024),
            dark17=darknetConv2D(
                None, 512, ksize=1, pad=0),
            dark18=darknetConv2D(None, 1024),

            # new layer
            dark19=darknetConv2D(None, 1024),
            dark20=darknetConv2D(None, 1024),
            dark21=darknetConv2D(None, 1024),
            conv22=L.Convolution2D(
                None, n_boxes * (5 + n_classes), ksize=1, stride=1, pad=0, nobias=True),
            bias22=L.Bias(shape=(n_boxes * (5 + n_classes), )), )
        self.train = True
        self.n_boxes = n_boxes
        self.n_classes = n_classes
        self.anchors = cupy.array(
            [[1.3203125, 1.796527777777778], [1.4625000000000001, 2.004166666666667],
             [1.7671875, 2.473611111111111], [1.2864583333333333, 1.7333333333333334],
             [2.803125, 4.098611111111111]],
            dtype=cupy.float32)
        self.thresh = 0.6

    def __call__(self, x, t):
        # ネットワーク出力の計算
        # common layer
        h = CRP(self.dark1, x, train=self.train, pooling=True)
        h = CRP(self.dark2, h, train=self.train, pooling=True)
        h = CRP(self.dark3, h, train=self.train)
        h = CRP(self.dark4, h, train=self.train)
        h = CRP(self.dark5, h, train=self.train, pooling=True)
        h = CRP(self.dark6, h, train=self.train)
        h = CRP(self.dark7, h, train=self.train)
        h = CRP(self.dark8, h, train=self.train, pooling=True)
        h = CRP(self.dark9, h, train=self.train)
        h = CRP(self.dark10, h, train=self.train)
        h = CRP(self.dark11, h, train=self.train)
        h = CRP(self.dark12, h, train=self.train)
        h = CRP(self.dark13, h, train=self.train)
        high_resolution_feature = F.space2depth(h, 2)  # 高解像度特徴量をサイズ落として保存
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = CRP(self.dark14, h, train=self.train)
        h = CRP(self.dark15, h, train=self.train)
        h = CRP(self.dark16, h, train=self.train)
        h = CRP(self.dark17, h, train=self.train)
        h = CRP(self.dark18, h, train=self.train)
        # new layer
        h = CRP(self.dark19, h, train=self.train)
        h = CRP(self.dark20, h, train=self.train)
        h = F.concat((high_resolution_feature, h), axis=1)  # output concatnation
        h = CRP(self.dark21, h, train=self.train)
        h = self.bias22(self.conv22(h))

        # NW出力の整形
        n_batch, _, grid_h, grid_w = h.shape
        _, n_data, _ = t.shape  # n_batch, n_data, (label,x,y,w,h)
        x_pred, y_pred, w_pred, h_pred, conf_pred, prob_pred = F.split_axis(
            F.reshape(h, (n_batch, self.n_boxes, self.n_classes + 5, grid_h, grid_w)),
            [1, 2, 3, 4, 5],
            axis=2)

        x_pred = F.squeeze(F.sigmoid(x_pred))
        y_pred = F.squeeze(F.sigmoid(y_pred))
        w_pred = F.squeeze(w_pred)
        h_pred = F.squeeze(h_pred)
        conf_pred = F.squeeze(F.sigmoid(conf_pred))
        """
        pred, prob_pred = F.split_axis(
            F.reshape(h, (n_batch, self.n_boxes, self.n_classes + 5, grid_h, grid_w)), [5], axis=2)
        x_pred, y_pred, w_pred, h_pred, conf_pred = F.separate(F.sigmoid(pred), axis=2)
        """
        prob_pred = F.softmax(F.transpose(prob_pred, (0, 2, 1, 3, 4)))
        # 真の値の整形
        label, center_x, center_y, width, height = F.separate(t, axis=2)

        # objectの存在するanchor boxの探索
        anchor_index = self.get_anchor_index(width, height, grid_w, grid_h)
        pb_x, pb_y, pb_w, pb_h = self.get_pred_box(x_pred, y_pred, w_pred, h_pred,
                                                   (n_data, *x_pred.shape))
        tb_x, tb_y, tb_w, tb_h = self.get_true_box(center_x, center_y, width, height,
                                                   (n_data, *x_pred.shape))
        best_ious = F.max(multi_box_iou(pb_x, pb_y, pb_w, pb_h, tb_x, tb_y, tb_w, tb_h), axis=0)

        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする
        # truth の周りの grid は真の値を予測値と同じ(lossなしとみなす)
        # (効率的?に計算するため右辺ではゼロ行列を利用してステップ関数を作っている)
        conf = F.ceil(
            F.maximum(
                best_ious - self.thresh, F.tile(
                    cupy.array(
                        0, dtype=cupy.float32), x_pred.shape))) * conf_pred

        # learning lateの初期化
        learning_scale = 0.1

        # オブジェクトのない位置のloss計算
        loss_x = F.sum(F.square(x_pred - 0.5) * learning_scale) / 2
        loss_y = F.sum(F.square(y_pred - 0.5) * learning_scale) / 2
        loss_w = F.sum(F.square(w_pred) * learning_scale) / 2
        loss_h = F.sum(F.square(h_pred) * learning_scale) / 2
        loss_conf = F.sum(F.squared_difference(conf_pred, conf) * learning_scale) / 2
        # loss_prob = 0 # オブジェクトのない位置は学習しない

        # オブジェクトのある位置のloss計算
        batch = [index[0] for index in np.ndindex(n_batch, n_data)]
        anchor = [int(anchor_index[index].data) for index in np.ndindex(n_batch, n_data)]
        x_index = [int(center_x[index].data * grid_w) for index in np.ndindex(n_batch, n_data)]
        y_index = [int(center_y[index].data * grid_h) for index in np.ndindex(n_batch, n_data)]
        l = [int(label[index].data) for index in np.ndindex(n_batch, n_data)]
        x_pred_extract = F.reshape(
            cupy.array(x_pred[batch, anchor, y_index, x_index].data), center_x.shape)
        y_pred_extract = F.reshape(
            cupy.array(y_pred[batch, anchor, y_index, x_index].data), center_y.shape)
        w_pred_extract = F.reshape(
            cupy.array(w_pred[batch, anchor, y_index, x_index].data), width.shape)
        h_pred_extract = F.reshape(
            cupy.array(h_pred[batch, anchor, y_index, x_index].data), height.shape)
        conf_pred_extract = F.reshape(
            cupy.array(conf_pred[batch, anchor, y_index, x_index].data), center_x.shape)
        conf_extract = F.reshape(
            cupy.array(conf[batch, anchor, y_index, x_index].data), center_x.shape)
        # padding 箇所はlearning rateを0にして無害化
        padding_entries = Variable(
            cupy.reshape(
                cupy.ceil(cupy.tanh(-cupy.array(
                    l, dtype=cupy.float32))), label.shape))
        learning_scale = F.tile(
            cupy.array(
                0.1, dtype=cupy.float32), width.shape) * (1 - padding_entries)
        abs_anchors = (self.anchors / cupy.array([grid_w, grid_h])).astype(cupy.float32)
        w_anchor = F.reshape(abs_anchors[anchor][:, 0], width.shape)
        h_anchor = F.reshape(abs_anchors[anchor][:, 1], height.shape)

        prob_pred_extract = F.reshape(prob_pred[batch, :, anchor, y_index, x_index].data,
                                      (*center_x.shape, self.n_classes))
        one_hots = cupy.zeros(prob_pred_extract.shape, dtype=cupy.float32)
        one_hots[batch, [index[1] for index in np.ndindex(n_batch, n_data)], l] = 1
        # prob  = px + (1-p)y
        # p:(label が -1(paddingしたところ) が 1, 他は0) を broadcast
        # x: prob_pred_extract (paddingの箇所は loss が 0 になるように設定)
        # y: 1-of-K vector (真の重み)
        prob = F.linear_interpolate(
            F.broadcast_to(
                F.reshape(padding_entries, (*padding_entries.shape, 1)), prob_pred_extract.shape),
            prob_pred_extract, Variable(one_hots))

        # 余分に足した分を引く
        loss_x -= F.sum(F.square(x_pred_extract - 0.5) * learning_scale) / 2
        loss_y -= F.sum(F.square(y_pred_extract - 0.5) * learning_scale) / 2
        loss_w -= F.sum(F.square(w_pred_extract) * learning_scale) / 2
        loss_h -= F.sum(F.square(h_pred_extract) * learning_scale) / 2
        loss_conf -= F.sum(
            F.squared_difference(conf_pred_extract, conf_extract) * learning_scale) / 2
        # 真の位置のlearning rateを設定
        learning_scale = F.ceil(learning_scale)
        # オブジェクトのある位置のlossを足し上げる
        center_x_grid = center_x * grid_w
        center_y_grid = center_y * grid_h
        center_x_shift = F.floor(center_x_grid)
        center_y_shift = F.floor(center_y_grid)
        loss_x += F.sum(
            F.squared_difference(center_x_grid - center_x_shift, x_pred_extract) *
            learning_scale) / 2
        loss_y += F.sum(
            F.squared_difference(center_y_grid - center_y_shift, y_pred_extract) *
            learning_scale) / 2
        loss_w += F.sum(
            F.squared_difference(
                F.log(F.maximum(width, -width)) - F.log(w_anchor),
                w_pred_extract) * learning_scale) / 2
        loss_h += F.sum(
            F.squared_difference(
                F.log(F.maximum(height, -height)) - F.log(h_anchor),
                h_pred_extract) * learning_scale) / 2
        loss_conf += F.sum(
            F.squared_difference(
                multi_box_iou(center_x, center_y, width, height, (x_pred_extract + center_x_shift)
                              / grid_w, (y_pred_extract + center_y_shift) / grid_h,
                              F.exp(w_pred_extract) * w_anchor, F.exp(h_pred_extract) * h_anchor),
                conf_pred_extract) * learning_scale) / 2 * 10
        loss_prob = F.sum(F.squared_difference(prob, prob_pred_extract)) / 2
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_prob

        chainer.report({
            'loss': loss.data,
            'loss_x': loss_x.data,
            'loss_y': loss_y.data,
            'loss_w': loss_w.data,
            'loss_h': loss_h.data,
            'loss_conf': loss_conf.data,
            'loss_prob': loss_prob.data
        }, self)

        return loss

    def init_anchor(self, anchors):
        self.anchors = anchors

    def get_anchor_index(self, width, height, grid_w, grid_h):
        abs_anchors = (self.anchors / cupy.array([grid_w, grid_h])).astype(cupy.float32)
        ex_w = F.broadcast_to(width, (self.n_boxes, *width.shape))
        ex_h = F.broadcast_to(height, (self.n_boxes, *height.shape))
        ex_aw = F.broadcast_to(
            F.reshape(abs_anchors[:, 0], (self.n_boxes, 1, 1)), (self.n_boxes, *width.shape))
        ex_ah = F.broadcast_to(
            F.reshape(abs_anchors[:, 1], (self.n_boxes, 1, 1)), (self.n_boxes, *height.shape))
        intersection = F.minimum(ex_w, ex_aw) * F.minimum(ex_h, ex_ah)
        return F.argmax(intersection / (ex_w * ex_h + ex_aw * ex_ah - intersection), axis=0)

    def get_pred_box(self, x_pred, y_pred, w_pred, h_pred, shape):
        _, _, grid_h, grid_w = x_pred.shape
        x_shift = F.broadcast_to(cupy.arange(grid_w, dtype=cupy.float32), x_pred.shape)
        y_shift = F.broadcast_to(
            cupy.arange(
                grid_h, dtype=cupy.float32).reshape(grid_h, 1), y_pred.shape)
        w_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 0], (self.n_boxes, 1, 1)), w_pred.shape)
        h_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 1], (self.n_boxes, 1, 1)), h_pred.shape)
        box_x = F.broadcast_to((x_pred + x_shift) / grid_w, shape)
        box_y = F.broadcast_to((y_pred + y_shift) / grid_h, shape)
        box_w = F.broadcast_to(F.exp(w_pred) * w_anchor / grid_w, shape)
        box_h = F.broadcast_to(F.exp(h_pred) * h_anchor / grid_h, shape)
        return box_x, box_y, box_w, box_h

    @staticmethod
    def make_true_box(data, shape):
        n_batch, n_data = data.shape
        return F.broadcast_to(F.reshape(F.swapaxes(data, 0, 1), (n_data, n_batch, 1, 1, 1)), shape)

    def get_true_box(self, center_x, center_y, width, height, shape):
        return self.make_true_box(center_x, shape), self.make_true_box(
            center_y, shape), self.make_true_box(width, shape), self.make_true_box(height, shape)

    def predictor(self, x):
        # ネットワーク出力の計算
        # common layer
        h = CRP(self.dark1, x, train=self.train, pooling=True)
        h = CRP(self.dark2, h, train=self.train, pooling=True)
        h = CRP(self.dark3, h, train=self.train)
        h = CRP(self.dark4, h, train=self.train)
        h = CRP(self.dark5, h, train=self.train, pooling=True)
        h = CRP(self.dark6, h, train=self.train)
        h = CRP(self.dark7, h, train=self.train)
        h = CRP(self.dark8, h, train=self.train, pooling=True)
        h = CRP(self.dark9, h, train=self.train)
        h = CRP(self.dark10, h, train=self.train)
        h = CRP(self.dark11, h, train=self.train)
        h = CRP(self.dark12, h, train=self.train)
        h = CRP(self.dark13, h, train=self.train)
        high_resolution_feature = F.space2depth(h, 2)  # 高解像度特徴量をサイズ落として保存
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = CRP(self.dark14, h, train=self.train)
        h = CRP(self.dark15, h, train=self.train)
        h = CRP(self.dark16, h, train=self.train)
        h = CRP(self.dark17, h, train=self.train)
        h = CRP(self.dark18, h, train=self.train)
        # new layer
        h = CRP(self.dark19, h, train=self.train)
        h = CRP(self.dark20, h, train=self.train)
        h = F.concat((high_resolution_feature, h), axis=1)  # output concatnation
        h = CRP(self.dark21, h, train=self.train)
        h = self.bias22(self.conv22(h))

        n_batch, _, grid_h, grid_w = h.shape

        # NW出力の整形
        pred, prob_pred = F.split_axis(
            F.reshape(h, (n_batch, self.n_boxes, self.n_classes + 5, grid_h, grid_w)), [5], axis=2)
        x_pred, y_pred, w_pred, h_pred, conf_pred = F.separate(F.sigmoid(pred), axis=2)
        prob_pred = F.sigmoid(F.transpose(prob_pred, (0, 2, 1, 3, 4)))

        x_shift = F.broadcast_to(cupy.arange(grid_w, dtype=cupy.float32), x_pred.shape)
        y_shift = F.broadcast_to(
            cupy.arange(
                grid_h, dtype=cupy.float32).reshape(grid_h, 1), y_pred.shape)
        w_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 0], (self.n_boxes, 1, 1)), w_pred.shape)
        h_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 1], (self.n_boxes, 1, 1)), h_pred.shape)
        box_x = F.broadcast_to((x_pred + x_shift) / grid_w, x_pred.shape)
        box_y = F.broadcast_to((y_pred + y_shift) / grid_h, y_pred.shape)
        box_w = F.broadcast_to(F.exp(w_pred) * w_anchor / grid_w, w_pred.shape)
        box_h = F.broadcast_to(F.exp(h_pred) * h_anchor / grid_h, h_pred.shape)

        return box_x, box_y, box_w, box_h, conf_pred, prob_pred
