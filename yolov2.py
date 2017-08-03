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
            [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125],
             [1.9375, 3.25]],
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

        n_batch, _, grid_h, grid_w = h.shape
        _, n_data, _ = t.shape  # n_batch, n_data, (label,x,y,w,h)

        # NW出力の整形
        pred, prob_pred = F.split_axis(
            F.reshape(h, (n_batch, self.n_boxes, self.n_classes + 5, grid_h, grid_w)), [5], axis=2)
        x_pred, y_pred, w_pred, h_pred, conf_pred = F.separate(F.sigmoid(pred), axis=2)
        prob_pred = F.sigmoid(F.transpose(prob_pred, (0, 2, 1, 3, 4)))
        # learning lateの初期化
        box_learning_scale = F.tile(cupy.array(0.1, dtype=cupy.float32), conf_pred.shape)
        conf_learning_scale = F.tile(cupy.array(0.1, dtype=cupy.float32), conf_pred.shape)

        # 真の値の整形
        label, center_x, center_y, width, height = F.separate(t, axis=2)

        # objectの存在するanchor boxの探索
        abs_anchors = (self.anchors / cupy.array([grid_w, grid_h])).astype(cupy.float32)
        ex_w = F.broadcast_to(width, (self.n_boxes, *width.shape))
        ex_h = F.broadcast_to(height, (self.n_boxes, *height.shape))
        ex_aw = F.broadcast_to(
            F.reshape(abs_anchors[:, 0], (self.n_boxes, 1, 1)), (self.n_boxes, *width.shape))
        ex_ah = F.broadcast_to(
            F.reshape(abs_anchors[:, 1], (self.n_boxes, 1, 1)), (self.n_boxes, *height.shape))
        intersection = F.minimum(ex_w, ex_aw) * F.minimum(ex_h, ex_ah)
        anchor_index = F.argmax(
            intersection / (ex_w * ex_h + ex_aw * ex_ah - intersection), axis=0)

        # objectの予測位置を算出
        x_shift = F.broadcast_to(cupy.arange(grid_w, dtype=cupy.float32), x_pred.shape)
        y_shift = F.broadcast_to(
            cupy.arange(
                grid_h, dtype=cupy.float32).reshape(grid_h, 1), y_pred.shape)
        w_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 0], (self.n_boxes, 1, 1)), w_pred.shape)
        h_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 1], (self.n_boxes, 1, 1)), h_pred.shape)
        box_x = F.broadcast_to((x_pred + x_shift) / grid_w, (n_data, *x_pred.shape))
        box_y = F.broadcast_to((y_pred + y_shift) / grid_h, (n_data, *y_pred.shape))
        box_w = F.broadcast_to(F.exp(w_pred) * w_anchor / grid_w, (n_data, *w_pred.shape))
        box_h = F.broadcast_to(F.exp(h_pred) * h_anchor / grid_h, (n_data, *h_pred.shape))

        # 真の位置と比較(IOUにて)
        # バッチサイズ分まとめて比較
        tbox_x = F.broadcast_to(
            F.reshape(F.swapaxes(center_x, 0, 1), (n_data, n_batch, 1, 1, 1)),
            (n_data, *x_pred.shape))
        tbox_y = F.broadcast_to(
            F.reshape(F.swapaxes(center_y, 0, 1), (n_data, n_batch, 1, 1, 1)),
            (n_data, *x_pred.shape))
        tbox_w = F.broadcast_to(
            F.reshape(F.swapaxes(width, 0, 1), (n_data, n_batch, 1, 1, 1)),
            (n_data, *x_pred.shape))
        tbox_h = F.broadcast_to(
            F.reshape(F.swapaxes(height, 0, 1), (n_data, n_batch, 1, 1, 1)),
            (n_data, *x_pred.shape))
        best_ious = F.max(
            multi_box_iou(box_x, box_y, box_w, box_h, tbox_x, tbox_y, tbox_w, tbox_h), axis=0)

        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする
        # (truthの周りのgridはconfをそのまま維持)。
        # (効率的?に計算するため右辺ではゼロ行列を利用してステップ関数を作っている)
        zeros = F.tile(cupy.array(0, dtype=cupy.float32), x_pred.shape)
        conf_learning_scale *= F.ceil(F.maximum(self.thresh - best_ious, zeros))
        conf = F.ceil(F.maximum(best_ious - self.thresh, zeros)) * conf_pred * best_ious

        # オブジェクトのない位置のloss計算
        comma_5s = F.tile(cupy.array(0.5, dtype=cupy.float32), x_pred.shape)
        loss_x = F.sum(F.squared_difference(x_pred, comma_5s) * box_learning_scale) / 2
        loss_y = F.sum(F.squared_difference(y_pred, comma_5s) * box_learning_scale) / 2
        loss_w = F.sum(F.square(w_pred) * box_learning_scale) / 2
        loss_h = F.sum(F.square(h_pred) * box_learning_scale) / 2
        loss_conf = F.sum(F.squared_difference(conf_pred, conf) * conf_learning_scale) / 2
        # loss_prob = 0 # オブジェクトのない位置は学習しない

        # オブジェクトのある位置のloss計算
        comma_5s = F.tile(cupy.array(0.5, dtype=cupy.float32), center_x.shape)
        batch = [index[0] for index in np.ndindex(n_batch, n_data)]
        anchor = [int(anchor_index[index].data) for index in np.ndindex(n_batch, n_data)]
        x_index = [
            min(int(center_x[index].data * grid_w), grid_w - 1)
            for index in np.ndindex(n_batch, n_data)
        ]
        y_index = [
            min(int(center_y[index].data * grid_h), grid_h - 1)
            for index in np.ndindex(n_batch, n_data)
        ]
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
        # padding 箇所はlearning rateを0にして無害化
        box_learning_scale_extract = F.reshape(
            cupy.array(box_learning_scale[batch, anchor, y_index, x_index].data),
            width.shape) * F.ceil(1 + width)
        conf_learning_scale_extract = F.reshape(
            cupy.array(conf_learning_scale[batch, anchor, y_index, x_index].data),
            height.shape) * F.ceil(1 + height)
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
                Variable(
                    cupy.reshape(
                        cupy.maximum(
                            -cupy.array(
                                l, dtype=cupy.float32),
                            cupy.zeros(
                                len(l), dtype=cupy.float32)), (*center_x.shape, 1))),
                prob_pred_extract.shape),
            prob_pred_extract,
            F.reshape(Variable(one_hots), prob_pred_extract.shape))
        # 余分に足した分を引く
        loss_x -= F.sum(
            F.squared_difference(x_pred_extract, comma_5s) * box_learning_scale_extract) / 2
        loss_y -= F.sum(
            F.squared_difference(y_pred_extract, comma_5s) * box_learning_scale_extract) / 2
        loss_w -= F.sum(F.square(w_pred_extract) * box_learning_scale_extract) / 2
        loss_h -= F.sum(F.square(h_pred_extract) * box_learning_scale_extract) / 2
        loss_conf -= F.sum(F.square(conf_pred_extract) * conf_learning_scale_extract) / 2
        # 真の位置のlearning rateを設定
        # 1 にする
        box_learning_scale = F.ceil(box_learning_scale_extract)
        # 10 にする(元論文では5)
        conf_learning_scale = F.ceil(box_learning_scale_extract) * 10
        # オブジェクトのある位置のlossを足し上げる
        center_x_grid = center_x * grid_w
        center_y_grid = center_y * grid_h
        center_x_shift = F.floor(center_x_grid)
        center_y_shift = F.floor(center_y_grid)
        loss_x += F.sum(
            F.squared_difference(center_x_grid - center_x_shift, x_pred_extract) *
            box_learning_scale) / 2
        loss_y += F.sum(
            F.squared_difference(center_y_grid - center_y_shift, y_pred_extract) *
            box_learning_scale) / 2
        loss_w += F.sum(
            F.squared_difference(
                F.log(F.maximum(width, -width)) - F.log(w_anchor),
                w_pred_extract) * box_learning_scale) / 2
        loss_h += F.sum(
            F.squared_difference(
                F.log(F.maximum(height, -height)) - F.log(h_anchor),
                h_pred_extract) * box_learning_scale) / 2
        loss_conf += F.sum(
            F.squared_difference(
                multi_box_iou(center_x, center_y, width, height, (x_pred_extract + center_x_shift)
                              / grid_w, (y_pred_extract + center_y_shift) / grid_h,
                              F.exp(w_pred_extract) * w_anchor, F.exp(h_pred_extract) * h_anchor),
                conf_pred_extract) * conf_learning_scale) / 2
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

    def make_teaching_data(self):
        pass

    def init_anchor(self, anchors):
        self.anchors = anchors

    def predictor(self, x):
        pass
