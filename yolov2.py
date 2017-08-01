import numpy as np
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
    zeros = Variable(np.zeros(w.shape, dtype=np.float32))
    zeros.to_gpu()

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
        self.anchors = np.array(
            [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125],
             [1.9375, 3.25]],
            dtype=np.float32)
        self.thresh = 0.6
        self.seen = 0
        self.unstable_seen = 5000

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
        _, n_data, _ = t.shape  # batch_size,num_data, (label,x,y,w,h)

        # NW出力の整形
        """
        # 問題なければこっちの記述に切り替えたい
        t_pred, conf_pred, prob = F.split_axis(
            F.reshape(h, (n_batch, self.n_boxes, self.n_classes + 5, grid_h, grid_w)), (4, 5),
            axis=2)
        t_pred = F.sigmoid(t_pred)
        """
        x_pred, y_pred, w_pred, h_pred, conf_pred, prob_pred = F.split_axis(
            F.reshape(h, (n_batch, self.n_boxes, self.n_classes + 5, grid_h, grid_w)),
            (1, 2, 3, 4, 5),
            axis=2)
        x_pred = F.sigmoid(F.squeeze(x_pred))  # n_batch, n_class, grid_w, grid_h
        y_pred = F.sigmoid(F.squeeze(y_pred))  # n_batch, n_class, grid_w, grid_h
        w_pred = F.sigmoid(F.squeeze(w_pred))  # n_batch, n_class, grid_w, grid_h
        h_pred = F.sigmoid(F.squeeze(h_pred))  # n_batch, n_class, grid_w, grid_h
        conf_pred = F.sigmoid(F.squeeze(conf_pred))  # n_batch, n_class, grid_w, grid_h
        prob_pred = F.sigmoid(F.transpose(prob_pred, (0, 2, 1, 3, 4)))

        self.seen += n_batch
        """
        if self.seen < self.unstable_seen:  # centerの存在しないbbox誤差学習スケールは基本0.1
            box_learning_scale = np.tile(0.1, px.shape).astype(np.float32)
        else:
            box_learning_scale = np.tile(0, px.shape).astype(np.float32)
        # """
        # learning lateの初期化
        box_learning_scale = F.tile(np.array(0.1, dtype=np.float32), conf_pred.shape)
        conf_learning_scale = F.tile(np.array(0.1, dtype=np.float32), conf_pred.shape)
        box_learning_scale.to_gpu()
        conf_learning_scale.to_gpu()

        # 真の値の整形
        label, center_x, center_y, width, height = F.separate(t, axis=2)

        # objectの存在するanchor boxの探索
        abs_anchors = (self.anchors / np.array([grid_w, grid_h])).astype(np.float32)
        ex_w = F.broadcast_to(width, (self.n_boxes, *width.shape))
        ex_h = F.broadcast_to(height, (self.n_boxes, *height.shape))
        ex_aw = F.broadcast_to(
            F.reshape(abs_anchors[:, 0], (self.n_boxes, 1, 1)), (self.n_boxes, *width.shape))
        ex_ah = F.broadcast_to(
            F.reshape(abs_anchors[:, 1], (self.n_boxes, 1, 1)), (self.n_boxes, *height.shape))
        zeros = F.tile(np.array(0, dtype=np.float32), ex_w.shape)
        ex_aw.to_gpu(), ex_ah.to_gpu(), zeros.to_gpu()
        anchor_index = F.argmax(
            multi_box_iou(zeros, zeros, ex_w, ex_h, zeros, zeros, ex_aw, ex_ah), axis=0)

        # objectの予測位置を算出
        x_shift = F.broadcast_to(np.arange(grid_w, dtype=np.float32), x_pred.shape)
        y_shift = F.broadcast_to(
            np.arange(
                grid_h, dtype=np.float32).reshape(grid_h, 1), y_pred.shape)
        w_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 0], (self.n_boxes, 1, 1)), w_pred.shape)
        h_anchor = F.broadcast_to(
            F.reshape(self.anchors[:, 1], (self.n_boxes, 1, 1)), h_pred.shape)
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
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
        zeros = F.tile(np.array(0, dtype=np.float32), x_pred.shape)
        zeros.to_gpu()
        conf_learning_scale *= F.ceil(F.maximum(self.thresh - best_ious, zeros))
        conf = F.ceil(F.maximum(best_ious - self.thresh, zeros)) * conf_pred * best_ious
        conf.to_gpu()

        # temp:オブジェクトのない位置のloss計算
        comma_5s = F.tile(np.array(0.5, dtype=np.float32), x_pred.shape)
        comma_5s.to_gpu()
        loss_x = F.sum(F.square(x_pred - comma_5s) * box_learning_scale) / 2
        loss_y = F.sum(F.square(y_pred - comma_5s) * box_learning_scale) / 2
        loss_w = F.sum(F.square(w_pred) * box_learning_scale) / 2
        loss_h = F.sum(F.square(h_pred) * box_learning_scale) / 2
        loss_conf = F.sum(F.square(conf_pred - conf) * conf_learning_scale) / 2
        # loss_prob = 0 # オブジェクトのない位置は学習しない

        # temp:オブジェクトのある位置のloss計算
        comma_5s = F.tile(np.array(0.5, dtype=np.float32), center_x.shape)
        comma_5s.to_gpu()
        x_pred_extract = np.zeros(center_x.shape, dtype=np.float32)
        y_pred_extract = np.zeros(center_y.shape, dtype=np.float32)
        w_pred_extract = np.zeros(width.shape, dtype=np.float32)
        h_pred_extract = np.zeros(height.shape, dtype=np.float32)
        conf_pred_extract = np.zeros(height.shape, dtype=np.float32)
        box_learning_scale_extract = np.zeros(center_x.shape, dtype=np.float32)
        conf_learning_scale_extract = np.zeros(center_x.shape, dtype=np.float32)
        w_anchor = np.zeros(width.shape, dtype=np.float32)
        h_anchor = np.zeros(height.shape, dtype=np.float32)
        prob = np.zeros((n_batch, self.n_classes, self.n_boxes, grid_h, grid_w), dtype=np.float32)
        for index in np.ndindex(n_batch, n_data):
            batch = index[0]
            anchor = int(anchor_index[index].data)
            x_index = int(center_x[index].data * grid_w)
            y_index = int(center_y[index].data * grid_h)
            l = int(label[index].data)
            x_pred_extract[index] = x_pred[batch, anchor, y_index, x_index].data
            y_pred_extract[index] = y_pred[batch, anchor, y_index, x_index].data
            w_pred_extract[index] = w_pred[batch, anchor, y_index, x_index].data
            h_pred_extract[index] = h_pred[batch, anchor, y_index, x_index].data
            conf_pred_extract[index] = conf_pred[batch, anchor, y_index, x_index].data
            box_learning_scale_extract[index] = box_learning_scale[batch, anchor, y_index,
                                                                   x_index].data
            conf_learning_scale_extract[index] = conf_learning_scale[batch, anchor, y_index,
                                                                     x_index].data
            w_anchor[index] = abs_anchors[anchor][0]
            h_anchor[index] = abs_anchors[anchor][1]
            prob[batch, l, anchor, y_index, x_index] = 1
        x_pred_extract = Variable(x_pred_extract)
        y_pred_extract = Variable(y_pred_extract)
        w_pred_extract = Variable(w_pred_extract)
        h_pred_extract = Variable(h_pred_extract)
        conf_pred_extract = Variable(conf_pred_extract)
        box_learning_scale_extract = Variable(box_learning_scale_extract)
        conf_learning_scale_extract = Variable(conf_learning_scale_extract)
        w_anchor = Variable(w_anchor)
        h_anchor = Variable(h_anchor)
        prob = Variable(prob)
        x_pred_extract.to_gpu(), y_pred_extract.to_gpu()
        w_pred_extract.to_gpu(), h_pred_extract.to_gpu(), conf_pred_extract.to_gpu()
        box_learning_scale_extract.to_gpu(), conf_learning_scale_extract.to_gpu()
        w_anchor.to_gpu(), h_anchor.to_gpu(), prob.to_gpu()
        # 余分に足した分を引く
        # ToDo:learning_rateのサンプリング(厳密には上で行う)
        loss_x -= F.sum(F.square(x_pred_extract - comma_5s) * box_learning_scale_extract) / 2
        loss_y -= F.sum(F.square(y_pred_extract - comma_5s) * box_learning_scale_extract) / 2
        loss_w -= F.sum(F.square(w_pred_extract) * box_learning_scale_extract) / 2
        loss_h -= F.sum(F.square(h_pred_extract) * box_learning_scale_extract) / 2
        loss_conf -= F.sum(F.square(conf_pred_extract) * conf_learning_scale_extract) / 2

        box_learning_scale = F.tile(np.array(1.0, dtype=np.float32), center_x.shape)
        box_learning_scale.to_gpu()
        conf_learning_scale = F.tile(np.array(10, dtype=np.float32), center_x.shape)
        conf_learning_scale.to_gpu()
        # オブジェクトのある位置のlossを足し上げる
        center_x_grid = center_x * grid_w
        center_y_grid = center_y * grid_h
        center_x_shift = F.floor(center_x_grid)
        center_y_shift = F.floor(center_y_grid)
        loss_x += F.sum(
            F.square(center_x_grid - center_x_shift - x_pred_extract) * box_learning_scale) / 2
        loss_y += F.sum(
            F.square(center_y_grid - center_y_shift - y_pred_extract) * box_learning_scale) / 2
        loss_w += F.sum(
            F.square(F.log(width) - F.log(w_anchor) - w_pred_extract) * box_learning_scale) / 2
        loss_h += F.sum(
            F.square(F.log(height) - F.log(h_anchor) - h_pred_extract) * box_learning_scale) / 2
        loss_conf += F.sum(
            F.square(
                multi_box_iou(center_x, center_y, width, height, (x_pred_extract + center_x_shift)
                              / grid_w, (y_pred_extract + center_y_shift) / grid_h,
                              F.exp(w_pred_extract) * w_anchor, F.exp(h_pred_extract) * h_anchor) -
                conf_pred_extract) * conf_learning_scale) / 2
        loss_prob = F.sum(F.square(prob - prob_pred)) / 2
        # """
        print("x_loss: %f  y_loss: %f  w_loss: %f  h_loss: %f  c_loss: %f   p_loss: %f" %
              (loss_x.data, loss_y.data, loss_w.data, loss_h.data, loss_conf.data, loss_prob.data))
        # """
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_prob
        # chainer.report({'loss': loss}, self)

        return loss

    def make_teaching_data(self):
        pass

    def init_anchor(self, anchors):
        self.anchors = anchors
