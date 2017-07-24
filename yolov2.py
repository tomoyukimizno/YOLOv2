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

    left = F.maximum(x1 - len1_half, x2 - len2_half)
    right = F.minimum(x1 + len1_half, x2 + len2_half)

    return right - left


def multi_box_intersection(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    w = multi_overlap(a_x, a_w, b_x, b_w)
    h = multi_overlap(a_y, a_h, b_y, b_h)
    zeros = Variable(np.zeros(w.shape, dtype=np.float32))
    zeros.to_gpu()

    w = F.maximum(w, zeros)
    h = F.maximum(h, zeros)

    area = w * h
    return area


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
        high_resolution_feature = F.space2depth(h, 2)  # 高解像度特徴量をreorgでサイズ落として保存しておく
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

        batch_size, _, grid_h, grid_w = h.shape
        _, num_data, _ = t.shape  # batch_size,num_data, (label,x,y,w,h)

        # ネットワーク出力の整形
        px, py, pw, ph, pconf, prob = F.split_axis(
            F.reshape(h, (batch_size, self.n_boxes, self.n_classes + 5, grid_h, grid_w)),
            (1, 2, 3, 4, 5),
            axis=2)
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        px = F.sigmoid(F.reshape(px, (batch_size, -1)))
        py = F.sigmoid(F.reshape(py, (batch_size, -1)))
        pw = F.sigmoid(F.reshape(pw, (batch_size, -1)))
        ph = F.sigmoid(F.reshape(ph, (batch_size, -1)))
        pconf = F.sigmoid(F.reshape(pconf, (batch_size, -1)))
        prob = F.sigmoid(F.reshape(prob, (batch_size, self.n_classes, -1)))

        self.seen += batch_size
        """
        if self.seen < self.unstable_seen:  # centerの存在しないbbox誤差学習スケールは基本0.1
            box_learning_scale = np.tile(0.1, px.shape).astype(np.float32)
        else:
            box_learning_scale = np.tile(0, px.shape).astype(np.float32)
        # """
        # learning lateの初期化
        box_learning_scale = F.tile(np.array(0.1, dtype=np.float32), px.shape)
        conf_learning_scale = F.tile(np.array(0.1, dtype=np.float32), pconf.shape)
        box_learning_scale.to_gpu()
        conf_learning_scale.to_gpu()

        # objectの予測位置を算出
        x_shift = F.broadcast_to(np.arange(px.shape[-1], dtype=np.float32) % grid_w, px.shape)
        y_shift = F.broadcast_to(np.arange(py.shape[-1], dtype=np.float32) // grid_h, py.shape)
        w_anchor = F.reshape(
            F.broadcast_to(
                F.expand_dims(
                    self.anchors[:, 0], axis=1), (batch_size, self.n_boxes, grid_h * grid_w)),
            pw.shape)
        h_anchor = F.reshape(
            F.broadcast_to(
                F.expand_dims(
                    self.anchors[:, 1], axis=1), (batch_size, self.n_boxes, grid_h * grid_w)),
            pw.shape)
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        box_x = F.broadcast_to((px + x_shift) / grid_w, (num_data, *px.shape))
        box_y = F.broadcast_to((py + y_shift) / grid_h, (num_data, *py.shape))
        box_w = F.broadcast_to(F.exp(pw) * w_anchor / grid_w, (num_data, *pw.shape))
        box_h = F.broadcast_to(F.exp(ph) * h_anchor / grid_h, (num_data, *ph.shape))

        # 真の位置と比較(IOUにて)
        # バッチサイズ分まとめて比較
        label, center_x, center_y, width, height = F.separate(t, axis=2)
        tbox_x = F.broadcast_to(
            F.expand_dims(
                F.swapaxes(center_x, 0, 1), axis=2), (num_data, batch_size, box_x.shape[-1]))
        tbox_y = F.broadcast_to(
            F.expand_dims(
                F.swapaxes(center_y, 0, 1), axis=2), (num_data, batch_size, box_y.shape[-1]))
        tbox_w = F.broadcast_to(
            F.expand_dims(
                F.swapaxes(width, 0, 1), axis=2), (num_data, batch_size, box_w.shape[-1]))
        tbox_h = F.broadcast_to(
            F.expand_dims(
                F.swapaxes(height, 0, 1), axis=2), (num_data, batch_size, box_h.shape[-1]))
        best_ious = F.max(
            multi_box_iou(box_x, box_y, box_w, box_h, tbox_x, tbox_y, tbox_w, tbox_h), axis=0)

        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする
        # (truthの周りのgridはconfをそのまま維持)。
        # (効率的?に計算するため右辺ではゼロ行列を利用してステップ関数を作っている)
        zeros = F.tile(np.array(0, dtype=np.float32), px.shape)
        zeros.to_gpu()
        conf_learning_scale *= F.ceil(F.maximum(self.thresh - best_ious, zeros))
        tconf = F.ceil(F.maximum(best_ious - self.thresh, zeros)) * pconf * best_ious

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
        tn = F.argmax(multi_box_iou(zeros, zeros, ex_w, ex_h, zeros, zeros, ex_aw, ex_ah), axis=0)

        index_x = F.floor(center_x * grid_w)
        index_y = F.floor(center_y * grid_h)
        change_index = tn * grid_h * grid_w + F.cast(index_y, np.int32) * grid_h + F.cast(index_x,
                                                                                          np.int32)
        change_index.to_cpu()
        index_x.to_cpu(), index_y.to_cpu()
        change_index = [[index[0], int(change_index[index].data)]
                        for index in np.ndindex(batch_size, num_data)]
        change_matrix = np.zeros(px.shape, dtype=np.float32)
        change_x = np.zeros(px.shape, dtype=np.float32)
        change_y = np.zeros(py.shape, dtype=np.float32)
        change_w = np.zeros(pw.shape, dtype=np.float32)
        change_h = np.zeros(ph.shape, dtype=np.float32)

        for index in change_index:
            change_matrix[index[0], index[1]] = 1
            change_x[index[0], index[1]] = index_x.data - truth_w
            change_y[index[0], index[1]] = 1
            change_w[index[0], index[1]] = 1
            change_h[index[0], index[1]] = 1
        change_matrix = Variable(change_matrix)
        change_matrix.to_gpu()
        ones = F.tile(np.array(1.0, dtype=np.float32), box_learning_scale.shape)
        ones.to_gpu()
        box_learning_scale = F.linear_interpolate(change_matrix, ones, box_learning_scale)

        tt = chainer.cuda.to_cpu(t.data)

        # 教師データの用意
        # wとhが0になるように学習(e^wとe^hは1に近づく -> 担当するbboxの倍率1)
        # 活性化後のxとyが0.5になるように学習()
        tw = F.tile(np.array(0, dtype=np.float32), pw.shape)
        th = F.tile(np.array(0, dtype=np.float32), ph.shape)
        tx = F.tile(np.array(0.5, dtype=np.float32), px.shape)
        ty = F.tile(np.array(0.5, dtype=np.float32), py.shape)

        # confidenceのtruthは基本0、iouがthresh以上のものは学習しない
        # ただしobjectの存在するgridのbest_boxのみ真のIOUに近づかせる
        # tconf = np.zeros(pconf.shape, dtype=np.float32)
        # best_anchor以外は学習させない(自身との二乗和誤差 = 0)
        tconf = F.tile(np.array(0, dtype=np.float32), pconf.shape)
        tprob = prob.data.copy()
        tconf.to_gpu()

        tx = F.linear_interpolate(change_matrix, center_x * grid_w - index_x, tx)
        ty = F.linear_interpolate(change_matrix, center_y * grid_h - index_y, ty)
        tw = F.linear_interpolate(change_matrix, np.log(width / abs_anchors[:, 0]), tw)
        th = F.linear_interpolate(change_matrix, np.log(truth_box[4] / abs_anchors[:, 1]), th)

        for batch in range(batch_size):
            for truth_box in tt[batch]:
                truth_w = int(truth_box[1] * grid_w)
                truth_h = int(truth_box[2] * grid_h)

                # objectの存在するanchorについて、centerを0.5ではなく、真の座標に近づかせる。
                # anchorのスケールを1ではなく真のスケールに近づかせる。学習スケールを1にする。
                tprob[batch, :, tn, truth_h, truth_w] = 0
                tprob[batch, truth_box[0], tn, truth_h, truth_w] = 1

                # IOUの観測
                full_truth_box = Box(truth_box[1], truth_box[2], truth_box[3], truth_box[4])
                predicted_box = Box(
                    (px[batch][tn][0][truth_h][truth_w].data.get() + truth_w) / grid_w,
                    (py[batch][tn][0][truth_h][truth_w].data.get() + truth_h) / grid_h,
                    np.exp(pw[batch][tn][0][truth_h][truth_w].data.get()) * abs_anchors[trn][0],
                    np.exp(ph[batch][tn][0][truth_h][truth_w].data.get()) * abs_anchors[tn][1])
                predicted_iou = box_iou(full_truth_box, predicted_box)
                tconf[batch, tn, :, truth_h, truth_w] = predicted_iou
                conf_learning_scale[batch, tn, :, truth_h, truth_w] = 10.0

        # loss計算
        tx, ty, tw, th, tconf, tprob = Variable(tx), Variable(ty), Variable(tw), Variable(
            th), Variable(tconf), Variable(tprob)
        box_learning_scale, conf_learning_scale = Variable(box_learning_scale), Variable(
            conf_learning_scale)
        tx.to_gpu(), ty.to_gpu(), tw.to_gpu(), th.to_gpu(), tconf.to_gpu(), tprob.to_gpu()
        box_learning_scale.to_gpu()
        conf_learning_scale.to_gpu()

        x_loss = F.sum((tx - px)**2 * box_learning_scale) / 2
        y_loss = F.sum((ty - py)**2 * box_learning_scale) / 2
        w_loss = F.sum((tw - pw)**2 * box_learning_scale) / 2
        h_loss = F.sum((th - ph)**2 * box_learning_scale) / 2
        c_loss = F.sum((tconf - pconf)**2 * conf_learning_scale) / 2
        p_loss = F.sum((tprob - prob)**2) / 2
        print("x_loss: %f  y_loss: %f  w_loss: %f  h_loss: %f  c_loss: %f   p_loss: %f" %
              (F.sum(x_loss).data, F.sum(y_loss).data, F.sum(w_loss).data, F.sum(h_loss).data,
               F.sum(c_loss).data, F.sum(p_loss).data))

        loss = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss
        # chainer.report({'loss': loss}, self)

        return loss

    def make_teaching_data(self):
        pass

    def init_anchor(self, anchors):
        self.anchors = anchors

    def predictor(self, input_x):
        output = self.predictor(input_x)
        batch_size, input_channel, input_h, input_w = input_x.shape
        batch_size, _, grid_h, grid_w = output.shape
        x, y, w, h, conf, prob = F.split_axis(
            F.reshape(output, (batch_size, self.predictor.n_boxes, self.predictor.n_classes + 5,
                               grid_h, grid_w)), (1, 2, 3, 4, 5),
            axis=2)
        x = F.sigmoid(x)  # xのactivation
        y = F.sigmoid(y)  # yのactivation
        conf = F.sigmoid(conf)  # confのactivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob)  # probablitiyのacitivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))

        # x, y, w, hを絶対座標へ変換
        x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape))
        y_shift = Variable(
            np.broadcast_to(
                np.arange(
                    grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape))
        w_anchor = Variable(
            np.broadcast_to(
                np.reshape(
                    np.array(
                        self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)),
                w.shape))
        h_anchor = Variable(
            np.broadcast_to(
                np.reshape(
                    np.array(
                        self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)),
                h.shape))
        # x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        box_x = (x + x_shift) / grid_w
        box_y = (y + y_shift) / grid_h
        box_w = F.exp(w) * w_anchor / grid_w
        box_h = F.exp(h) * h_anchor / grid_h

        return box_x, box_y, box_w, box_h, conf, prob
