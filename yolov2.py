import numpy as np
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from lib.utils import Box, box_iou, multi_box_iou


def darknetConv2D(in_channel, out_channel, ksize=3, pad=1):
    return chainer.Chain(
        c=L.Convolution2D(
            in_channel, out_channel, ksize=ksize, stride=1, pad=pad, nobias=True),
        n=L.BatchNormalization(
            out_channel, use_beta=False),
        b=L.Bias(shape=(out_channel, )), )


def CRP(c, h, stride=2, pooling=False):
    # convolution -> leakyReLU -> MaxPooling
    h = c.b(c.n(c.c(h)))
    h = F.leaky_relu(h, slope=0.1)
    if pooling:
        h = F.max_pooling_2d(h, ksize=2, stride=stride, pad=0)
    return h


def reorg(h, stride=2):
    batch_size, input_channel, input_height, input_width = h.shape
    output_height, output_width = input_height // stride, input_width // stride
    output_channel = input_channel * (stride**2)
    output = F.transpose(
        F.reshape(h, (batch_size, input_channel, output_height, stride, output_width, stride)),
        (0, 1, 2, 4, 3, 5))
    output = F.transpose(
        F.reshape(output, (batch_size, input_channel, output_height, output_width, -1)),
        (0, 4, 1, 2, 3))
    output = F.reshape(output, (batch_size, output_channel, output_height, output_width))
    return output


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
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125],
                        [2.59375, 2.78125], [1.9375, 3.25]]
        self.thresh = 0.6
        self.seen = 0
        self.unstable_seen = 5000

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
        h = CRP(self.dark13, h)
        high_resolution_feature = reorg(h)  # 高解像度特徴量をreorgでサイズ落として保存しておく
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = CRP(self.dark14, h)
        h = CRP(self.dark15, h)
        h = CRP(self.dark16, h)
        h = CRP(self.dark17, h)
        h = CRP(self.dark18, h)

        # new layer
        h = CRP(self.dark19, h)
        h = CRP(self.dark20, h)
        h = F.concat((high_resolution_feature, h), axis=1)  # output concatnation
        h = CRP(self.dark21, h)
        h = self.bias22(self.conv22(h))

        batch_size, _, grid_h, grid_w = h.shape
        self.seen += batch_size
        px, py, pw, ph, pconf, prob = F.split_axis(
            F.reshape(h, (batch_size, self.n_boxes, self.n_classes + 5, grid_h, grid_w)),
            (1, 2, 3, 4, 5),
            axis=2)
        px = F.sigmoid(px)  # xのactivation
        py = F.sigmoid(py)  # yのactivation
        pconf = F.sigmoid(pconf)  # confのactivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob)  # probablitiyのacitivation

        # 教師データの用意
        # wとhが0になるように学習(e^wとe^hは1に近づく -> 担当するbboxの倍率1)
        # 活性化後のxとyが0.5になるように学習()
        tw = np.zeros(pw.shape, dtype=np.float32)
        th = np.zeros(ph.shape, dtype=np.float32)
        tx = np.tile(0.5, px.shape).astype(np.float32)
        ty = np.tile(0.5, py.shape).astype(np.float32)
        # confidenceのtruthは基本0、iouがthresh以上のものは学習しない
        # ただしobjectの存在するgridのbest_boxのみ真のIOUに近づかせる
        tconf = np.zeros(pconf.shape, dtype=np.float32)
        # best_anchor以外は学習させない(自身との二乗和誤差 = 0)
        tprob = prob.data.copy()
        """
        if self.seen < self.unstable_seen:  # centerの存在しないbbox誤差学習スケールは基本0.1
            box_learning_scale = np.tile(0.1, px.shape).astype(np.float32)
        else:
            box_learning_scale = np.tile(0, px.shape).astype(np.float32)
        # """
        box_learning_scale = np.tile(0.1, px.shape).astype(np.float32)
        conf_learning_scale = np.tile(0.1, pconf.shape).astype(np.float32)

        # 全bboxとtruthのiouを計算(batch単位で計算する)
        x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), px.shape[1:]))
        y_shift = Variable(
            np.broadcast_to(
                np.arange(
                    grid_h, dtype=np.float32).reshape(grid_h, 1), py.shape[1:]))
        w_anchor = Variable(
            np.broadcast_to(
                np.reshape(
                    np.array(
                        self.anchors, dtype=np.float32)[:, 0], (self.n_boxes, 1, 1, 1)),
                pw.shape[1:]))
        h_anchor = Variable(
            np.broadcast_to(
                np.reshape(
                    np.array(
                        self.anchors, dtype=np.float32)[:, 1], (self.n_boxes, 1, 1, 1)),
                ph.shape[1:]))
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        best_ious = []
        tt = chainer.cuda.to_cpu(t.data)
        n_truth_boxes = tt[0].shape[0]
        for batch in range(batch_size):
            box_x = (px[batch] + x_shift) / grid_w
            box_y = (py[batch] + y_shift) / grid_h
            box_w = F.exp(pw[batch]) * w_anchor / grid_w
            box_h = F.exp(ph[batch]) * h_anchor / grid_h
            ious = []
            for truth_index in range(n_truth_boxes):
                truth_box_x = Variable(
                    np.full(
                        box_x.shape, tt[batch, truth_index, 1], dtype=np.float32))
                truth_box_y = Variable(
                    np.full(
                        box_x.shape, tt[batch, truth_index, 2], dtype=np.float32))
                truth_box_w = Variable(
                    np.full(
                        box_x.shape, tt[batch, truth_index, 3], dtype=np.float32))
                truth_box_h = Variable(
                    np.full(
                        box_x.shape, tt[batch, truth_index, 4], dtype=np.float32))
                truth_box_x.to_gpu(), truth_box_y.to_gpu(), truth_box_w.to_gpu(
                ), truth_box_h.to_gpu()
                ious.append(
                    multi_box_iou(
                        Box(box_x, box_y, box_w, box_h),
                        Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)).data.get())
            ious = np.array(ious)
            best_ious.append(np.max(ious, axis=0))
        best_ious = np.array(best_ious)
        """
        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする
        # (truthの周りのgridはconfをそのまま維持)。
        tconf[best_ious > self.thresh] = pconf.data[best_ious > self.thresh]
        conf_learning_scale[best_ious > self.thresh] = 0
        """

        # objectの存在するanchor boxのみ、x、y、w、h、conf、probを個別修正
        abs_anchors = self.anchors / np.array([grid_w, grid_h])
        for batch in range(batch_size):
            for truth_box in tt[batch]:
                truth_w = int(truth_box[1] * grid_w)
                truth_h = int(truth_box[2] * grid_h)
                truth_n = 0
                best_iou = 0.0
                for anchor_index, abs_anchor in enumerate(abs_anchors):
                    iou = box_iou(
                        Box(0, 0, truth_box[3], truth_box[4]),
                        Box(0, 0, abs_anchor[0], abs_anchor[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_n = anchor_index

                # objectの存在するanchorについて、centerを0.5ではなく、真の座標に近づかせる。anchorのスケールを1ではなく真のスケールに近づかせる。学習スケールを1にする。
                box_learning_scale[batch, truth_n, :, truth_h, truth_w] = 1.0
                tx[batch, truth_n, :, truth_h, truth_w] = truth_box[1] * grid_w - truth_w
                ty[batch, truth_n, :, truth_h, truth_w] = truth_box[2] * grid_h - truth_h
                tw[batch, truth_n, :, truth_h, truth_w] = np.log(truth_box[3] /
                                                                 abs_anchors[truth_n][0])
                th[batch, truth_n, :, truth_h, truth_w] = np.log(truth_box[4] /
                                                                 abs_anchors[truth_n][1])
                tprob[batch, :, truth_n, truth_h, truth_w] = 0
                tprob[batch, truth_box[0], truth_n, truth_h, truth_w] = 1

                # IOUの観測
                full_truth_box = Box(truth_box[1], truth_box[2], truth_box[3], truth_box[4])
                predicted_box = Box(
                    (px[batch][truth_n][0][truth_h][truth_w].data.get() + truth_w) / grid_w,
                    (py[batch][truth_n][0][truth_h][truth_w].data.get() + truth_h) / grid_h,
                    np.exp(pw[batch][truth_n][0][truth_h][truth_w].data.get()) *
                    abs_anchors[truth_n][0],
                    np.exp(ph[batch][truth_n][0][truth_h][truth_w].data.get()) *
                    abs_anchors[truth_n][1])
                predicted_iou = box_iou(full_truth_box, predicted_box)
                tconf[batch, truth_n, :, truth_h, truth_w] = predicted_iou
                conf_learning_scale[batch, truth_n, :, truth_h, truth_w] = 10.0

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
