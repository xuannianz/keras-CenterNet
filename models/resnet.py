from keras_resnet.models import ResNet18, ResNet34, ResNet50
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda
from keras.applications import ResNet50V2
from keras.models import Model
from keras.initializers import normal, constant, zeros
import keras.backend as K
import tensorflow as tf

from losses import loss


def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    keep = (hmax == heat)
    return heat * tf.cast(keep, tf.float32)


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=100):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    return detections


def centernet(num_classes, backbone='resnet18', input_size=512, max_objects=100):
    assert backbone in ['resnet18', 'resnet34', 'resnet50']
    output_size = input_size // 4
    image_input = Input(shape=(None, None, 3))
    hm_input = Input(shape=(output_size, output_size, 3))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone == 'resnet18':
        resnet = ResNet18(image_input, include_top=False, freeze_bn=True)
    elif backbone == 'resnet34':
        resnet = ResNet34(image_input, include_top=False, freeze_bn=True)
    else:
        resnet = ResNet50(image_input, include_top=False, freeze_bn=True)

    # (b, 16, 16, 512)
    C5 = resnet.outputs[-1]

    # decoder
    x = C5
    for i in range(3):
        x = Conv2DTranspose(256, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer=normal(mean=0., stddev=0.001))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same')(x)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, bias_initializer=constant(value=-2.19), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same')(x)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer=normal(0, 0.001))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same')(x)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer=normal(0, 0.001))(y3)

    loss_ = loss(y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input)
    model = Model(inputs=[image_input], outputs=[y1, y2, y3])

    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    return model, prediction_model


if __name__ == '__main__':
    centernet(num_classes=20)
    # import numpy as np
    #
    # hm = np.load('/home/adam/workspace/github/xuannianz/CenterNet/hm.npy')
    # hm = np.transpose(hm, (0, 2, 3, 1))
    # wh = np.load('/home/adam/workspace/github/xuannianz/CenterNet/wh.npy')
    # wh = np.transpose(wh, (0, 2, 3, 1))
    # reg = np.load('/home/adam/workspace/github/xuannianz/CenterNet/reg.npy')
    # reg = np.transpose(reg, (0, 2, 3, 1))
    # tf_dets = decode(tf.constant(hm), tf.constant(wh), tf.constant(reg))
    # print(tf_dets[0, :5])
    # dets = np.load('/home/adam/workspace/github/xuannianz/CenterNet/dets.npy')
    # print(dets[0, :5])
