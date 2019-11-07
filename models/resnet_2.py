from keras_resnet.models import ResNet18, ResNet34, ResNet50
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.layers import UpSampling2D, Concatenate
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

from losses import loss


def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


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


def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    # 每一个元素表示一个 batch_item 上属于某一个类的 boxes
    detections_per_class = []
    for cls_id in range(num_classes):
        # (num_keep_this_class_boxes, 4) score 大于 score_threshold 的当前 class 的 boxes
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = K.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    # score 大于 score_threshold 的所有 class 的 boxes
    batch_item_detections = K.concatenate(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections


def decode(hm, wh, reg, max_objects=100, nms=True, num_classes=20, score_threshold=0.1):
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
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections


def centernet(num_classes, backbone='resnet50', input_size=512, max_objects=100, score_threshold=0.1, nms=True):
    assert backbone in ['resnet18', 'resnet34', 'resnet50']
    output_size = input_size // 4
    image_input = Input(shape=(None, None, 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
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

    # C5 (b, 16, 16, 512)
    C2, C3, C4, C5 = resnet.outputs

    C5 = Dropout(rate=0.5)(C5)
    C4 = Dropout(rate=0.4)(C4)
    C3 = Dropout(rate=0.3)(C3)
    C2 = Dropout(rate=0.2)(C2)
    x = C5

    # decoder
    x = Conv2D(256, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([C4, x])
    x = Conv2D(256, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    # (b, 32, 32, 512)
    x = ReLU()(x)

    x = Conv2D(128, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([C3, x])
    x = Conv2D(128, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    # (b, 64, 64, 128)
    x = ReLU()(x)

    x = Conv2D(64, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([C2, x])
    x = Conv2D(64, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    # (b, 128, 128, 512)
    x = ReLU()(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=score_threshold,
                                         nms=nms,
                                         num_classes=num_classes))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model


if __name__ == '__main__':
    import numpy as np

    model, *_ = centernet(num_classes=20)
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    #
    # hm = np.load('/home/adam/workspace/github/xuannianz/CenterNet/hm.npy')
    # hm = np.transpose(hm, (0, 2, 3, 1))
    # wh = np.load('/home/adam/workspace/github/xuannianz/CenterNet/wh.npy')
    # wh = np.transpose(wh, (0, 2, 3, 1))
    # reg = np.load('/home/adam/workspace/github/xuannianz/CenterNet/reg.npy')
    # reg = np.transpose(reg, (0, 2, 3, 1))
    # tf_dets = decode(tf.constant(hm), tf.constant(wh), tf.constant(reg))
    # sess = tf.Session()
    # print(sess.run(tf_dets[0, :5]))
    # dets = np.load('/home/adam/workspace/github/xuannianz/CenterNet/dets.npy')
    # print(dets[0, :5])

    # y1 = np.load('y1.npy')
    # y2 = np.load('y2.npy')
    # y3 = np.load('y3.npy')
    # tf_dets, scores = decode(tf.constant(y1), tf.constant(y2), tf.constant(y3))
    # scores, *_ = topk(tf.constant(hm))
    # hm = nms(y1)
    # sess = tf.Session()
    # print(sess.run(tf_dets))
    # print(sess.run(tf.reduce_sum(hm)))
    # hm = nms(tf.constant(y1))
    # print(K.eval(hm))

    # detections = np.load('debug/1106/detections.npy')
    # detections = tf.constant(detections)
    # detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
    #                                                      num_classes=20,
    #                                                      score_threshold=0.1),
    #                        elems=[detections],
    #                        dtype=tf.float32)
