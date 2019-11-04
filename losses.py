import tensorflow as tf
import keras.backend as K
from keras.losses import mean_absolute_error


def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = tf.log(hm_pred) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = tf.log(1 - hm_pred) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    num_pos = tf.Print(num_pos, [num_pos], message='\nnum_pos:')
    pos_loss = tf.Print(pos_loss, [pos_loss], message='\npos_loss:')
    neg_loss = tf.Print(neg_loss, [neg_loss], message='\nneg_loss:')
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: -(pos_loss + neg_loss) / num_pos, lambda: -neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    # (b, k, 1)
    mask = tf.expand_dims(mask, axis=-1)
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-7)
    return reg_loss


def loss(args):
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    total_loss = tf.Print(total_loss, [hm_loss, wh_loss, reg_loss], message='\n:')
    return total_loss
