import tensorflow as tf
import keras.backend as K


def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = tf.log(hm_pred) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = tf.log(1 - hm_pred) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    loss = tf.cond(tf.greater(num_pos, 0), lambda: -(pos_loss - neg_loss) / num_pos, lambda: -neg_loss)
    return loss


def loss(hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, index):
    hm_loss = focal_loss(hm_pred, hm_true)
    pass
