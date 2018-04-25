import tensorflow as tf


def sequence_mask(lens, max_len):
    len_t = tf.expand_dims(lens, 1)
    range_t = tf.range(0, max_len, 1)
    range_row = tf.expand_dims(range_t, 0)
    mask = tf.cast(tf.less(range_row, len_t), tf.float32)
    return mask


def layer_normalization(inputs, epsilon=1e-8, scope='ln', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [2], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def init_embedding(variable_name, voc_size, embed_size, fix_zero=False, init_value=None):
    if init_value is None:
        embedding = tf.get_variable(variable_name,
                                    shape=[voc_size, embed_size],
                                    initializer=\
                                    tf.random_normal_initializer(mean=0.0,
                                                                 stddev=1.0,
                                                                 seed=123),
                                    dtype=tf.float32)
    else:
        embedding = tf.get_variable(variable_name,
                                    shape=[voc_size, embed_size],
                                    initializer=tf.constant(init_value),
                                    dtype=tf.float32)
    if fix_zero:
        embedding = tf.concat(0, [tf.zeros((1, embed_size)),
                                  tf.slice(embedding, [1, 0], [-1, -1])])

    return embedding


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn


def label_smooth(labels, num_class):
    labels = tf.one_hot(labels, depth=num_class)
    return 0.9 * labels + 0.1 / num_class
