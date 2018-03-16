import tensorflow as tf


def a_softmax_loss(embeddings, labels, l, num_class, m, name='a_softmax'):
    """
    Angular softmax loss
    Args:
        embeddings:[N, dim] tensor, data
        labels:[N, 1] tensor, label
        l: lambda
        num_class: number of class
        m: critical factor
        name: name of variable scope
    """
    if m not in [1, 2, 4]:
        raise ValueError('unsupported m=%d, only 1, 2, 4 is allowable' % (m, ))
    emb_shape = embeddings.get_shape()
    with tf.variable_scope(name):
        w = tf.get_variable('W', [emb_shape[1], num_class], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        eps = 1e-8
        logits = tf.matmul(embeddings, w) / (tf.norm(w, axis=0) + eps)
        ordinal = tf.constant(list(range(0, emb_shape[0])), tf.int64)
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        sel_logits = tf.gather_nd(logits, ordinal_labels)
        emb_norm = tf.nor(embeddings, axis=1) + eps
        cos_theta = tf.div(sel_logits, emb_norm)
