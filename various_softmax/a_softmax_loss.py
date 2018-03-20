import tensorflow as tf


def a_softmax_loss(embeddings, labels, num_class, m, name='a_softmax'):
    """
    Angular softmax loss
    Args:
        embeddings:[N, dim] tensor, data
        labels:[N] tensor, label
        num_class: number of class
        m: critical factor
        name: name of variable scope
    Returns:
        logits: [N, num_class] tensor, logits for each class
        loss: [N] tensor, loss for each feature
    """
    if m not in [1, 2, 4]:
        raise ValueError('unsupported m=%d, only 1, 2, 4 is allowable' % (m, ))
    emb_shape = embeddings.get_shape()
    with tf.variable_scope(name):
        w = tf.get_variable('W', [emb_shape[1], num_class], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        eps = 1e-8
        logits = tf.matmul(embeddings, w) / (tf.norm(w, axis=0) + eps)
        if labels is None:
            return logits, None
        #ordinal = tf.constant(list(range(0, emb_shape[0])), tf.int64)
        ordinal = tf.range(start=0, limit=tf.shape(embeddings, out_type=tf.int64)[0], dtype=tf.int64)
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        sel_logits = tf.gather_nd(logits, ordinal_labels)
        emb_norm = tf.norm(embeddings, axis=1) + eps
        cos_theta = tf.div(sel_logits, emb_norm)
        if m == 1:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            if m == 2:
                sign0 = tf.sign(cos_theta)
                phy = 2*tf.multiply(sign0, tf.square(cos_theta)) - 1
            elif m == 4:
                cos_theta2 = tf.square(cos_theta)
                cos_theta4 = tf.square(cos_theta2)
                sign0 = tf.sign(cos_theta)
                sign3 = tf.multiply(tf.sign(2*cos_theta2 - 1), sign0)
                sign4 = 2*sign0 + sign3 - 3
                phy = sign3 * (8 * cos_theta4 - 8 * cos_theta2 + 1) + sign4
            else:
                raise ValueError('unsupported m=%d' % (m, ))
            scaled_logits = tf.multiply(phy, emb_norm)
            comb_logits = tf.add(logits, tf.scatter_nd(ordinal_labels, tf.subtract(scaled_logits, sel_logits),
                                                       tf.shape(logits, out_type=tf.int64)))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=comb_logits, labels=labels))
    return logits, loss

