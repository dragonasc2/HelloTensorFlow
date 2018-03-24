import tensorflow as tf
import numpy as np

def arc_softmax_loss(embeddings, labels, num_class, s, m, name='arc_softmax'):
    """
    Angular softmax loss
    Args:
        embeddings:[N, dim] tensor, data
        labels:[N] tensor, label
        num_class: number of class
        s: float, scale factor
        m: critical factor
        name: name of variable scope
    Returns:
        logits: [N, num_class] tensor, logits for each class
        loss: [N] tensor, loss for each feature
    """
    normed_embeddings = tf.nn.l2_normalize(embeddings, 1)
    emb_shape = embeddings.get_shape()
    with tf.variable_scope(name):
        weights = tf.get_variable('W', [emb_shape[1], num_class], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        normed_weights = tf.nn.l2_normalize(weights, axis=0)
        cos_theta = tf.matmul(normed_embeddings, normed_weights, name='cos_theta')
        logits = tf.multiply(cos_theta, s, name='logits')
        if labels is None:
            return logits, None
        ordinal = tf.range(start=0, limit=tf.shape(embeddings, out_type=tf.int64)[0], dtype=tf.int64)
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        sel_cos_theta = tf.gather_nd(cos_theta, ordinal_labels, name='selected_cos_theta')
        sel_sin_theta = tf.sqrt(1 - tf.square(sel_cos_theta) + 1e-6, name='selected_sin_theta')
        ph = sel_cos_theta * np.cos(m) - sel_sin_theta * np.sin(m)
        comb_logits = tf.multiply(tf.add(cos_theta, tf.scatter_nd(ordinal_labels, tf.subtract(ph, sel_cos_theta),
                                                   tf.shape(cos_theta, out_type=tf.int64))), s, name='comb_logits')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=comb_logits, labels=labels))
    return logits, loss