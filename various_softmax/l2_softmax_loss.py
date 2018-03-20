import tensorflow as tf

def l2_softmax(embeddings, num_class):
    """
    L2-constrain softmax loss
    Args:
        embeddings:[N, dim] tensor, data
        num_class: num of classes
    Returns:
        logits: [N, num_class] tensor, logits for each class
    """
    embed_norm = embeddings / tf.norm(embeddings, axis=1, keepdims=True)
    scale_factor = tf.Variable(5, dtype=tf.float32)
    scaled_embedding = embed_norm * scale_factor
    logits = tf.layers.dense(scaled_embedding, num_class)
    return logits