import tensorflow as tf
import numpy as np

class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")   # token ints
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")               # one-hot encoded class
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope('discriminator'):

            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    conv = tf.layers.conv1d(
                        inputs=self.embedded_chars,
                        filters=num_filter,
                        kernel_size=filter_size,
                        strides=1,
                        padding="valid",
                        data_format='channels_last',
                        activation=tf.nn.relu,
                        name="conv-filters-of-size-%s" % filter_size
                    )
                    # Maxpooling over the outputs
                    pool_size = sequence_length - filter_size + 1
                    pooled = tf.layers.max_pooling1d(
                        inputs=conv,
                        pool_size=pool_size,
                        strides=1,
                        padding='valid',
                        data_format='channels_last',
                        name="maxpool_for-filters-of-size-%s" % filter_size
                    )
                    pooled = tf.squeeze(pooled, [1])
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool_flat = tf.concat(pooled_outputs, 1)

            # Add highway
            with tf.name_scope("highway"):
                g = tf.layers.dense(self.h_pool_flat, num_filters_total, activation=tf.nn.relu)
                t = tf.layers.dense(self.h_pool_flat, num_filters_total, activation=tf.nn.sigmoid)
                self.h_highway = t * g + (1.0 - t) * self.h_pool_flat

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.layers.dropout(self.h_highway, rate=(1.0 - self.dropout_keep_prob), training=True)

                # ^ test time has keep prob placeholder set to 1.0,
                # so the "training=True" setting won't matter

            # Logits, etc.
            with tf.name_scope("output"):
                self.logit_batch = tf.layers.dense(self.h_drop, 1, activation=None)

                self.ypred_for_auc = tf.nn.sigmoid(self.logit_batch)

                self.predictions = tf.cast(tf.greater_equal(self.ypred_for_auc, tf.constant(0.5)),
                                           dtype=tf.int32)

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logit_batch,
                    labels=tf.expand_dims(self.input_y[:,1], 1)
                ))

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
