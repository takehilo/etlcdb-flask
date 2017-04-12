import tensorflow as tf

import etlcdb_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '',
                            """Path to the ETL8G data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    images, labels = etlcdb_input.inputs(eval_data=eval_data,
                                         data_dir=FLAGS.data_dir,
                                         batch_size=FLAGS.batch_size)

    return images, labels


def _variable_weight(name, shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape, initializer=initializer)


def _variable_bias(name, shape):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name, shape, initializer=initializer)


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_weight('weights', [5, 5, 1, 32])
        conv = tf.nn.conv2d(images, kernel,
                            strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_bias('biases', [32])
        pre_activation = conv + biases
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        tf.summary.histogram('activations', conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME',
                           name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_weight('weights', [5, 5, 32, 64])
        conv = tf.nn.conv2d(pool1, kernel,
                            strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_bias('biases', [64])
        pre_activation = conv + biases
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        tf.summary.histogram('activations', conv2)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME',
                           name='pool2')

    with tf.variable_scope('local3') as scope:
        weights = _variable_weight('weights', [7 * 7 * 64, 1024])
        biases = _variable_bias('biases', [1024])

        reshape = tf.reshape(pool2, [-1, 7 * 7 * 64])
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)
        tf.summary.histogram('activations', local3)

    keep_prob = tf.constant(0.5)
    dropout1 = tf.nn.dropout(local3, keep_prob)

    with tf.variable_scope('local4') as scope:
        weights = _variable_weight('weights', [1024, 45])
        biases = _variable_bias('biases', [45])

        logits = tf.add(tf.matmul(dropout1, weights), biases, name=scope.name)

    return logits, keep_prob


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def training(loss, global_step):
    tf.summary.scalar('loss', loss)

    opt = tf.train.AdamOptimizer(1e-4)
    grads = opt.compute_gradients(loss)
    apply_gradients = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradients]):
        train_op = tf.no_op(name='train')

    return train_op


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_sum = tf.summary.scalar('accuracy', accuracy)

    return accuracy, accuracy_sum
