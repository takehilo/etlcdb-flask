import argparse
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

import etlcdb

FLAGS = None


def load_dataset():
    X, y = etlcdb.load_dataset()
    return train_test_split(X, y, test_size=0.3)


def deepnn(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 45])
    b_fc2 = bias_variable([45])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def train():
    X_train, X_test, y_train, y_test = load_dataset()

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 45])

    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_mask = np.random.choice(X_train.shape[0], 50)
        x_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]

        if i % 50 == 0:
            acc = sess.run(accuracy,
                           feed_dict={x: X_train, y_: y_train, keep_prob: 1.0})
            print('step {0}, training accuracy {1}'.format(i, acc))

            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: x_batch, y_: y_batch,
                                             keep_prob: 0.5})
            train_writer.add_summary(summary, i)
        else:
            sess.run(train_step, feed_dict={x: x_batch, y_: y_batch,
                                            keep_prob: 0.5})

    print('test accuracy {0}'.format(
        sess.run(accuracy, feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})))

    # saver = tf.train.Saver()
    # saver.save(sess, 'etlcdb-cnn')


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str,
        default='/tmp/tensorflow/cnn/logs/cnn_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
