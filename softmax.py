import argparse
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split

import etlcdb

FLAGS = None


def load_dataset():
    X, y = etlcdb.load_dataset()
    return train_test_split(X, y, test_size=0.2)


def train():
    X_train, X_test, y_train, y_test = load_dataset()

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 45]))
    b = tf.Variable(tf.zeros([45]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 45])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    for i in range(1000):
        if i % 50 == 0:
            acc = sess.run(accuracy_summary, feed_dict={x: X_test, y_: y_test})
            test_writer.add_summary(acc, i)
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: X_train, y_: y_train})
            train_writer.add_summary(summary, i)
        else:
            sess.run(train_step, feed_dict={x: X_train, y_: y_train})

    print(sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))

    # saver = tf.train.Saver()
    # saver.save(sess, 'etlcdb-softmax')

    train_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str,
        default='/tmp/tensorflow/softmax/logs/softmax_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
