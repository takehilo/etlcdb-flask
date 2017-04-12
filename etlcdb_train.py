from datetime import datetime
import time

import tensorflow as tf

import etlcdb

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/etlcdb_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def run():
    global_step = tf.contrib.framework.get_or_create_global_step()

    images_batch, labels_batch = etlcdb.inputs(False)

    logits, keep_prob = etlcdb.inference(images_batch)

    loss = etlcdb.loss(logits, labels_batch)

    train_op = etlcdb.training(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.

        def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = \
                    FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)

                print(
                    '{0}: step {1}, loss = {2:.2f} ({3:.1f} examples/sec; '
                    ' {4:.3f} sec/batch)'
                    .format(
                        datetime.now(), self._step, loss_value,
                        examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               _LoggerHook()]) as sess:

        while not sess.should_stop():
            sess.run(train_op)


def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    run()


if __name__ == '__main__':
    tf.app.run()
