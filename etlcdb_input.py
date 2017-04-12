import os

import tensorflow as tf

IMAGE_SIZE = 28

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5625
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1620


def read_etl8g(filename_queue):
    class ETL8GRecord(object):
        pass

    result = ETL8GRecord()

    label_bytes = 1
    result.height = 28
    result.width = 28
    result.depth = 1
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])

    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filename = os.path.join(data_dir, 'etlcdb.bin')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filename = os.path.join(data_dir, 'etlcdb_test.bin')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    if not tf.gfile.Exists(filename):
        raise ValueError('Failed to find file: {0}'.format(filename))

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([filename])

    # Read examples from files in the filename queue.
    read_input = read_etl8g(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 1])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    isShuffle = not eval_data
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=isShuffle)
