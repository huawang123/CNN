import tensorflow as tf
import numpy as np

# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def read_image_label_pairs_from_tfrecord(tfrecords_filename):
    """Return label/image pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective label matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from
    Returns
    -------
    image_annotation_pairs : array of tuples (img, label)
        The image and label that were read from the file
    """

    images_labels = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['image/height']
                     .int64_list
                     .value[0])
        width = int(example.features.feature['image/width']
                    .int64_list
                    .value[0])
        # img_format = (example.features.feature['image/format']
        #               .bytes_list
        #               .value[0])
        label = int(example.features.feature['image/class/label']
                    .int64_list
                    .value[0])
        image = (example.features.feature['image/encoded']
                             .bytes_list
                             .value[0])
        img_1d = np.fromstring(image, dtype=np.float64)
        images_labels.append((img_1d.reshape((height, width, -1)),label))
    return images_labels


def read_tfrecord_and_decode_into_image_label_pair_tensors(tfrecord_filenames_queue, size):
    """Return label/image tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped label/image tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    Returns
    -------
    image, label : tuple of tf.int32 (image, label)
        Tuple of label/image tensors
    """

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)

    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    depth = tf.cast(features['image/depth'], tf.int64)

    image = tf.reshape(image, [size,size,2])   #height,width,depth？？？
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label