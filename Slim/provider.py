import tensorflow as tf
from parse_tf.decode import read_tfrecord_and_decode_into_image_label_pair_tensors


class Data_set(object):
    def __init__(self, mode, tf_data_path, batch_size, min_after_dequeue, preprocessing_name, actual_image_size, crop_image_size,shuffle):
        self.split_name = mode
        self.tfrecord_file = tf_data_path
        self.batch_size = batch_size
        self.min_after_dequeue = min_after_dequeue
        self.preprocessing_name = preprocessing_name
        self.actual_image_size = actual_image_size
        self.train_image_size = crop_image_size
        self.shuffle = shuffle

    def pre_processing(self, img, Isize=(256, 256), crop_size=(256, 256), method='default'):
        # resize and crop
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize_images(img, [Isize[0], Isize[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.random_crop(img, [crop_size[0], crop_size[1], 3])

        # preprocess
        if method == 'default':
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            img = tf.image.per_image_standardization(img)
            img = tf.reshape(img, [crop_size[0], crop_size[1], 3])

        # keras preprocess module
        return img

    def read_processing_generate_image_label_batch(self):

        if self.shuffle:
            # get filename list
            tfrecord_filename_train = tf.gfile.Glob(self.tfrecord_file + '*_%s_*' % self.split_name)
            # The file name list generator
            filename_queue_train = tf.train.string_input_producer(tfrecord_filename_train, num_epochs=None)
            # get tensor of image/label
            image, label = read_tfrecord_and_decode_into_image_label_pair_tensors(filename_queue_train,
                                                                                  self.actual_image_size)
        else:
            # get filename list
            tfrecord_filename_test = tf.gfile.Glob(self.tfrecord_file + '*_%s_*' % self.split_name)
            # The file name list generator
            filename_queue_test = tf.train.string_input_producer(tfrecord_filename_test, num_epochs=None, shuffle=False)
            # get tensor of image/label
            image, label = read_tfrecord_and_decode_into_image_label_pair_tensors(filename_queue_test,
                                                                                  self.actual_image_size)

        image = self.pre_processing(image, self.actual_image_size, self.train_image_size)

        if self.shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=self.batch_size,
                                                        capacity=256 + self.min_after_dequeue,
                                                        num_threads=8,
                                                        min_after_dequeue=self.min_after_dequeue)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                              batch_size=self.batch_size)

        return image_batch, label_batch



