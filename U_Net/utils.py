import tensorflow as tf
import os
import numpy as np
import sys
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

# def concat(tensors, axis, *args, **kwargs):
#     return tf.concat(tensors, axis, *args, **kwargs)

# def like_rgb_label(x, y):
#     """Concatenate conditioning vector on feature map axis."""
#     x_shapes = x.get_shape().as_list()
#     y_shapes = y.get_shape().as_list()
#     y = tf.reshape(y, [y_shapes[0],1, 1, y_shapes[-1]])#[batch_size, 1, 1, num_classes]
#     y_ = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[-1]])#[batch_size, width, height, num_classes]
#     return y*y_

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def conv2d(input_, output_dim, kernel=(3,3), stride=(2,2),padding='SAME', activation='',use_bn=False, is_training=False, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride[0], stride[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        hidden = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if use_bn:
            hidden = bn(x=hidden, is_training=is_training, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
        elif activation == 'lrelu':
            hidden =  lrelu(hidden)
        elif activation == 'sigmoid':
            hidden =  tf.nn.sigmoid(hidden)
        return hidden

def deconv2d(input_, output_size, output_channel, kernel=(3,3), stride=(2,2),padding='SAME',
             activation='',use_bn=False, is_training=False,  name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        batch_size = input_.get_shape().as_list()[0]
        w = tf.get_variable('w', [kernel[0], kernel[1], output_channel, input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.02))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[batch_size,output_size,output_size,output_channel],
                                        strides=[1, stride[0], stride[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
        hidden = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if use_bn:
            hidden = bn(x=hidden, is_training=is_training, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
        elif activation == 'lrelu':
            hidden =  lrelu(hidden)
        elif activation == 'sigmoid':
            hidden =  tf.nn.sigmoid(hidden)
        return hidden

def crop_and_concat(x1,x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)

def get_cost(predict, ground_true, variables, n_class,  cost_name, class_weights=None, regularizer=None):
    import numpy as np
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """
    flat_logits = tf.reshape(predict, [-1, n_class])
    flat_labels = tf.reshape(ground_true, [-1, n_class])
    if cost_name == "cross_entropy":
        if class_weights is not None:
            class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                          labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
            loss = tf.reduce_mean(weighted_loss)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
    elif cost_name == "dice_coefficient":
        eps = 1e-5
        prediction = pixel_wise_softmax_2(predict)
        intersection = tf.reduce_sum(prediction * ground_true)
        union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(ground_true)
        loss = -(2 * intersection / (union))
    else:
        raise ValueError("Unknown cost function: " % cost_name)

    if regularizer is not None:
        regularizers = sum([tf.nn.l2_loss(variable) for variable in variables])
        loss += (regularizer * regularizers)
    return loss

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)

def IOU(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]
    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7
    return tf.reduce_mean(intersection / denominator)

def iou(output, target, threshold=0.5, axis=[1, 2, 3], smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, 1 means totally match.
    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou  # , pre, truth, inse, union

def save_model(saver, sess, logdir, global_stepe, ls):
    save_path = logdir + 'model.ckpt'
    saver.save(sess, save_path, global_step=global_stepe)
    print('\nloss {:.9f} in epoch : {})'
            '\ncheckpoint has been saved in : {}'.format(ls, global_stepe, logdir))

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
#
# def linear(input_, output_size, scope=None):
#     shape = input_.get_shape().as_list()
#     with tf.variable_scope(scope or "Linear"):
#         matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                  tf.random_normal_initializer(stddev=0.02))
#         bias = tf.get_variable("bias", [output_size],
#         initializer=tf.constant_initializer(0.0))
#         return tf.matmul(input_, matrix) + bias

class DataProvider(object):
    def __init__(self, TRAIN_PATH, VALID_PATH, TEST_PATH):
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 3
        self.TRAIN_PATH  = TRAIN_PATH
        self.TEST_PATH = TEST_PATH
        self.VALID_PATH = VALID_PATH
        if os.path.exists(self.TRAIN_PATH + 'image.npy'):
            self.X_train = np.load(self.TRAIN_PATH + 'image.npy')
            self.Y_train = np.load(self.TRAIN_PATH + 'mask.npy')
        else:
            self.get_train_image_mask_pair()
        if os.path.exists(self.VALID_PATH + 'image.npy'):
            self.X_valid = np.load(self.VALID_PATH + 'image.npy')
            self.Y_valid = np.load(self.VALID_PATH + 'mask.npy')
        else:
            self.get_valid_image_mask_pair()
        if os.path.exists(self.TEST_PATH + 'image.npy'):
            self.X_test = np.load(self.TEST_PATH + 'image.npy')
        else:
            self.get_tst_image()

    def pre_processing(self, img, method='default'):
        # resize and crop
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # preprocess
        if method == 'default':
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            img = tf.image.random_saturation(img, 0.3, 0.5)
            img = tf.image.per_image_standardization(img)
        # keras preprocess module
        return img

    def get_train_image_mask_pair(self):
        # Get train IDs
        train_ids = next(os.walk(self.TRAIN_PATH))[1]
        # Get and resize train images and masks
        self.X_train = np.zeros((len(train_ids), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.Y_train = np.zeros((len(train_ids), self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.bool)
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = self.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.IMG_CHANNELS]
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
            # img = self.pre_processing(img, method='default')
            self.X_train[n] = img
            mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.int32)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant',
                                                  preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
                self.Y_train[n] = mask
        print('Done!')
        np.save(self.TRAIN_PATH + 'image.npy', self.X_train)
        np.save(self.TRAIN_PATH + 'mask.npy', self.Y_train)

    def get_valid_image_mask_pair(self):
        # Get train IDs
        valid_ids = next(os.walk(self.VALID_PATH))[1]
        # Get and resize train images and masks
        self.X_valid = np.zeros((len(valid_ids), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.Y_valid = np.zeros((len(valid_ids), self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.bool)
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(valid_ids), total=len(valid_ids)):
            path = self.VALID_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.IMG_CHANNELS]
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
            # img = self.pre_processing(img, method='default')
            self.X_valid[n] = img
            mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.int32)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant',
                                                  preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
                self.Y_valid[n] = mask
        print('Done!')
        np.save(self.VALID_PATH + 'image.npy', self.X_valid)
        np.save(self.VALID_PATH + 'mask.npy', self.Y_valid)

    def get_tst_image(self):
        # Get test IDs
        test_ids = next(os.walk(self.TEST_PATH))[1]
        # Get and resize test images
        self.X_test = np.zeros((len(test_ids), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        sizes_test = []
        print('Getting and resizing test images ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = self.TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.IMG_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
            # img = self.pre_processing(img, method='default')
            self.X_test[n] = img
        print('Done!')
        np.save(self.TEST_PATH + 'image.npy', self.X_test)

    def next_train_batch(self,bs):
        permutation = np.random.permutation(len(self.X_train))
        steps = int(len(self.X_train) / bs)
        for self.train_pointer in range(steps):
            image = np.zeros((bs, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
            mask = np.zeros((bs, self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.int32)
            for i in range(bs):
                image[i] = self.X_train[permutation[self.train_pointer*bs + i]]
                mask[i] = self.Y_train[permutation[self.train_pointer * bs + i]]
            yield (image,mask)

    def next_valid_batch(self,bs):
        permutation = np.random.permutation(len(self.X_valid))
        steps = int(len(self.X_valid) / bs)
        for self.valid_pointer in range(steps):
            image = np.zeros((bs, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
            mask = np.zeros((bs, self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.int32)
            for i in range(bs):
                image[i] = self.X_valid[permutation[self.valid_pointer*bs + i]]
                mask[i] = self.Y_valid[permutation[self.valid_pointer * bs + i]]
            yield (image,mask)

    def next_tst_batch(self,bs):
        permutation = np.random.permutation(len(self.X_test))
        steps = int(len(self.X_train) / bs)
        for self.tst_pointer in range(steps):
            image = np.zeros((bs, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
            for i in range(bs):
                image[i] = self.X_test[permutation[self.tst_pointer*bs + i]]
            yield image

    def reset_train_batch_pointer(self):
        self.train_pointer = 0

    def reset_valid_batch_pointer(self):
        self.valid_pointer = 0

    def reset_tst_batch_pointer(self):
        self.tst_pointer = 0