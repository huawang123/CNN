import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from tensorflow.contrib.layers import batch_norm

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)

def load_model(sess, saver, restore_checkpoint):
    print('Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(restore_checkpoint)
    try:

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Sucessful loading checkpoing...%s' % ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            raise TypeError('no checkpoint in %s' % restore_checkpoint)
    except Exception as e:
        print(e)

def get_image_label_pair(filelist_path):
    # 解析文本文件
    label_image = lambda x: x.strip().split('    ')
    with open(filelist_path) as f:
        label = [int(label_image(line)[0]) for line in f.readlines()]
    with open(filelist_path) as f:
        image_path_list = [label_image(line)[1] for line in f.readlines()]
    return image_path_list, label

def pre_processing(img, Isize, crop_size, method):
    # resize and crop
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [Isize, Isize], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.random_crop(img, [crop_size, crop_size, 3])

    # preprocess
    if method == 'default':
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=63)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img = tf.image.per_image_standardization(img)
        img = tf.reshape(img, [Isize, Isize, 3])

    # keras preprocess module
    return img

# 生成相同大小的批次
def get_batch(image, label, image_W=256, image_H=256, batch_size=32, capacity=256,min_after_dequeue=None, is_training=True):
    # image, label: 要生成batch的图像路径和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = pre_processing(image, image_H, image_H, 'default')
    # 统一图片大小
    # image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    if is_training:
        if not min_after_dequeue:
            image_batch, label_batch, filename = tf.train.batch([image, label, input_queue[0]],
                                                batch_size=batch_size,
                                                num_threads=64,   # 线程
                                                capacity=capacity)
        else:
            image_batch, label_batch, filename = tf.train.shuffle_batch([image, label, input_queue[0]],
                                                batch_size=batch_size,
                                                num_threads=64,  # 线程
                                                capacity=capacity + min_after_dequeue,
                                                min_after_dequeue=min_after_dequeue)

    else:
        image_batch, label_batch, filename = tf.train.batch([image, label, input_queue[0]],
                                                  batch_size=batch_size,
                                                  num_threads=64,  # 线程
                                                  capacity=capacity)
    return image_batch, label_batch, filename

@add_arg_scope
def conv2layer(images, kernel, stride, output_channel, activation='relu', padding_mode='SAME', bn=False, trainning=False, scope='conv2'):
    with tf.variable_scope(scope):
        # input_channel = images.get_shape().as_list()[3]
        # weights = tf.get_variable("w",
        #                           shape=[kernel[0], kernel[1], input_channel, output_channel],
        #                           dtype=tf.float32,
        #                           initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # biases = tf.get_variable("b",
        #                          shape=[output_channel],
        #                          dtype=tf.float32,
        #                          initializer=tf.constant_initializer(0.1))
        hidden = tf.layers.conv2d(images, output_channel, kernel, stride, padding_mode)
        if bn:
            hidden = bn2(images=hidden, training=trainning, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
            return hidden
        else:
            return hidden
        # conv = tf.nn.conv2d(images, output_channel, strides=[1, stride, stride, 1], padding=padding_mode)
        # pre_activation = tf.nn.bias_add(conv, biases)
        # if activation == 'relu':
        #     return tf.nn.relu(pre_activation, name="relu")
        # else:
        #     return pre_activation

@add_arg_scope
def deconv2layer(images, kernel, stride, output_channel, activation='relu', padding_mode='SAME', bn=False, trainning=False, scope='coner2'):
    with tf.variable_scope(scope):
        # input_channel = images.get_shape().as_list()[3]

        hidden = tf.layers.conv2d_transpose(images, output_channel, kernel, stride, padding_mode)
        if bn:
            hidden = bn2(images=hidden, training=trainning, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
            return hidden
        else:
            return hidden

@add_arg_scope
def pool2layer(images, kernel, stride, pooling_mode='max', padding_mode='SAME', scope='pool2'):
    with tf.variable_scope(scope):
        if pooling_mode == 'max':
            return tf.nn.max_pool(images, ksize=[1, kernel[0], kernel[1], 1], strides=[1, stride, stride, 1],
                               padding=padding_mode, name="pooling1")
        else:
            return tf.nn.avg_pool(images, ksize=[1, kernel[0], kernel[1], 1], strides=[1, stride, stride, 1],
                                  padding=padding_mode, name="pooling1")

@add_arg_scope
def lrn(images,scope='lrn'):
    with tf.variable_scope(scope):
        return tf.nn.lrn(images, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='lrn')

@add_arg_scope
def bn2(images, training, scope):
    with tf.variable_scope(scope):
        return batch_norm(inputs=images,scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True, is_training=training)

@add_arg_scope
def PReLU(x, scope):
    # PReLU(x) = x if x > 0, alpha*x otherwise

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha*(x - abs(x))*0.5

    return output

@add_arg_scope# function for 2D spatial dropout:
def spatial_dropout(x, drop_prob):
    # x is a tensor of shape [batch_size, height, width, channels]

    keep_prob = 1.0 - drop_prob
    input_shape = x.get_shape().as_list()

    batch_size = input_shape[0]
    channels = input_shape[3]

    # drop each channel with probability drop_prob:
    noise_shape = tf.constant(value=[batch_size, 1, 1, channels])
    x_drop = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)

    output = x_drop

    return output

@add_arg_scope# function for unpooling max_pool:
def max_unpool(inputs, pooling_indices, output_shape=None, k_size=[1, 2, 2, 1]):
    # NOTE! this function is based on the implementation by kwotsin in
    # https://github.com/kwotsin/TensorFlow-ENet

    # inputs has shape [batch_size, height, width, channels]

    # pooling_indices: pooling indices of the previously max_pooled layer

    # output_shape: what shape the returned tensor should have

    pooling_indices = tf.cast(pooling_indices, tf.int32)
    input_shape = tf.shape(inputs, out_type=tf.int32)

    one_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_pooling_indices*batch_range
    y = pooling_indices//(output_shape[2]*output_shape[3])
    x = (pooling_indices//output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_pooling_indices*feature_range

    inputs_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, inputs_size]))
    values = tf.reshape(inputs, [inputs_size])

    ret = tf.scatter_nd(indices, values, output_shape)

    return ret

@add_arg_scope
def batch_norm1(x, n_out, phase_train, name='bn'):
    beta = tf.get_variable(name + '/beta', shape=[n_out], initializer=tf.constant_initializer(),
                           trainable=phase_train)
    gamma = tf.get_variable(name + '/gamma', shape=[n_out], initializer=tf.random_normal_initializer(1., 0.02),
                            trainable=phase_train)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name=name + '/moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(tf.cast(phase_train, tf.bool),
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean),
                                 ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

def configure_optimizer(opt, learning_rate, adadelta_rho=0.95, adadelta_epsilon=1.0,
                        adagrad_initial_accumulator_value=0.1,
                        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1.0,
                        ftrl_learning_rate_power=-0.5, ftrl_initial_accumulator_value=0.1, ftrl_l1=0.0, ftrl_l2=0.0,
                        momentum_momentum=0.9,
                        rmsprop_decay=0.9, rmsprop_momentum=0.9, rmsprop_epsilon=1.0):
  """Configures the optimizer used for training.
  Args:
    learning_rate: A scalar or `Tensor` learning rate.
    adadelta_rho: The decay rate for adadelta.
    adagrad_initial_accumulator_value: Starting value for the AdaGrad accumulators.
    adam_beta1: The exponential decay rate for the 1st moment estimates.
    adam_beta2: The exponential decay rate for the 2nd moment estimates.
    *_epsilon: Epsilon term for the optimizer.
    ftrl_learning_rate_power: The learning rate power.
    ftrl_initial_accumulator_value: Starting value for the FTRL accumulators.
    ftrl_l1: The FTRL l1 regularization strength.
    ftrl_l2: The FTRL l2 regularization strength.
    momentum_momentum: The momentum for the MomentumOptimizer and RMSPropOptimizer.
    rmsprop_momentum: Momentum.
    rmsprop_decay: Decay term for RMSProp.
  Returns:
    An instance of an optimizer.
  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if opt == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=adadelta_rho,
        epsilon=adadelta_epsilon)
  elif opt == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=adagrad_initial_accumulator_value)
  elif opt == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=adam_beta1,
        beta2=adam_beta2,
        epsilon=adam_epsilon)
  elif opt == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=ftrl_learning_rate_power,
        initial_accumulator_value=ftrl_initial_accumulator_value,
        l1_regularization_strength=ftrl_l1,
        l2_regularization_strength=ftrl_l2)
  elif opt == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=momentum_momentum,
        name='Momentum')
  elif opt == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=rmsprop_decay,
        momentum=rmsprop_momentum,
        epsilon=rmsprop_epsilon)
  elif opt == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', opt)
  return optimizer

def configure_learning_rate(start_learning_rate, num_samples_per_epoch,
                             batch_size, global_step, num_epochs_per_decay=2.0,
                             learning_rate_decay_factor=0.94, end_learning_rate=0.00005,
                             learning_rate_decay_type='exponential', replicas_to_aggregate=0):
  """Configures the learning rate.
  Args:
      start_learning_rate: start_learning_rate
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.
    end_learning_rate: The minimal end learning rate used by a polynomial decay learning rate.
    learning_rate_decay_factor: Learning rate decay factor.
    num_epochs_per_decay: Number of epochs after which learning rate decays.
    replicas_to_aggregate: 0 or 1 The Number of gradients to collect before updating params.
  Returns:
    A `Tensor` representing the learning rate.
  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / batch_size *
                    num_epochs_per_decay)
  if replicas_to_aggregate:
    decay_steps /= replicas_to_aggregate

  if learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(start_learning_rate,
                                      global_step,
                                      decay_steps,
                                      learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif learning_rate_decay_type == 'fixed':
    return tf.constant(start_learning_rate, name='fixed_learning_rate')
  elif learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(start_learning_rate,
                                     global_step,
                                     decay_steps,
                                     end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  elif learning_rate_decay_type == 'mylearn':
    return tf.cond(tf.less(global_step, 5000),
                   lambda: tf.constant(0.01),
                   lambda: tf.cond(tf.less(global_step, 10000),
                                   lambda: tf.constant(0.001),
                                   lambda: tf.constant(0.0001)))
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     learning_rate_decay_type)

def count_trainable_params():
    total_parameters = 0
    a = []
    for variable in tf.trainable_variables():
        a.append(variable)
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

def model_identifier(model_type):
    print ("training model is : %s " % model_type)

def save_model(saver, sess, logdir, global_stepe, loss):
    save_path = logdir + 'model.ckpt'
    saver.save(sess, save_path, global_step=global_stepe)
    print('\nLoss in {} step : {:.9f})'
          '\ncheckpoint has been saved in : {}'.format(global_stepe, loss, logdir))

def load_model_pretrain(opt_load, checkpoint_exclude_scopes, sess, checkpoint_path):
    global_initial = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    variables_to_initial = []
    all_initial = []
    variables_to_initial.append(global_initial)
    all_initial.append(global_initial)
    for variable in tf.trainable_variables():#tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):#
        excluded = False
        for exclusion in exclusions:
            if exclusion in variable.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(variable)
        if excluded:
            variables_to_initial.append(variable)
        all_initial.append(variable)
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[len(tf.trainable_variables())+1:]:
        variables_to_initial.append(variable)
        all_initial.append(variable)

    if opt_load:
        # if tf.gfile.IsDirectory(checkpoint_path):
        # checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        # else:
        #     checkpoint_path = checkpoint_path
        tf.logging.info('Fine-tuning from %s' % checkpoint_path)
        # lasd = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(variables_to_restore)
        try:
            saver.restore(sess, checkpoint_path)
        except Exception as e:
            raise IOError("Failed to load model " "from save path: %s" % checkpoint_path)

        saver.restore(sess, checkpoint_path)
        print("Successfully load model from save path: %s" % checkpoint_path)
    else:
        variables_to_initial = all_initial
    return variables_to_initial

def variable_summaries_all(accurary=0.5):
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
    # Add summaries for losses.
    summaries.add(tf.summary.scalar('my/%s' % 'accurary', accurary))
    # Add summaries for variables.
    for variable in tf.trainable_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))
    # Add summaries for end_points.
    # for end_point in end_points:
    #     x = end_points[end_point]
    #     summaries.add(tf.summary.histogram('activations/' + end_point, x))
    #     summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))

def tf_confusion_metrics(logits, labels):
    predicted = tf.round(tf.nn.sigmoid(logits))
    actual = labels

    # Count true positives, true negatives, false positives and false negatives.
    tp = tf.count_nonzero(predicted * actual)
    tn = tf.count_nonzero((predicted - 1) * (actual - 1))
    fp = tf.count_nonzero(predicted * (actual - 1))
    fn = tf.count_nonzero((predicted - 1) * actual)

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)

    # Add metrics to TensorBoard.
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Precision', precision)
    tf.summary.scalar('Recall', recall)
    tf.summary.scalar('f-measure', fmeasure)

    return accuracy, precision, recall, fmeasure

def losses(logits, labels, weiht_decay):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        c_loss = tf.reduce_mean(cross_entropy, name="loss")
        l2_loss = tf.add_n([weiht_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = c_loss + l2_loss
        tf.summary.scalar(scope.name + "loss", loss)
    return loss

def trainning(optimizer, loss, start_learning_rate, batch_size, num_samples_per_epoch):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.variable_scope('Train') as scope:
        lr = configure_learning_rate(start_learning_rate=start_learning_rate,
                                              num_samples_per_epoch=num_samples_per_epoch,
                                              batch_size=batch_size, global_step=global_step)
        # lr = config.start_learning_rate
        optimizer = configure_optimizer(optimizer, lr)
        # learning_step = tf.train.GradientDescentOptimizer(lr).minimize(myloss, global_step=global_step)
        learning_step = optimizer.minimize(loss, global_step=global_step)
        tf.summary.scalar(scope.name + "lr", lr)
    return learning_step
    # learning_rate = configure_learning_rate(start_learning_rate)
    # with tf.name_scope("optimizer"):
    #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #
    #     train_op = optimizer.minimize(loss, global_step=global_step)
    # return train_op

def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels, dtype=tf.int64))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def build_generator(z_prior, keep_prob, trainable, reuse):
    with tf.variable_scope("gen", reuse=reuse) as gen:
        with arg_scope([deconv2layer,pool2layer], stride=1, padding_mode='SAME'):
            with arg_scope([deconv2layer], activation='relu',trainning=trainable):
                # net = tf.expand_dims(z_prior, axis=[1, 2], name='squeeze_logits')
                net = deconv2layer(images=z_prior, kernel=[1, 1], output_channel=256, scope='fc2')
                net = tf.nn.dropout(net, keep_prob=keep_prob)
                net = deconv2layer(images=net, kernel=[128, 128], output_channel=256, padding_mode='VALID', scope='fc1')
                net = 1
                a = 1

    g_params=[v for v in tf.global_variables() if v.name.startswith(gen.name)]
    with tf.name_scope("gen_params"):
      for param in g_params:
        variable_summaries(param)
    return g_params

def build_discriminator(images_true, images_generated, keep_prob, trainable=False, reuse=False):
    images = tf.stack(images_true, images_generated, 1)
    with tf.variable_scope("vgg16", reuse=reuse) as sc:
        with arg_scope([conv2layer,pool2layer], stride=1, padding_mode='SAME'):
            with arg_scope([conv2layer], activation=tf.nn.relu,trainning=trainable):
                net = conv2layer(images=images, kernel=[3, 3], output_channel=64, scope='conv1')
                net = bn2(images=net, training=trainable, reuse=reuse, scope='bn1')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=64, scope='conv2')
                net = bn2(images=net,training=trainable, reuse=reuse, scope='bn2')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling1')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=128, scope='conv3')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=128, scope='conv4')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling2')

                net = conv2layer(images=net, kernel=[128, 128], output_channel=256, padding_mode='VALID', scope='fc1')
                net = tf.nn.dropout(net, keep_prob=keep_prob)
                net = conv2layer(images=net, kernel=[1, 1], output_channel=256, scope='fc2')
                # net = tf.nn.dropout(net, keep_prob=keep_prob)

                net = conv2layer(images=net, kernel=[1, 1], output_channel=n_classes, activation=None, scope='logits')
                net = tf.squeeze(net, axis=[1, 2], name='squeeze_logits')

                return net

    x_data=tf.unstack(x_data,seq_size,1)
    x_generated=list(x_generated);
    x_in = tf.concat([x_data, x_generated],1);
    x_in=tf.unstack(x_in,seq_size,0);
    lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob) for _ in range(d_num_layers)]);
    with tf.variable_scope("dis") as dis:
      weights=tf.Variable(tf.random_normal([n_hidden, 1]));
      biases=tf.Variable(tf.random_normal([1]));
      outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_in, dtype=tf.float32);
      res=tf.matmul(outputs[-1], weights) + biases;
      y_data = tf.nn.sigmoid(tf.slice(res, [0, 0], [batch_size, -1], name=None));
      y_generated = tf.nn.sigmoid(tf.slice(res, [batch_size, 0], [-1, -1], name=None));
      d_params=[v for v in tf.global_variables() if v.name.startswith(dis.name)];
    with tf.name_scope("desc_params"):
      for param in d_params:
        variable_summaries(param);
    return y_data, y_generated, d_params;

