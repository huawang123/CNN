import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from model_segnet_utils import get_filename_list, get_batch, conv_layer_with_bn,deconv_layer, cal_loss, per_class_acc, get_hist, get_all_test_data
from model_segnet_utils import variable_with_weight_decay, variable_on_cpu, msra_initializer, add_loss_summaries, print_hist_summery, writeImage

def inference(images, labels, num_class, batch_size, phase_train):
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")
    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")
    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")
    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")
    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")
    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, num_class],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = variable_on_cpu('biases', [num_class], tf.constant_initializer(0.0))
      conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    loss = cal_loss(conv_classifier, labels, num_class)

    return loss, logit

def train(total_loss, learning_rate, moving_average_decay, global_step):
    """ fix lr """
    lr = learning_rate
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op

def training(FLAGS, is_finetune=False):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  log_dir = FLAGS.log_dir
  image_dir = FLAGS.image_dir
  val_dir = FLAGS.val_dir
  finetune_ckpt = FLAGS.finetune
  num_class = FLAGS.num_class
  learning_rate = FLAGS.learning_rate
  moving_average_decay = FLAGS.moving_average_decay
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  capacity = FLAGS.capacity
  gpu = FLAGS.gpu
  num_val = FLAGS.num_val
  min_after_dequeue = FLAGS.min_after_dequeue
  # should be changed if your model stored by different convention
  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

  image_filenames, label_filenames = get_filename_list(image_dir)
  val_image_filenames, val_label_filenames = get_filename_list(val_dir)

  with tf.Graph().as_default():
    train_data_node = tf.placeholder( tf.float32, shape=[batch_size, image_h, image_w, image_c])
    train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    global_step = tf.Variable(0, trainable=False)

    images, labels, _ = get_batch(image_filenames, label_filenames,image_h,
                                       image_w,image_c, batch_size,capacity,is_training=True, min_after_dequeue=min_after_dequeue)
    val_images, val_labels, _ = get_batch(val_image_filenames, val_label_filenames,image_h,
                                       image_w,image_c, batch_size,capacity,is_training=True)

    # Build a Graph that computes the logits predictions from the inference model.
    loss, eval_prediction = inference(train_data_node, train_labels_node,num_class, batch_size, phase_train)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = train(loss, learning_rate, moving_average_decay, global_step)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        # Build an initialization operation to run below.
        if (is_finetune == True):
            saver.restore(sess, finetune_ckpt)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Summery placeholders
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        average_pl = tf.placeholder(tf.float32)
        acc_pl = tf.placeholder(tf.float32)
        iu_pl = tf.placeholder(tf.float32)
        average_summary = tf.summary.scalar("test_average_loss", average_pl)
        acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
        iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

        for step in range(startstep, startstep + max_steps):
            image_batch, label_batch = sess.run([images, labels])
            # since we still use mini-batches in validation, still set bn-layer phase_train = True
            feed_dict = {
                train_data_node: image_batch,
                train_labels_node: label_batch,
                phase_train: True
            }
            start_time = time.time()

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if (step + 1) % 10 == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

                # eval current training batch pre-class accuracy
                pred = sess.run(eval_prediction, feed_dict=feed_dict)
                per_class_acc(pred, label_batch)

            if (step + 1) % 100 == 0 or (step + 1) == max_steps:
                print("start validating.....")
                total_val_loss = 0.0
                hist = np.zeros((num_class, num_class))
                for test_step in range(int(num_val)):
                    val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

                    _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                        train_data_node: val_images_batch,
                        train_labels_node: val_labels_batch,
                        phase_train: True
                    })
                    total_val_loss += _val_loss
                    hist += get_hist(_val_pred, val_labels_batch)
                print("val loss: ", total_val_loss / num_val)
                acc_total = np.diag(hist).sum() / hist.sum()
                iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / num_val})
                acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                print_hist_summery(hist)
                print(" end validating.... ")

                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(test_summary_str, step)
                summary_writer.add_summary(acc_summary_str, step)
                summary_writer.add_summary(iu_summary_str, step)
                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

def tst(FLAGS):
  gpu = FLAGS.gpu
  num_class = FLAGS.num_class
  moving_average_decay = FLAGS.moving_average_decay
  test_dir = FLAGS.test_dir
  test_ckpt = FLAGS.testing
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # testing should set BATCH_SIZE = 1
  batch_size = 1

  image_filenames, label_filenames = get_filename_list(test_dir)

  test_data_node = tf.placeholder(
      tf.float32,
      shape=[batch_size, image_h, image_w, image_c])

  test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 360, 480, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  loss, logits = inference(test_data_node, test_labels_node,num_class, batch_size, phase_train)

  pred = tf.argmax(logits, axis=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
      moving_average_decay)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
  os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  # sess_config = tf.ConfigProto(gpu_options=gpu_options)
  with tf.Session() as sess:
      # Load checkpoint
      saver.restore(sess, test_ckpt)

      images, labels = get_all_test_data(image_filenames, label_filenames)

      # threads = tf.train.start_queue_runners(sess=sess)
      hist = np.zeros((num_class, num_class))
      for image_batch, label_batch in zip(images, labels):

          feed_dict = {
              test_data_node: image_batch,
              test_labels_node: label_batch,
              phase_train: False
          }

          dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
          # output_image to verify
          if (FLAGS.save_image):
              writeImage(im[0], 'testing_image.png')
              # writeImage(im[0], 'out_image/'+str(image_filenames[count]).split('/')[-1])

          hist += get_hist(dense_prediction, label_batch)
          # count+=1
      acc_total = np.diag(hist).sum() / hist.sum()
      iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
      print("acc: ", acc_total)
      print("mean IU: ", np.nanmean(iu))