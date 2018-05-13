import tensorflow as tf
import os
import datetime
import numpy as np
from provider import Data_set
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

learning_rate = 0.0001
image_size = (256, 256)
crop_size = (256, 256)
batch_size = 32
min_after_dequeue = 5000
n_classes = 2
max_step = 20000
tf_data_path = '/storage/wanghua/kaggle/tf/cat_dogk/'
log_path = '/storage/wanghua/kaggle/log/cat_dog/'
restore_checkpoint = '/storage/wanghua/kaggle/log/cat_dog/'

model_name = 'inception_v3'
weight_decay_rate = 0.0002
train_flag = 'test'
gpu = 1
test_nums = 5000

def get_image_label_batch(mode, tf_data_path, batch_size, min_after_dequeue,
                           preprocessing_name, actual_image_size=(256,256), crop_image_size=(256,256),shuffle=True):
    ########################################
    # Preprocess get batch data   #
    ########################################
    with tf.name_scope(mode+'_get_batch'):
        Data = Data_set(mode, tf_data_path, batch_size, min_after_dequeue,
                           preprocessing_name, actual_image_size=actual_image_size, crop_image_size=crop_image_size,shuffle=shuffle)
        image_batch, label_batch = Data.read_processing_generate_image_label_batch()
    return image_batch, label_batch

def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(end_points, test_label_batch):
    with tf.variable_scope("accuracy") as scope:
        correct_1 = tf.equal(tf.arg_max(end_points['Predictions'], dimension=1), test_label_batch)
        accuracy = tf.reduce_mean(tf.cast(correct_1, dtype=tf.float32))
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def model(model_name, num_classes, weight_decay):
    from net import nets_factory
    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(model_name,
                                             num_classes=num_classes,
                                             weight_decay=weight_decay)
    return network_fn

def tr():
    train_image_batch, train_label_batch = get_image_label_batch('train', tf_data_path, batch_size, min_after_dequeue,
                                                                 'default', image_size, crop_size, shuffle=True)
    network_fn = model(model_name=model_name, num_classes=n_classes, weight_decay=weight_decay_rate)
    train_logits, train_end_points = network_fn(train_image_batch, is_training=True, reuse=False)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_end_points, train_label_batch)

    eval_image_batch, eval_label_batch = get_image_label_batch('test', tf_data_path, batch_size, min_after_dequeue,
                                                                 'default', image_size, crop_size, shuffle=False)
    eval_logit, eval_end_point = network_fn(eval_image_batch, is_training=True, reuse=True)
    eval_accuracy = evaluation(eval_end_point, eval_label_batch)

    summary_op = tf.summary.merge_all()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver()
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        f = open(log_path + 'log_gpu:%d.txt' % gpu, 'a')
        for step in range(max_step):
            start = datetime.datetime.now()
            [_, tra_loss, tra_acc] = sess.run([train_op, train_loss, train_acc])
            during_time = datetime.datetime.now() - start
            # tmp = sam[0,:,:,:]
            # import matplotlib
            # # Force matplotlib to not use any Xwindows backend.
            # matplotlib.use('Agg')
            # import matplotlib.pyplot as plt
            # plt.imshow(tmp)
            # plt.savefig('/home/wh/dsd.png', dpi=300)
            if (step + 1) % 50 == 0:
                print('Step %d, Train loss = %.3f, train accuracy = %.3f      during  %s' % (step, tra_loss, tra_acc, during_time))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if (step + 1) % 50 == 0 or (step + 1) == max_step:
                checkpoint_path = log_path + 'model.ckpt'
                saver.save(sess, checkpoint_path, global_step=step)
                num_batches = int(test_nums / batch_size)
                acc_list = []
                for p in range(num_batches):
                    [tmp_accuracy] = sess.run([eval_accuracy])
                    acc_list.append(tmp_accuracy)
                ac = np.mean(acc_list)
                print('Validation accuracy %0.5f, Step %s\n' % (ac, step))
                f.write('\nAcc is : %s in step%s\n\n' % (ac, step))
                f.flush()

def t():
    test_batch_size = 32
    network_fn = model(model_name=model_name, num_classes=n_classes,weight_decay=weight_decay_rate)

    test_image_batch, test_label_batch = get_image_label_batch('test', tf_data_path, batch_size, min_after_dequeue,
                                                               'default', image_size, crop_size, shuffle=False)
    test_logit, test_end_point = network_fn(test_image_batch, is_training=True, reuse=False)
    test_accuracy = evaluation(test_end_point, test_label_batch)

    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver()
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            num_batches = int(test_nums / batch_size)
            acc_list = []
            for p in range(num_batches):
                [tmp_accuracy] = sess.run([test_accuracy])
                acc_list.append(tmp_accuracy)
            ac = np.mean(acc_list)
            print('Test accuracy %0.5f\n' % ac)

        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)

if train_flag == 'train':
    tr()
else:
    t()


