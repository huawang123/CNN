import tensorflow as tf
import os
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
import utils
from model_vgg import vgg16

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def train_val(config):
    image_path_list, label_list = utils.get_image_label_pair(config.train_data_path)
    train_image_batch, train_label_batch, _ = utils.get_batch(image_path_list, label_list, config.train_image_size, config.train_image_size,
                                                        config.batch_size, config.capacity, config.min_after_dequeue)
    train_logits = vgg16(train_image_batch, config.num_class, keep_prob=config.keep_prob, trainable=True, reuse=False)
    train_loss = utils.losses(train_logits, train_label_batch, config.weight_decay)
    train_op = utils.trainning(config.optimizer, train_loss, config.start_learning_rate, config.batch_size, len(label_list))
    train_acc = utils.evaluation(train_logits, train_label_batch)

    eval_image_path_list, eval_label_list = utils.get_image_label_pair(config.eval_data_path)
    eval_image_batch, eval_label_batch, _ = utils.get_batch(eval_image_path_list, eval_label_list, config.train_image_size,
                                                      config.train_image_size, config.batch_size, config.capacity)
    eval_logit = vgg16(eval_image_batch, config.num_class, keep_prob=1.0, trainable=False, reuse=True)
    eval_accuracy = utils.evaluation(eval_logit, eval_label_batch)

    # get variable_summaries
    utils.variable_summaries_all(eval_accuracy)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # GPU configure
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % config.gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                 log_device_placement=config.log_device_placement,
                                 allow_soft_placement=config.allow_soft_placement
                                 )
    with tf.Session(config=sess_config) as sess:
        train_writer = tf.summary.FileWriter(config.log_path, sess.graph)
        utils.model_identifier(config.model_name)
        utils.count_trainable_params()
        saver = tf.train.Saver()
        utils.load_model(sess=sess, saver=saver, restore_checkpoint=config.restore_checkpoint)
        with tf.device('/gpu:%d' % config.gpu):
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                f = open(config.log_path + 'log_gpu:%d.txt' % config.gpu, 'a')
                for step in range(config.max_step):
                    if coord.should_stop():
                        break
                    start = datetime.datetime.now()
                    [_, tra_loss, tra_acc] = sess.run([train_op, train_loss, train_acc])
                    during_time = datetime.datetime.now() - start
                    if (step + 1) % 50 == 0:
                        print('Step %d, Train loss = %.3f, train accuracy = %.3f      during  %s' % (
                        step, tra_loss, tra_acc, during_time))
                        summary_str = sess.run(summary_op)
                        train_writer.add_summary(summary_str, step)
                    if (step + 1) % 100 == 0 or (step + 1) == config.max_step:
                        checkpoint_path = config.log_path + 'model.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)
                        num_batches = int(5000 / config.batch_size)
                        acc_list = []
                        for p in range(num_batches):
                            [tmp_accuracy, t, tt] = sess.run([eval_accuracy, eval_logit, eval_label_batch])
                            acc_list.append(tmp_accuracy)
                        ac = np.mean(acc_list)
                        print('Validation accuracy %0.5f, Step %s\n' % (ac, step))
                        f.write('\nAcc is : %s in step%s\n\n' % (ac, step))
                        f.flush()

            except tf.errors.OutOfRangeError:
                print("done!")
            finally:
                coord.request_stop()
            coord.join(threads)

def t_one_image(config):
    test_image_path_list, test_label_list = utils.get_image_label_pair(config.eval_data_path)
    batch_size = 1
    with tf.Graph().as_default():
        test_image_batch, test_label_batch, filename = utils.get_batch(test_image_path_list, test_label_list, config.train_image_size,
                                                                 config.train_image_size,batch_size, config.capacity)
        one_logit = vgg16(test_image_batch, config.num_class, keep_prob=1.0, trainable=False, reuse=False)
        one_predict = tf.nn.softmax(one_logit)

        os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % config.gpu
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            utils.load_model(sess=sess, saver=saver, restore_checkpoint=config.restore_checkpoint)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                correct_count = 0; pre_list = []; true_list = []; filename_list = []
                for _ in tqdm(range(len((test_label_list)))):
                    if coord.should_stop():
                        break
                    # image=cv2.resize(image,(img_width,img_height),interpolation=cv2.INTER_CUBIC)
                    pr, pt, file = sess.run([one_predict,test_label_batch, filename])
                    # print("########",prediction)
                    max_index = np.argmax(pr)
                    # print("Pre label %s True label %s        filename %s" % (max_index, pt[0] , file[0]))
                    pre_list.append(max_index); true_list.append(pt[0]); filename_list.append(file[0])
                    if max_index == pt[0]:
                        correct_count += 1.0
                print('Validation accuracy ', correct_count / len(test_label_list))
            except tf.errors.OutOfRangeError:
                print("done!")
            finally:
                coord.request_stop()
            coord.join(threads)