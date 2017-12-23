import tensorflow as tf
import os
import datetime
from PIL import Image
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

gpu = 5
learning_rate = 0.0001
img_width = 256
img_height = 256
batch_size = 32
capacity = 256
min_after_dequeue = 10000
n_classes = 2
max_step = 2000000
train_data_path = 'train.txt'
eval_data_path = 'test.txt'
log_path = ''
restore_checkpoint = ''


def get_one_image(filename):
    image = Image.open(filename)
    image = image.resize([img_width, img_height])
    image = np.asarray(image)
    return image

def get_image_label_pair(filelist_path):
    # 解析文本文件
    label_image = lambda x: x.strip().split('    ')
    with open(filelist_path) as f:
        label = [int(label_image(line)[0]) for line in f.readlines()]
    with open(filelist_path) as f:
        image_path_list = [label_image(line)[1] for line in f.readlines()]
    return image_path_list, label

image_path_list, label_list = get_image_label_pair(train_data_path)
test_image_path_list, test_label_list = get_image_label_pair(eval_data_path)

# 生成相同大小的批次
def get_batch(image, label, image_W=256, image_H=256, batch_size=32, capacity=256,min_after_dequeue=None):
    # image, label: 要生成batch的图像和标签list
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

    # 统一图片大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)   # 标准化数据
    if not min_after_dequeue:
        image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size=batch_size,
                                                num_threads=64,   # 线程
                                                capacity=capacity)
    else:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                num_threads=64,  # 线程
                                                capacity=capacity + min_after_dequeue,
                                                min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch

def inference(images, batch_size, n_classes, reuse=False):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1", reuse=reuse) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn", reuse=reuse) as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2", reuse=reuse) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn", reuse=reuse) as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1", reuse=reuse) as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2", reuse=reuse) as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear", reuse=reuse) as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
    return softmax_linear

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


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels, dtype=tf.int64))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # correct_1 = tf.equal(tf.arg_max(end_points['Predictions'], dimension=1), test_label_batch)
        # accuracy = tf.reduce_mean(tf.cast(correct_1, dtype=tf.float32))
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def tr():
    train_image_batch, train_label_batch = get_batch(image_path_list, label_list, img_width, img_height, batch_size, capacity, min_after_dequeue)
    train_logits = inference(train_image_batch, batch_size, n_classes)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_label_batch)

    test_image_batch, test_label_batch = get_batch(test_image_path_list, test_label_list, img_width, img_height, batch_size, capacity)
    test_logit = inference(test_image_batch, batch_size, n_classes, reuse=True)

    test_accuracy = evaluation(test_logit, test_label_batch)

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
        try:
            f = open(log_path + 'log_gpu:%d.txt' % gpu, 'a')
            for step in range(max_step):
                if coord.should_stop():
                    break
                start = datetime.datetime.now()
                [_, tra_loss, tra_acc] = sess.run([train_op, train_loss, train_acc])
                during_time = datetime.datetime.now() - start
                if (step + 1) % 50 == 0:
                    print('Step %d, Train loss = %.3f, train accuracy = %.3f      during  %s' % (step, tra_loss, tra_acc, during_time))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if (step + 1) % 100 == 0 or (step + 1) == max_step:
                    checkpoint_path = log_path + 'model.ckpt'
                    saver.save(sess, checkpoint_path, global_step=step)
                    num_batches = int(5000 / batch_size)
                    acc_list = []
                    for p in range(num_batches):
                        [tmp_accuracy, t, tt] = sess.run([test_accuracy, test_logit, test_label_batch])
                        # if p < 2:
                        #     print(t, tt, tmp_accuracy)
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
tr()







