import tensorflow as tf
import os
import datetime
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

learning_rate = 0.0001
image_size = (256, 256)
crop_size = (256, 256)
batch_size = 32
capacity = 256
min_after_dequeue = 10000
n_classes = 2
max_step = 20000
train_data_path = '/storage/wanghua/kaggle/filelist/cat_dog/train.txt'
eval_data_path = '/storage/wanghua/kaggle/filelist/cat_dog/test.txt'
log_path = '/storage/wanghua/kaggle/log/cat_dog/'
restore_checkpoint = '/storage/wanghua/kaggle/log/cat_dog/'

model_name = 'inception_v3'
weight_decay_rate = 0.0002

def get_image_label_pair(filelist_path):
    # 解析文本文件
    label_image = lambda x: x.strip().split('    ')
    with open(filelist_path) as f:
        label = [int(label_image(line)[0]) for line in f.readlines()]
    with open(filelist_path) as f:
        image_path_list = [label_image(line)[1] for line in f.readlines()]
    return image_path_list, label

def pre_processing(img, Isize=(256,256), crop_size=(256,256), method='default'):
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

# 生成相同大小的批次
def get_batch(image, label, image_size=(256,256), crop_size=(256,256), batch_size=32, capacity=256,min_after_dequeue=None, is_training=True):
    # image, label: 要生成batch的图像路径和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    with tf.variable_scope('input'):
        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.int64)
        # 生成队列
        input_queue = tf.train.slice_input_producer([image, label])
        image_contents = tf.read_file(input_queue[0])
        label = input_queue[1]
        image = tf.image.decode_jpeg(image_contents, channels=3)
        # pre_processing
        image = pre_processing(image, image_size, crop_size, 'default')

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
    train_image_batch, train_label_batch, _ = get_batch(train_image_path_list, train_label_list, image_size, crop_size, batch_size, capacity, min_after_dequeue)
    network_fn = model(model_name=model_name, num_classes=n_classes,weight_decay=weight_decay_rate)
    train_logits, train_end_points = network_fn(train_image_batch, is_training=True, reuse=False)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_end_points, train_label_batch)

    eval_image_batch, eval_label_batch, _ = get_batch(test_image_path_list, test_label_list, image_size, crop_size, batch_size, capacity)
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
                    num_batches = int(test_nums / batch_size)
                    acc_list = []
                    for p in range(num_batches):
                        [tmp_accuracy] = sess.run([eval_accuracy])
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

def t():
    test_batch_size = 32
    network_fn = model(model_name=model_name, num_classes=n_classes,weight_decay=weight_decay_rate)

    test_image_batch, test_label_batch, _ = get_batch(test_image_path_list, test_label_list, image_size, crop_size, test_batch_size, capacity)
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

train_image_path_list, train_label_list = get_image_label_pair(train_data_path)
test_image_path_list, test_label_list = get_image_label_pair(eval_data_path)
train_flag = False
gpu = 0
test_nums = 5000
if train_flag:
    tr()
else:
    t()


