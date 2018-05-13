import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import numpy as np
from model_unet import unet
from utils import get_cost, pixel_wise_softmax_2, iou, save_model, DataProvider
import tensorflow as tf
# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
N_CLASS = 2
EPOCHES = 500
GPU = 0
LEARNING_RATE=0.001

global_step = tf.Variable(0, name="global_step", trainable=False)
x = tf.placeholder("float", shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
y_ = tf.placeholder("int32", shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 1])
y = tf.squeeze(tf.one_hot(y_, N_CLASS, axis=3))
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

map, encode = unet(x, layers=3, features_root=16, n_class=N_CLASS, is_training=True, reuse=False)
t_vars = tf.trainable_variables()
cost = get_cost(map, y, variables=t_vars, n_class=N_CLASS, cost_name='cross_entropy',regularizer=0.0004)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(cost, global_step=global_step)

predicter = pixel_wise_softmax_2(map)
correct_pred = tf.equal(tf.argmax(predicter, 3), tf.argmax(y, 3))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
iou_score = iou(map,y)

TRAIN_PATH ='/storage/wanghua/kaggle/data/cell/stage1_train/'
VALID_PATH =  '/storage/wanghua/kaggle/data/cell/stage1_train/'
TEST_PATH =  '/storage/wanghua/kaggle/data/cell/stage1_test/'
LOG_PATH = '/home/wh/'
data = DataProvider(TRAIN_PATH, VALID_PATH, TEST_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    saver = tf.train.Saver()
    for epoch in range(EPOCHES):
        print ('---------Epoch: %s---------' % epoch)
        lss = []
        for image,mask in data.next_train_batch(BATCH_SIZE):
            _, loss, m, acc = sess.run([train_op, cost, map, accuracy],feed_dict={x:image,y_:mask,keep_prob:0.8})
            lss.append(loss)
        ac = []
        ouu = []
        for image,mask in data.next_valid_batch(BATCH_SIZE):
            acc,ou = sess.run([accuracy,iou_score], feed_dict={x: image, y_: mask, keep_prob: 0.8})
            ac.append(acc)
            ouu.append(ou)
        print("-------------------%s--------------------------%s" % (np.mean(ac),np.mean(ouu)))
        save_model(saver, sess, LOG_PATH, epoch, np.mean(lss))
        data.reset_train_batch_pointer()
        data.reset_valid_batch_pointer()