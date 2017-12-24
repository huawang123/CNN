import tensorflow as tf
from flow_vgg import train_val, t_one_image
flags = tf.app.flags

# about data
flags.DEFINE_string("train_data_path", '/storage/wanghua/kaggle/filelist/cat_dog/train.txt', "train_data_path")
flags.DEFINE_string("eval_data_path", '/storage/wanghua/kaggle/filelist/cat_dog/test.txt', "eval_data_path")
flags.DEFINE_integer("num_class", 2, "num_class")

# Read Batch Config
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("min_after_dequeue", 10000, "min nums data filename in queue")
flags.DEFINE_integer("capacity", 256, "capacity")

# Data pre-processing related parameters
flags.DEFINE_string("preprocessing_name", 'default', "pre-processing_name")
flags.DEFINE_integer("train_image_size", 256, "train_image_size")

# Selection and training model and log store related parameters
flags.DEFINE_string("model_name", 'vgg16', "model name")
flags.DEFINE_string("optimizer", 'adam', "optimizer")
flags.DEFINE_integer("weight_decay", 1e-7, "weight_decay")
flags.DEFINE_integer("keep_prob", 0.8, "keep_prob")
flags.DEFINE_integer("max_step", 50000, "max_steps")

# initial and store related parameters
flags.DEFINE_integer("snapshot", 100, "snapshot")
flags.DEFINE_integer("snapshot_save", 100, "saver model in snapshot")
flags.DEFINE_string("train_split_name", True, "True: training and validation; False:testing")
# flags.DEFINE_bool("opt_load", True, "Whether load model existed")
# flags.DEFINE_string("check_point_path", ' ', "check_point_path")
# flags.DEFINE_string("checkpoint_exclude_scopes", [],
#                     "checkpoint_exclude_scopes contains the logits if num_class change")
flags.DEFINE_string("log_path", '/storage/wanghua/kaggle/log/cat_dog/', "log path")
flags.DEFINE_string("restore_checkpoint", '/storage/wanghua/kaggle/log/cat_dog/', "restore_checkpoint path")

# learning related parameters
flags.DEFINE_integer("decay_steps",600, "decay_steps")
flags.DEFINE_integer("start_learning_rate", 0.008, "start_learning_rate")
flags.DEFINE_integer("decay_size", 0.5, "decay_size")

# device related parameters
flags.DEFINE_integer("gpu", 0, "which GPU")
# flags.DEFINE_string("cpu", '0', "which CPU")
flags.DEFINE_bool("log_device_placement", False, "Whether printing equipment distribution log file")
flags.DEFINE_bool("allow_soft_placement", False, "If the specified device does not exist automatically assigned")
flags.DEFINE_integer("gpu_memory_fraction", 0.8, "per_gpu_memory_fraction")

def main(_):
    # get args
    config = flags.FLAGS

    if config.train_split_name:
        train_val(config)
    else:
        t_one_image(config)

if __name__ == "__main__":
    tf.app.run()