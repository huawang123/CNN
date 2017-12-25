"Author Hua Wang"
import tensorflow as tf
from data_tool import split_and_tfrecords
flags = tf.app.flags

# define args

flags.DEFINE_integer("Num_shard", 10, "tf-data numbers")
flags.DEFINE_integer("actual_image_size", 256, "actual_image_size")
flags.DEFINE_integer("train_ratio", 0.8, "the ration train in all data")

flags.DEFINE_string("raw_data_dir", '/storage/litong/data_li/label_data.h5', "h5 data path")
flags.DEFINE_string("file_list", '/storage/litong/data_li/datalabel_list.txt', "filelist save path")
flags.DEFINE_string("tfdata_path", '/storage/litong/data_li/tfreocord/', "tf-records save path")

def main(_):
    # get args
    config = flags.FLAGS
    # split train and test data and writer into tfrecords
    split_and_tfrecords(config.train_ratio, config.raw_data_dir, config.tfdata_path, config.Num_shard, config.actual_image_size)


if __name__ == "__main__":
    tf.app.run()
