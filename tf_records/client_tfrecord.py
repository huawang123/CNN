"Author Hua Wang"
import tensorflow as tf
from parse_tf.convert_tf import get_tfdata

flags = tf.app.flags

# define args
flags.DEFINE_string("split_name", 'all', "convert which part of the data{'all','train','test'}")
flags.DEFINE_integer("Num_shard", 10, "tf-data numbers")

flags.DEFINE_string("dataset_name", 'cat_dog', "dataset name")
flags.DEFINE_integer("actual_image_size", 256, "actual_image_size")

flags.DEFINE_string("filelist_dir", '/storage/wanghua/kaggle/filelist/cat_dog/', "filelist save path")
flags.DEFINE_string("tfdata_path", '/storage/wanghua/kaggle/tf/cat_dogk/', "tf-records save path")

def main(_):
    # get args
    config = flags.FLAGS

    # convert data to tf-records
    get_tfdata(config)

if __name__ == "__main__":
    tf.app.run()