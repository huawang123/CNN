import tensorflow as tf
import os
import math
from tqdm import tqdm
from PIL import Image
import numpy as np

def get_tfdata_path(tfrecords_path, Dataset_name, shard_id, Num_shard, split_name):
    if not os.path.exists(tfrecords_path):
        os.makedirs(tfrecords_path)
    tfrecords_filename = tfrecords_path + '/%s_%s_%05d-of-%05d_classify.tfrecords' % \
                                          (Dataset_name, split_name, shard_id, Num_shard)
    return tfrecords_filename

# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value = [value]))

def _image_to_tfexample(image_data, height, width, depth, label):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      # 'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/depth': _int64_feature(depth),
      'image/class/label': _int64_feature(label),
  }))

def write_image_label_pairs_to_tfrecord(filename_pairs, tfrecords_path, Dataset_name,
                                        Num_shard, split_name, Patch_size):
    """Writes given label/image pairs to the tfrecords file.
    The function reads each label/image pair given filenames
    of image and respective label and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (label, img_filepath)
        Array of tuples of label/image
    tfrecords_filename : string
        Tfrecords filename to write the label/image pairs
    """

    # filename_pairs = random.shuffle(filename_pairs)
    # print(filename_pairs)
    num_per_shard = int(math.ceil(len(filename_pairs) / float(Num_shard)))
    permutation = np.random.permutation(len(filename_pairs))
    for shard_id in range(Num_shard):
        tfrecords_filename = get_tfdata_path(tfrecords_path, Dataset_name, shard_id, Num_shard, split_name)
        with tf.python_io.TFRecordWriter(tfrecords_filename) as tfrecords_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filename_pairs))
            for ii in tqdm(range(start_ndx, end_ndx)):

                im = Image.open(filename_pairs[permutation[ii]][1])
                im = im.resize((Patch_size, Patch_size))
                data = np.asarray(im)
                # c = np.asarray(data)
                # d = Image.fromarray(c, mode='RGB')
                # plt.imshow(d)
                # plt.savefig('/home/wh/gg.png', dpi=300)
                # print(filename_pairs[permutation[ii]][1])
                rows, cols, depth = data.shape
                try:
                    if rows != Patch_size or cols != Patch_size or depth != 3:
                        raise TypeError('Size not match : %s\n' % filename_pairs[0][1])
                except Exception as e:
                    print(e)
                    raise
                data_str = data.tostring()
                example = _image_to_tfexample(data_str, data.shape[0],data.shape[1], data.shape[2],
                                              int(filename_pairs[ii][0]))
                tfrecords_writer.write(example.SerializeToString())
                # sys.stdout.write('\r>> Converting %s image %d/%d shard %d\n' % (split_name,
                #      ii+1, len(filename_pairs), shard_id))
                # sys.stdout.flush()