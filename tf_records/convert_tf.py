from parse_tf.encode import write_image_label_pairs_to_tfrecord
from tools.get_filelist import get_filename_pairs
def get_tfdata(config):
    filetrain_pairs = get_filename_pairs(config.filelist_dir + 'train.txt')#得到（（filename，label）,...）
    filetest_pairs = get_filename_pairs(config.filelist_dir + 'test.txt')
    if config.split_name == 'all' or config.split_name == 'train':
        write_image_label_pairs_to_tfrecord(filetrain_pairs,config.tfdata_path,
                                        config.dataset_name,config.Num_shard,
                                        'train',config.actual_image_size)
    if config.split_name == 'all' or config.split_name == 'test':
        write_image_label_pairs_to_tfrecord(filetest_pairs,config.tfdata_path,
                                        config.dataset_name,config.Num_shard,
                                        'test',config.actual_image_size)
