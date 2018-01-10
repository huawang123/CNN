from parse_tf.encode import write_image_label_pairs_to_tfrecord

def parse_split_file(filelist_path):
    # 解析文本文件
    with open(filelist_path) as f:
        return [line.strip() for line in f.readlines()]

def get_filename_pairs(filelist_path):
    filenames = parse_split_file(filelist_path)
    filename_pairs = []
    for file in filenames:
        filename_pairs.append((file.split('    ')[0],file.split('    ')[1]))
    return filename_pairs

def get_tfdata(config):
    filetrain_pairs = get_filename_pairs(config.filelist_dir + 'train.txt')
    filetest_pairs = get_filename_pairs(config.filelist_dir + 'test.txt')
    if config.split_name == 'all' or config.split_name == 'train':
        write_image_label_pairs_to_tfrecord(filetrain_pairs,config.tfdata_path,
                                        config.dataset_name,config.Num_shard,
                                        'train',config.actual_image_size)
    if config.split_name == 'all' or config.split_name == 'test':
        write_image_label_pairs_to_tfrecord(filetest_pairs,config.tfdata_path,
                                        config.dataset_name,config.Num_shard,
                                        'test',config.actual_image_size)
