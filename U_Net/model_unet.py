
from utils import *
from collections import OrderedDict

def unet(x, layers, features_root, n_class, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("unet", reuse=reuse):
        batch_size = x.get_shape().as_list()[0]
        dw_convs = OrderedDict()
        up_convs = OrderedDict()
        net = x
        for layer in range(0, layers):
            net = conv2d(net, output_dim=2**layer*features_root, kernel=(3, 3), stride=(1, 1), padding='SAME', activation='relu', use_bn=True,
                         is_training=is_training, name='conv%s-0' % layer)
            net = conv2d(net, output_dim=2**layer*features_root, kernel=(3, 3), stride=(1, 1), padding='SAME', activation='relu', use_bn=True,
                         is_training=is_training, name='conv%s-1' % layer)
            dw_convs[layer] = net
            if layer <layers-1:
                net = max_pool(net, 2)

        in_node = dw_convs[layers-1]

        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer +  1) * features_root
            output_size = net.get_shape().as_list()[1]
            net = deconv2d(net, output_size=output_size*2, output_channel=features//2, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True,
                           is_training=True, name='d_conv_%s' % layer)
            net = crop_and_concat(dw_convs[layer], net)
            up_convs[layer] = net
            net = conv2d(net, output_dim=features//2, kernel=(3, 3), stride=(1, 1), padding='SAME', activation='relu', use_bn=True,
                         is_training=is_training, name='up_conv%s-0' % layer)
            net = conv2d(net, output_dim=features//2, kernel=(3, 3), stride=(1, 1), padding='SAME', activation='relu', use_bn=True,
                         is_training=is_training, name='up_conv%s-1' % layer)
        out_map = conv2d(net, output_dim=n_class, kernel=(1, 1), stride=(1, 1), activation='', padding='SAME', name='map')#sigmoid
        return out_map,in_node