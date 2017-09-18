import mxnet as mx

vgg_type = \
{
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def conv_factory(data, num_filter, kernel, stride=(1, 1), pad=(1, 1), with_bn=False):
    net = mx.sym.Convolution(data,
                             num_filter=num_filter,
                             kernel=kernel,
                             stride=stride,
                             pad=pad,
                             no_bias=True)
    if with_bn:
        net = mx.sym.BatchNorm(net, fix_gamma=False)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    return net


def vgg_body_factory(structure_list):
    net = mx.sym.Variable("data")
    for item in structure_list:
        if type(item) == str:
            net = mx.sym.Pooling(net, kernel=(2, 2), stride=(2, 2), pool_type="max")
        else:
            net = conv_factory(net, num_filter=item, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    return net

def add_args(parser):
  pass


def get_symbol(args, net_type='D'):
    net = vgg_body_factory(vgg_type[net_type])
    # group 3
    net = mx.sym.Flatten(net)
    #net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    # group 4
    #net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    # group 5
    net = mx.sym.FullyConnected(net, num_hidden=1024, no_bias=True)
    net = mx.sym.SoftmaxOutput(net, name="softmax")
    return net, (3, 224, 224), 1000
