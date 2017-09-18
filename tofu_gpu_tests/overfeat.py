import mxnet as mx

# symbol net
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
    return net


def add_args(parser):
  pass


def get_symbol(args):
    net = mx.sym.Variable("data")
    # 1
    net = conv_factory(net, num_filter=96, kernel=(7, 7), stride=(2, 2), pad=(0, 0))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(3, 3), pool_type="max")
    # 2
    net = conv_factory(net, num_filter=256, kernel=(7, 7), stride=(1, 1), pad=(0, 0))
    net = mx.sym.Pooling(net, kernel=(2, 2), stride=(2, 2), pool_type="max")
    # 3
    net = conv_factory(net, num_filter=512, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    # 4
    net = conv_factory(net, num_filter=512, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    # 5
    net = conv_factory(net, num_filter=1024, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    # 6
    net = conv_factory(net, num_filter=1024, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(3, 3), pool_type="max")
    # 7 
    net = mx.sym.Flatten(net)
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    # 8
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    # 9
    net = mx.sym.FullyConnected(net, num_hidden=1024, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")

    net = mx.sym.SoftmaxOutput(net, name='softmax')
    return net, (3, 221, 221), 1000
