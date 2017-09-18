import mxnet as mx

def add_args(parser):
  pass

# symbol net
def conv_factory(data, num_filter, kernel, stride=(1, 1), pad=(1, 1), with_bn=False):
    net = mx.sym.Convolution(data,
                             num_filter=num_filter,
                             kernel=kernel,
                             stride=stride,
                             pad=pad,
                             no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    return net

def get_symbol(args):
    net = mx.sym.Variable("data")
    # group 0
    net = conv_factory(net, num_filter=64, kernel=(11, 11), stride=(4, 4), pad=(2, 2))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # group 1
    net = conv_factory(net, num_filter=192, kernel=(5, 5), stride=(1, 1), pad=(2, 2))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # group 2
    net = conv_factory(net, num_filter=384, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = conv_factory(net, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = conv_factory(net, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # group 3
    net = mx.sym.Flatten(net)
    #net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True, name="&mp&fc0")
    net = mx.sym.Activation(net, act_type="relu", name="&mp&relu0")
    # group 4
    #net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True, name="&mp&fc1")
    net = mx.sym.Activation(net, act_type="relu", name="&mp&relu1")
    # group 5
    net = mx.sym.FullyConnected(net, num_hidden=1024, no_bias=True, name="&mp&fc2")
    net = mx.sym.SoftmaxOutput(net, name="softmax")
    return net, (3, 224, 224), 1000
