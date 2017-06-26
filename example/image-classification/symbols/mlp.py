"""
a simple multilayer perceptron
"""
import mxnet as mx

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    net  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=2048, no_bias=True)
    net = mx.symbol.Activation(data = net, name='relu1', act_type="relu")
    net  = mx.symbol.FullyConnected(data = net, name = 'fc2', num_hidden = 2048, no_bias=True)
    net = mx.symbol.Activation(data = net, name='relu2', act_type="relu")
    net  = mx.symbol.FullyConnected(data = net, name = 'fc3', num_hidden = 2048, no_bias=True)
    net = mx.symbol.Activation(data = net, name='relu3', act_type="relu")
    net  = mx.symbol.FullyConnected(data = net, name = 'fc4', num_hidden = 2048, no_bias=True)
    net = mx.symbol.Activation(data = net, name='relu4', act_type="relu")
    net  = mx.symbol.FullyConnected(data = net, name = 'fc5', num_hidden = 2048, no_bias=True)
    net = mx.symbol.Activation(data = net, name='relu5', act_type="relu")
    net  = mx.symbol.FullyConnected(data = net, name='fc', num_hidden=num_classes, no_bias=True)
    mlp  = mx.symbol.SoftmaxOutput(data = net, name = 'softmax')
    return mlp
