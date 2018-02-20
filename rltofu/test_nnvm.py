#!/usr/bin/python

from __future__ import print_function
from collections import deque
import argparse
import mxnet as mx
import numpy as np
import os
import pprint as pp


if __name__ == '__main__':
    data = mx.symbol.Variable('Data')
    net = mx.symbol.FullyConnected(data, num_hidden=4096, no_bias=True)
    net = mx.symbol.Activation(net, act_type='sigmoid')
    net = mx.symbol.SoftmaxOutput(net)
    input_names = (data,)
    input_shapes = ((256, 4096),)
    args_shape, _, _ = net.infer_shape(Data=input_shapes[0])
    args = [mx.nd.zeros(shape, mx.gpu(0)) for shape in args_shape]
    args_grad = {}
    for name, shape in zip(net.list_arguments(), args_shape):
        if name != 'data' and not name.endswith('label'):
            args_grad[name] = mx.nd.zeros(shape, mx.gpu(0))
    mx.executor.set_device_placement(1, [0 for _ in range(9)])
    executor = net.bind(ctx=mx.gpu(0), args=args, args_grad=args_grad,
                        grad_req='write')
    executor.save_graph('graph.json')
    for i in range(10):
        executor.forward()
        executor.backward()
        for grad in args_grad.values():
            grad.wait_to_read()
    print('End')
