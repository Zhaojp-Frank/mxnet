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
    net = mx.symbol.FullyConnected(data, num_hidden=8192, no_bias=True)
    net = mx.symbol.Activation(net, act_type='sigmoid')
    net = mx.symbol.SoftmaxOutput(net)
    input_names = (data,)
    input_shapes = ((256, 8192),)
    args_shape, _, _ = net.infer_shape(Data=input_shapes[0])
    placement = [0, 0, 0, 0, 1, 1, 1, 0, 0]
    arg_to_nid = {'Data':0, 'fullyconnected0_weight':1,
                  'softmaxoutput0_label':5}
    arg_to_shape = dict(zip(net.list_arguments(), args_shape))
    args, args_grad = mx.executor.set_device_placement(
                        2, [0, 0, 0, 0, 1, 1, 1, 0, 0], arg_to_nid,
                        arg_to_shape, mx.nd.zeros)
    executor = net.bind(ctx=mx.gpu(0), args=args, args_grad=args_grad,
                        grad_req='write')
    executor.save_graph('graph.json')
    for i in range(100):
        executor.forward()
        executor.backward()
        for grad in args_grad.values():
            grad.wait_to_read()
    print('End')
