# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 30
cold_skip = 5

rng = np.random.RandomState(seed=32)

has_activation = True
def Activation(data, **kwargs):
    if has_activation:
        return mx.sym.Activation(data=data, **kwargs)
    else:
        return data

expand_shortcut = False
def Expand(data, num_skips):
    if expand_shortcut:
        for i in range(num_skips):
            data = mx.sym.identity(data=data)
        return data
    else:
        return data

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix='', attr={}):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix), no_bias=True)
    act = mx.symbol.Activation(data=conv, act_type='relu', name='relu_%s%s' %(name, suffix), attr=attr)
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    c1x1 = Expand(c1x1, num_skips=4)
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    c3x3 = Expand(c3x3, num_skips=2)
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    cproj = Expand(cproj, num_skips=3)
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    c3x3 = Expand(c3x3, num_skips=2)
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max", name=('max_pool_%s_pool' % name))
    pooling = Expand(pooling, num_skips=5)
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3, name, attr):
    # conv 3x3
    conv = ConvFactory(data=data, name=name+'_conv',kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1), attr=attr)
    # pool
    pool = mx.symbol.Pooling(data=data, name=name+'_pool',kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', attr=attr)
    # concat
    concat = mx.symbol.Concat(*[conv, pool], name=name+'_ch_concat')
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, name, attr):
    # 1x1
    conv1x1 = ConvFactory(data=data, name=name+'_1x1', kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1, attr=attr)
    # 3x3
    conv3x3 = ConvFactory(data=data, name=name+'_3x3', kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3, attr=attr)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3], name=name+'_ch_concat')
    return concat


def get_symbol(num_classes, image_shape, **kwargs):
    (nchannel, height, width) = image_shape
    # attr = {'force_mirroring': 'true'}
    attr = {}

    # data
    data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='1')
    pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool_1', pool_type='max')
    # stage 2
    conv2red = ConvFactory(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='2_red')
    conv2 = ConvFactory(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='2')
    pool2 = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool_2', pool_type='max')
    # stage 2
    in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    pool = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')

    # linear classifier
    flatten = mx.symbol.Flatten(data=pool)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, no_bias=True)
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

def feed_args(net, arg_arrays):
    names = net.list_arguments()
    for name, arr in zip(names, arg_arrays):
        if not name.endswith('label'):
            # create random data
            arr[:] = 0.1 * rng.randn(*(arr.shape))

def test_mlp():
    has_mpi = False
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("Tofu MLP test code")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--num_layers', type=int, default=50, help='Number of hidden layers')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('-f', '--host_file', type=str,
                        help='Host file that contains addresses of all workers.')

    args = parser.parse_args()
    group2ctx = {'group:%d' % i : mx.gpu(i) for i in range(args.num_gpus)}
    if args.num_gpus == 1:
      default_ctx = mx.gpu(0)
    else:
      default_ctx = mx.cpu(0)

    image_shape = (3, 224, 224)
    num_classes = 1000
    net = get_symbol(num_classes, image_shape)

    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    in_shapes = {}
    #in_shapes = {'fc%d_weight' % i: (args.hidden_size, args.hidden_size) for i in range(args.num_layers)}
    in_shapes['data'] = (args.batch_size, ) + image_shape
    in_types = {}
    #in_types = {'fc%d_weight' % i : mx.base.mx_real_t for i in range(args.num_layers)}
    in_types['data'] = mx.base.mx_real_t
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**in_shapes)
    arg_types, out_types, aux_types = net.infer_type(**in_types)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, default_ctx, dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))
    # create gradient ndarray for all parameters.
    args_grad = {name : mx.nd.zeros(shape, default_ctx, dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 if name != 'data' and not name.endswith('label')}
    print('Argument grads: ', args_grad.keys())

    executor = net.bind(ctx=default_ctx,
                        args=arg_arrays,
                        args_grad=args_grad,
                        grad_req='write',
                        group2ctx=group2ctx)
    return

    feed_args(net, arg_arrays)
    all_time = []
    for i in range(num_loops):
        print('=> loop %d' % i);
        st_l = time.time()
        if i == cold_skip:
            t0 = time.time()
        outputs = executor.forward()
        executor.backward()
        for name, grad in args_grad.items():
            #print(name, grad.asnumpy())
            grad.wait_to_read()
        # XXX(minjie): Currently, the last output is used to synchronize all send nodes.
        # Send nodes may not appear on the dependency path of the local graph.
        # We need make sure all send nodes have finished before the end of the iteration.
        if len(outputs) > 0:
            outputs[-1].wait_to_read()
        ed_l = time.time()
        print('=> loop duration %f' % float(ed_l - st_l))
        if (i >= cold_skip):
             all_time.append(float(ed_l - st_l))
    t1 = time.time()

    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))
    print('std : %f' % np.asarray(all_time).std())


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_mlp()
