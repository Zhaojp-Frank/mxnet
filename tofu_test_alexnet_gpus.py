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
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    # group 4
    #net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096, no_bias=True)
    net = mx.sym.Activation(net, act_type="relu")
    # group 5
    net = mx.sym.FullyConnected(net, num_hidden=1024, no_bias=True)

    #return net, [('data', (args.batch_size, 3, 224, 224))]
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    return net

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
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
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
    net = get_symbol(args)

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
