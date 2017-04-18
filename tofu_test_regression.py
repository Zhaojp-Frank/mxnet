# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import logging
import argparse

num_loops = 25
cold_skip = 5

# symbol net

def get_symbol(args):
    net = mx.sym.Variable("data")
    net = mx.sym.FullyConnected(net, num_hidden=args.hidden_size)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    return net, [('data', (args.batch_size, args.feature_size)), ('softmax_label', (args.batch_size,))]


def test_net():
    # print logging by default
    #logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--feature_size', type=int, default=1024, help='Feature size')
    parser.add_argument('--hidden_size', type=int, default=1024*1024*128, help='Hidden size')
    parser.add_argument('-a', '--addresses', type=str, help='Addresses of all workers.')
    parser.add_argument('-i', '--worker_index', type=int, 
                        help='Index of this worker in addresses')
    parser.add_argument('-f', '--host_file', type=str,
                        help='Host file that contains addresses of all workers.')
    args = parser.parse_args()
    if args.host_file:
        addresses = []
        with open(args.host_file) as fp:
            for line in fp:
                if line.find(":") == -1:
                    addresses.append(line.strip() + ":9200")
                else:
                    addresses.append(line.strip())
    else:
        addresses = args.addresses.split(',')
    if args.worker_index is not None:
        worker_index = args.worker_index
    else:
        has_mpi = True
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        worker_index = comm.Get_rank()
    n_workers = len(addresses)
    group2ctx = {'group:%d' % i : mx.cpu(0, addresses[i]) for i in range(n_workers)}
    default_ctx = mx.cpu(0, addresses[worker_index])

    net, data_shapes = get_symbol(args)

    data_shapes = dict(data_shapes)
    data_types = {name: mx.base.mx_real_t for name, shp in data_shapes.items()}
    print(net.list_arguments())
    print(net.list_outputs())
    # infer shapes
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**data_shapes)
    arg_types, out_types, aux_types = net.infer_type(**data_types)
    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, default_ctx, dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))
    # create gradient ndarray for all parameters.
    args_grad = {name : mx.nd.zeros(shape, default_ctx, dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), 
                                               arg_shapes, arg_types)
                 if name != 'data' and not name.endswith('label')}
    print('Argument grads: ', args_grad.keys())

    executor = net.bind(ctx=default_ctx,
                        args=arg_arrays,
                        args_grad=args_grad,
                        grad_req='write',
                        group2ctx=group2ctx)

    all_time = []
    for i in range(num_loops):
        print('=> loop %d' % i);
        st_l = time.time()
        if i == cold_skip:
            t0 = time.time()
        outputs = executor.forward()
        executor.backward([outputs[0]])
        for name, grad in args_grad.items():
            grad.wait_to_read()
        if len(outputs) > 0:
            outputs[-1].wait_to_read()
        ed_l = time.time()
        print('=> loop duration %f' % float(ed_l - st_l))
        if (i >= cold_skip):
             all_time.append(float(ed_l - st_l))
    t1 = time.time()
    duration = t1 - t0
    print('duration %f, average %f' % (duration, 
                                       float(duration) / (num_loops - 
                                                          cold_skip)))
    print('std : %f' % np.asarray(all_time).std())


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_net()
