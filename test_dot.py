import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 25
cold_skip = 5

def get_symbol(args):
  A = mx.symbol.Variable('A')
  B = mx.symbol.Variable('B')
  #net = mx.symbol.Activation(A, act_type="sigmoid")
  net = mx.symbol.dot(A, B)
  net = mx.symbol.Activation(net, act_type="sigmoid")
  #net = mx.symbol.dot(net, B)
  return net

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser("Tofu MLP test code")
    parser.add_argument('-m', type=int, default=1024)
    parser.add_argument('-n', type=int, default=1024)
    parser.add_argument('-k', type=int, default=1024)

    args = parser.parse_args()
    net = get_symbol(args)
    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    data_shapes = {'A': (args.m, args.n), 'B': (args.n, args.k)}
    data_types = {'A': mx.base.mx_real_t, 'B': mx.base.mx_real_t}
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**data_shapes)
    arg_types, out_types, aux_types = net.infer_type(**data_types)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(), dtype=dtype) + i
                  for i, shape, dtype in zip(list(range(len(arg_shapes))), arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))

    executor = net.bind(ctx=mx.cpu(),
                        args=arg_arrays,
                        grad_req='null')

    for i in range(num_loops):
        print('=> loop %d' % i);
        if i == cold_skip:
            t0 = time.time()
        outputs = executor.forward()
        for o in outputs:
          o.wait_to_read()
    t1 = time.time()
    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))


if __name__ == '__main__':
    print('================ Test Dot ====================')
    main()
