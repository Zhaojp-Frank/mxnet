import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 25
cold_skip = 5

def get_symbol(args):
  data = []
  for i in range(args.num_args):
    data.append(mx.symbol.Variable('data%d' % i))
  #net = mx.symbol.ElementWiseSumOnlyFwd(*data)
  net = mx.symbol.ElementWiseSum(*data)
  return net

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser("Tofu MLP test code")
    parser.add_argument('--num_args', type=int, default=2, help='Number of inputs')
    parser.add_argument('--size', type=int, default=512*512, help='Size of the input array')

    args = parser.parse_args()
    net = get_symbol(args)
    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    data_shapes = {'data%d' % i: (args.size,) for i in range(args.num_args)}
    data_types = {'data%d' % i: mx.base.mx_real_t for i in range(args.num_args)}
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
          #print(o.asnumpy())
    t1 = time.time()
    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))


if __name__ == '__main__':
    print('================ Test ElementwiseSum  ====================')
    main()
