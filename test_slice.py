import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 25
cold_skip = 5

def get_symbol(args):
  X = mx.symbol.Variable('X')
  net = mx.symbol.SliceChannel(
      X, axis=0, num_outputs=args.num_outputs)
  return net

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser("Tofu MLP test code")
    parser.add_argument('--size', type=int, default=512*512)
    parser.add_argument('--num_outputs', type=int, default=2)

    args = parser.parse_args()
    net = get_symbol(args)
    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    data_shapes = {'X': (args.size,)}
    data_types = {'X': mx.base.mx_real_t}
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**data_shapes)
    arg_types, out_types, aux_types = net.infer_type(**data_types)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
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
