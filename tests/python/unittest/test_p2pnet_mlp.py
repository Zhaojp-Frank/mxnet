from __future__ import print_function
import argparse
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

TENSOR_ID=6234561
TENSOR_SHAPE=(100, 100)


BATCH_SIZE = 1024 
WEIGHT_SIZE = 1024

def Worker(addresses, worker_index):
    group2ctx = {'machine0': mx.cpu(0, addresses[0]),
                 'machine1': mx.cpu(0, addresses[1])}

    arg_arrays = {}
    with mx.AttrScope(ctx_group='machine0'):
        data0 = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE / 2), dtype=np.float32)
    with mx.AttrScope(ctx_group='machine1'):
        data1 = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE / 2), dtype=np.float32)

    weight_shape = (WEIGHT_SIZE / 2, WEIGHT_SIZE)
    for i in range(1):
        with mx.AttrScope(ctx_group='machine0'):
            var_name =  'w_%d_%d' % (i, 0)
            partial_activation0 = mx.symbol.dot(
                                      data0, mx.symbol.Variable(
                                          var_name, shape=weight_shape, 
                                          dtype=np.float32))
            partial_activation0 *= 1.5
            arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)

        with mx.AttrScope(ctx_group='machine1'):
            var_name =  'w_%d_%d' % (i, 1)
            partial_activation1 = mx.symbol.dot(
                                      data1, mx.symbol.Variable(
                                          var_name, shape=weight_shape, 
                                          dtype=np.float32))
            partial_activation1 *= 2
            arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)

        with mx.AttrScope(ctx_group='machine0'):
            data0 = mx.symbol.SliceChannel(partial_activation0 + partial_activation1, 
                                           axis=1, num_outputs=2)[0] * 100

        with mx.AttrScope(ctx_group='machine1'):
            data1 = mx.symbol.SliceChannel(partial_activation0 + partial_activation1, 
                                           axis=1, num_outputs=2)[1] * 10
        
    # net = data0 if worker_index == 0 else data1
    net = mx.symbol.Group([data0, data1])
    arg_shapes, out_shapes, aux_shapes = net.infer_shape()
    print("============>", out_shapes)
    arg_types, out_types, aux_types = net.infer_type()
    executor = net.bind(ctx=mx.cpu(0, addresses[worker_index]), args=arg_arrays,
                        group2ctx=group2ctx)
    print("-----------------------------------------------------------------")
    print(time.time())
    output = executor.forward()
    print(len(output))
    out = output[0].asnumpy()
    print(out)
    out = output[1].asnumpy()
    print(out)
    print(time.time())
    print("-----------------------------------------------------------------")
    # executor.backward()
    time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Test p2pnet operators with '
                                                 'new Context implementation.')
    parser.add_argument('-a', '--addresses', type=str,
                        help='Addresses of all workers.')
    parser.add_argument('-w', '--worker_index', type=int,
                        help='Index of this worker in addresses.')
    args = parser.parse_args()
    addresses = args.addresses.split(',')
    assert len(addresses) == 2
    Worker(addresses, int(args.worker_index))



if __name__ == "__main__":
    main()
