# pylint: skip-file
from __future__ import print_function
import argparse
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

TENSOR_ID=6234561
TENSOR_SHAPE=(100, 100)


def Worker(addresses, worker_index):
    group2ctx = {'machine0': mx.cpu(0, addresses[0]),
                 'machine1': mx.cpu(0, addresses[1])}
    with mx.AttrScope(ctx_group='machine0'):
        net_init = mx.symbol.P2PNetInit(mx.symbol.Variable('init_control'),
                                        address=addresses[worker_index])
        net = mx.symbol.P2PNetSend(data=mx.symbol.Variable('data'), 
                                   control=net_init, tensor_id=TENSOR_ID,
                                   address=addresses[0])
    with mx.AttrScope(ctx_group='machine1'):
        net = mx.symbol.P2PNetRecv(data=net, control=net_init, 
                                   shape=TENSOR_SHAPE, tensor_id=TENSOR_ID,
                                   address=addresses[1], dtype=np.float32)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(init_control=(2,), 
                                                         data=TENSOR_SHAPE)
    arg_types, out_types, aux_types = net.infer_type(
                                            init_control=mx.base.mx_real_t)
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(0, addresses[0]), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    executor = net.bind(ctx=mx.cpu(0, addresses[0]), args=arg_arrays,
                        group2ctx=group2ctx)
    executor.forward()
    # executor.backward()
    time.sleep(5)


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
    worker = Process(target=Worker, args=(addresses, args.worker_index))
    worker.start()
    worker.join()



if __name__ == "__main__":
    main()
