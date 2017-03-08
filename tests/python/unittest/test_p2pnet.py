# pylint: skip-file
from __future__ import print_function
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

TENSOR_ID=6234561
TENSOR_SHAPE=(100, 100)


def Worker1():
    net = mx.symbol.P2PNetInit(mx.symbol.Variable('init_control'),
                               address='127.0.0.1:5000')
    net = mx.symbol.P2PNetSend(data=mx.symbol.Variable('data'), control=net,
                               tensor_id=TENSOR_ID, address='127.0.0.1:5001')
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(init_control=(2,),
                                                         data=TENSOR_SHAPE)
    arg_types, out_types, aux_types = net.infer_type(
                                            init_control=mx.base.mx_real_t,
                                            data=mx.base.mx_real_t)
    print (arg_types, arg_shapes)
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)
    executor.forward()
    # executor.backward()
    time.sleep(5)


def Worker2():
    net = mx.symbol.P2PNetInit(mx.symbol.Variable('init_control'),
                               address='127.0.0.1:5001')
    net = mx.symbol.P2PNetRecv(data=mx.symbol.Variable('data'), control=net,
                               shape=TENSOR_SHAPE, dtype=np.float32,
                               tensor_id=TENSOR_ID, address='127.0.0.1:5000')
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(init_control=(2,),
                                                         data=(2,));
    arg_types, out_types, aux_types = net.infer_type(
                                        init_control=mx.base.mx_real_t,
                                        data=mx.base.mx_real_t)
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)
    executor.forward()
    time.sleep(5)


def main():
    w1 = Process(target=Worker1)
    w2 = Process(target=Worker2)
    w1.start()
    w2.start()
    w1.join()
    w2.join()


if __name__ == "__main__":
    main()
