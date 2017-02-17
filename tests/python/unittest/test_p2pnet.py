# pylint: skip-file
from __future__ import print_function
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

TENSOR_ID=12345
TENSOR_SHAPE=(100, 100)


def Worker1():
    net = mx.symbol.NetInit(mx.symbol.Variable('init_control'),
                            address='127.0.0.1:5000')
    net = mx.symbol.NetSend(data=mx.symbol.Variable('data'), control=net,
                            tensor_id=TENSOR_ID, address='127.0.0.1:5001')
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(init_control=(2,), 
                                                         data=TENSOR_SHAPE)
    arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)
    executor.forward()
    # executor.backward()
    time.sleep(30)

    # print(net.list_arguments())
    # print(net.list_outputs())

    # # infer shapes
    # arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(args.batch_size, args.hidden_size))
    # arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)

    # # create ndarrays for all arguments.
    # arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  # for shape, dtype in zip(arg_shapes, arg_types)]
    # print('Num arguments: ', len(arg_arrays))
    # # create gradient ndarray for all parameters.
    # grad_dict = {name : mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                 # for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 # if name != 'data'}
    # print('Argument grads: ', grad_dict.keys())

    # executor = net.bind(ctx=mx.cpu(0),
                        # args=arg_arrays,
                        # args_grad=grad_dict,
                        # grad_req='write')

    # for i in range(num_loops):
        # print('=> loop %d' % i);
        # if i == cold_skip:
            # t0 = time.time()
        # outputs = executor.forward()
        # executor.backward([outputs[0]])
        # for name, grad in grad_dict.items():
            # grad.wait_to_read()
    # t1 = time.time()

    # duration = t1 - t0
    # print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))


def Worker2():
    net = mx.symbol.NetInit(mx.symbol.Variable('init_control'),
                            address='127.0.0.1:5001')
    net = mx.symbol.NetRecv(control=net, shape=(100, 100), tensor_id=12345, 
                            address='127.0.0.1:5000')
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(init_control=(2,));
    arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)
    executor.forward()
    time.sleep(30)

    # print(net.list_arguments())
    # print(net.list_outputs())

    # # infer shapes
    # arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(args.batch_size, args.hidden_size))
    # arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)

    # # create ndarrays for all arguments.
    # arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  # for shape, dtype in zip(arg_shapes, arg_types)]
    # print('Num arguments: ', len(arg_arrays))
    # # create gradient ndarray for all parameters.
    # grad_dict = {name : mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                 # for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 # if name != 'data'}
    # print('Argument grads: ', grad_dict.keys())

    # executor = net.bind(ctx=mx.cpu(0),
                        # args=arg_arrays,
                        # args_grad=grad_dict,
                        # grad_req='write')

    # for i in range(num_loops):
        # print('=> loop %d' % i);
        # if i == cold_skip:
            # t0 = time.time()
        # outputs = executor.forward()
        # executor.backward([outputs[0]])
        # for name, grad in grad_dict.items():
            # grad.wait_to_read()
    # t1 = time.time()

    # duration = t1 - t0
    # print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))


def main():
    w1 = Process(target=Worker1)
    w2 = Process(target=Worker2)
    w1.start()
    w2.start()
    w1.join()
    w2.join()


if __name__ == "__main__":
    main()
