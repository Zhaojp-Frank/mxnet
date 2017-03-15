from __future__ import print_function
import argparse
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

BATCH_SIZE = 1024
WEIGHT_SIZE = 1024
NUM_LAYERS = 5
NUM_ITERATIONS = 2
NUM_IGNORED_ITERATIONS = 1

class Experiment:
    def __init__(self, iterations, ignored_iterations):
        self.iterations = iterations
        self.ignored_iterations = ignored_iterations
        self.count = 0
        self.exps = []

    def __enter__(self):
        self.count += 1
        if (self.count > self.ignored_iterations):
            self.begin = time.time()

    def __exit__(self, type, value, traceback):
        if (self.count > self.ignored_iterations):
            end = time.time()
            self.exps.append(end - self.begin)

    def Summary(self):
        assert len(self.exps) == (self.iterations - self.ignored_iterations), len(self.exps)
        exps = np.asarray(self.exps)
        avg = exps.mean()
        std = exps.std()
        print('avg: %f, std: %f' % (avg, std))


def Worker(addresses, worker_index):
    group2ctx = {'machine0': mx.cpu(0, addresses[0]),
                 'machine1': mx.cpu(0, addresses[1])}

    arg_arrays = {}
    with mx.AttrScope(ctx_group='machine0'):
        data0 = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE / 2), dtype=np.float32)
    with mx.AttrScope(ctx_group='machine1'):
        data1 = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE / 2), dtype=np.float32)

    weight_shape = (WEIGHT_SIZE / 2, WEIGHT_SIZE)
    for i in range(NUM_LAYERS):
        with mx.AttrScope(ctx_group='machine0'):
            var_name =  'w_%d_%d' % (i, 0)
            partial_activation0 = mx.symbol.dot(
                                      data0, mx.symbol.Variable(
                                          var_name, shape=weight_shape,
                                          dtype=np.float32))
            partial_activation0 = mx.symbol.SliceChannel(partial_activation0,
                                                         axis=1, num_outputs=2)
            arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)

        with mx.AttrScope(ctx_group='machine1'):
            var_name =  'w_%d_%d' % (i, 1)
            partial_activation1 = mx.symbol.dot(
                                      data1, mx.symbol.Variable(
                                          var_name, shape=weight_shape,
                                          dtype=np.float32))
            partial_activation1 = mx.symbol.SliceChannel(partial_activation1,
                                                         axis=1, num_outputs=2)
            arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)

        with mx.AttrScope(ctx_group='machine0'):
            data0 = partial_activation0[0] + partial_activation1[0]
            # data0 = mx.symbol.SliceChannel(partial_activation0 + partial_activation1,
                                           # axis=1, num_outputs=2)[0]

        with mx.AttrScope(ctx_group='machine1'):
            data1 = partial_activation0[1] + partial_activation1[1]
            # data1 = mx.symbol.SliceChannel(partial_activation0 + partial_activation1,
                                           # axis=1, num_outputs=2)[1]

    # net = data0 if worker_index == 0 else data1
    net = mx.symbol.Group([data0, data1])
    arg_shapes, out_shapes, aux_shapes = net.infer_shape()
    arg_types, out_types, aux_types = net.infer_type()
    executor = net.bind(ctx=mx.cpu(0, addresses[worker_index]), args=arg_arrays,
                        group2ctx=group2ctx)
    exp = Experiment(NUM_ITERATIONS, NUM_IGNORED_ITERATIONS)
    for i in range(NUM_ITERATIONS):
        with exp:
            output = executor.forward()
            out = output[-1].asnumpy()
            print("Finish an iteration")
    exp.Summary()


def Single():
    data0 = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE), dtype=np.float32)
    weight_shape = (WEIGHT_SIZE, WEIGHT_SIZE)
    arg_arrays = {}
    for i in range(NUM_LAYERS):
        var_name = 'w_%d' % i
        activation = mx.symbol.dot(data0, mx.symbol.Variable(var_name,
                                                             shape=weight_shape,
                                                             dtype=np.float32))
        arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)
        data0 = activation

    net = mx.symbol.Group(data0)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape()
    arg_types, out_types, aux_types = net.infer_type()
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)

    exp = Experiment(NUM_ITERATIONS, NUM_IGNORED_ITERATIONS)
    for i in range(NUM_ITERATIONS):
        with exp:
            output = executor.forward()
            out = output[0].asnumpy()
    exp.Summary()


def main():
    global BATCH_SIZE
    global WEIGHT_SIZE
    global NUM_LAYERS
    global NUM_ITERATIONS
    global NUM_IGNORED_ITERATIONS
    parser = argparse.ArgumentParser(description='Test p2pnet operators with '
                                                 'new Context implementation.')
    parser.add_argument('-a', '--addresses', type=str,
                        help='Addresses of all workers.')
    parser.add_argument('-i', '--worker_index', type=int,
                        help='Index of this worker in addresses.')
    parser.add_argument('-s', '--single_machine', action='store_const',
                        const=True, help='Use single machine only.')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size',
                        default=BATCH_SIZE)
    parser.add_argument('-w', '--weight_size', type=int, help='Weight size',
                        default=WEIGHT_SIZE)
    parser.add_argument('-l', '--num_layers', type=int, help='Number of layers',
                        default=NUM_LAYERS)
    parser.add_argument('-n', '--num_iterations', type=int,
                        help='Number of ignored iteraions',
                        default=NUM_ITERATIONS)
    parser.add_argument('-g', '--num_ignored_iterations', type=int,
                        help='Number of ignored iterations when timing.',
                        default=NUM_IGNORED_ITERATIONS)
    args = parser.parse_args()
    BATCH_SIZE=int(args.batch_size)
    WEIGHT_SIZE=int(args.weight_size)
    NUM_LAYERS=int(args.num_layers)
    NUM_ITERATIONS=int(args.num_iterations)
    NUM_IGNORED_ITERATIONS=int(args.num_ignored_iterations)
    if args.single_machine:
        Single()
    else:
        addresses = args.addresses.split(',')
        assert len(addresses) == 2
        Worker(addresses, int(args.worker_index))


if __name__ == "__main__":
    main()
