# pylint: skip-file
from __future__ import print_function
import argparse
import math
import mxnet as mx
import numpy as np
import time

BATCH_SIZE = 512
IN_CHANNEL = 1024
NUM_FILTER = 1024
HEIGHT = 6
WEIGHT = 6
NUM_LAYERS = 5
NUM_ITERATIONS = 3
NUM_IGNORED_ITERATIONS = 2
EXPERIMENT_NAME = ""

class Experiment:
    def __init__(self, iterations, ignored_iterations, name):
        self.iterations = iterations
        self.ignored_iterations = ignored_iterations
        self.count = 0
        self.exps = []
        self.name = name

    def __enter__(self):
        self.count += 1
        if (self.count > self.ignored_iterations):
            self.begin = time.time()

    def __exit__(self, type, value, traceback):
        if (self.count > self.ignored_iterations):
            end = time.time()
            self.exps.append(end - self.begin)
            print("Spent end %f  on this iteration " % (end - self.begin))

    def Summary(self):
        assert len(self.exps) == (self.iterations - self.ignored_iterations), len(self.exps)
        exps = np.asarray(self.exps)
        avg = exps.mean()
        std = exps.std()
        line = 'avg: %f, std: %f\n' % (avg, std)
        print(line)


def conv_test(default_ctx, group2ctx):
    print('Batch size: %d, #Channel: %d, #Filter: %d' % (BATCH_SIZE, IN_CHANNEL,
                                                         NUM_FILTER))
    weight_shape = (NUM_FILTER, IN_CHANNEL, 3, 3)
    grad_dict = {}
    net = mx.symbol.Variable('data')
    for i in range(NUM_LAYERS):
        net = mx.symbol.Convolution(data=net, 
                                    num_filter=NUM_FILTER, 
                                    kernel=(3,3), 
                                    stride=(1,1), 
                                    pad=(1,1),
                                    no_bias=True)

    data_shapes = {'data' : (BATCH_SIZE, IN_CHANNEL, HEIGHT, WEIGHT)}
    data_types = {name: mx.base.mx_real_t for name, shp in data_shapes.items()}

    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**data_shapes)
    arg_types, out_types, aux_types = net.infer_type(**data_types)
    print(out_shapes)
    
    arg_arrays = [mx.nd.ones(shape, mx.cpu(0), dtype=dtype) for shape, dtype in zip(arg_shapes, arg_types)]
    grad_dict = {name : mx.nd.ones(shape, mx.cpu(0), dtype=dtype)
                                  for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                                  if name != 'data'}
    print(grad_dict)
    
    executor = net.bind(ctx=default_ctx, 
                        args=arg_arrays, 
                        args_grad=grad_dict, 
                        grad_req='write', 
                        group2ctx=group2ctx)

    exp = Experiment(NUM_ITERATIONS, NUM_IGNORED_ITERATIONS, EXPERIMENT_NAME)
    for i in range(NUM_ITERATIONS):
        with exp:
            outputs = executor.forward()
            executor.backward([outputs[0]])
            for name, grad in grad_dict.items():
                grad.wait_to_read()
        print("=" * 30)
        print("Finish an iteration %d" % i)
    exp.Summary()

def main():
    global BATCH_SIZE
    global IN_CHANNEL
    global NUM_FILTER
    global NUM_LAYERS
    global NUM_ITERATIONS
    global NUM_IGNORED_ITERATIONS
    parser = argparse.ArgumentParser(description='Test MLP Convolution')
    #parser.add_argument('-a', '--test_all', action='store_const',
    #                    const=True, help='Test with all sizes.')
    parser.add_argument('-a', '--addresses', type=str, help='Addresses of all workers.')
    parser.add_argument('-i', '--worker_index', type=int, help='Index of this worker in addresses')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size',
                        default=BATCH_SIZE)
    parser.add_argument('-c', '--channel_size', type=int, help='Input channel size',
                        default=IN_CHANNEL)
    parser.add_argument('-f', '--filter_size', type=int, help='Filter size',
                        default=NUM_FILTER)
    parser.add_argument('-l', '--num_layers', type=int, help='Number of layers',
                        default=NUM_LAYERS)
    parser.add_argument('-n', '--num_iterations', type=int,
                        help='Number of ignored iteraions',
                        default=NUM_ITERATIONS)
    parser.add_argument('-g', '--num_ignored_iterations', type=int,
                        help='Number of ignored iterations when timing.',
                        default=NUM_IGNORED_ITERATIONS)
    parser.add_argument('-t', '--host_file', type=str,
                        help='Host file that contains addresses of all workers.')
    args = parser.parse_args()
    BATCH_SIZE = int(args.batch_size)
    IN_CHANNEL = int(args.channel_size)
    NUM_FILTER = int(args.filter_size)
    NUM_LAYERS = int(args.num_layers)
    NUM_ITERATIONS = int(args.num_iterations)
    NUM_IGNORED_ITERATIONS = int(args.num_ignored_iterations)
    
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
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        worker_index = comm.Get_rank()

    n_workers = len(addresses)
    group2ctx = {'group:%d' % i : mx.cpu(0, addresses[i]) for i in range(n_workers)}
    default_ctx = mx.cpu(0, addresses[worker_index])
    # addresses = args.addresses.split(',')
    n_workers = len(addresses)
    group2ctx = {'group:%d' % i : mx.cpu(0, addresses[i]) for i in range(n_workers)}
    # worker_index = args.worker_index
    default_ctx = mx.cpu(0, addresses[worker_index])

    conv_test(default_ctx, group2ctx)

    #if not args.test_all:
    #    conv_test()
    #else:
    #    for b in [64, 128, 256, 512, 1024]:
    #        for c in [64, 128, 256, 512, 1024]:
    #            BATCH_SIZE = b
    #            IN_CHANNEL = c
    #            NUM_FILTER = c
    #            conv_test()

if __name__ == "__main__":
    main()
