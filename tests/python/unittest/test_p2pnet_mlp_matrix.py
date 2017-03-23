# pylint: skip-file
from __future__ import print_function
import argparse
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

BATCH_SIZE = 1024
WEIGHT_SIZE = 1024
NUM_LAYERS = 5
NUM_ITERATIONS = 2
NUM_IGNORED_ITERATIONS = 1
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

    def Histogram(self):
        min = int(math.floor(self.exps[-1] * 10))
        max = int(math.ceil(self.exps[0] * 10))
        interval = (max - min) / 10.0
        hist = []
        for t in self.exps:
            if interval > 0:
                idx = int((t * 10 - min) / interval)
            else:
                idx = 0
            hist.append((min + idx * interval) / 10.0)
        plt.hist(hist)
        plt.title(self.name)
        plt.xlabel("Seconds")
        plt.ylabel("Frequency")
        plt.savefig("%s.png" % self.name)

    def Summary(self):
        assert len(self.exps) == (self.iterations - self.ignored_iterations), len(self.exps)
        self.exps.sort()
        self.Histogram()
        exps = np.asarray(self.exps)
        avg = exps.mean()
        std = exps.std()
        line = 'avg: %f, std: %f\n' % (avg, std)
        print(line)
        with open('%s.txt' % self.name, 'w') as fp:
            fp.write(line)


def SingleMKL(num_cut, with_add):
    weight_shape = (WEIGHT_SIZE / num_cut, WEIGHT_SIZE)
    arg_arrays = {}
    activations = [None for i in range(num_cut)]
    in_data = [None for i in range(num_cut)]
    out_data = None if with_add else [None for i in range(num_cut)]
    for i in range(num_cut):
        in_data[i] = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE / num_cut), dtype=np.float32)
    for i in range(NUM_LAYERS):
        for j in range(num_cut):
            var_name = 'w_%d_%d' % (i, j)
            activations[j] = mx.symbol.dot(in_data[j], mx.symbol.Variable(var_name,
                                                                shape=weight_shape,
                                                                dtype=np.float32))
            arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)
        if with_add:
            out_data = mx.symbol.ElementWiseSum(*activations)
        else:
            for k in range(num_cut):
                out_data[k] = activations[k]

    net = mx.symbol.Group(out_data)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape()
    arg_types, out_types, aux_types = net.infer_type()
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)
    print(out_shapes)

    exp = Experiment(NUM_ITERATIONS, NUM_IGNORED_ITERATIONS, EXPERIMENT_NAME)
    for i in range(NUM_ITERATIONS):
        with exp:
            output = executor.forward()
            for out in output:
                out.wait_to_read()
        print("=" * 30)
        print("Finish an iteration %d" % i)
    exp.Summary()


def main():
    global BATCH_SIZE
    global WEIGHT_SIZE
    global NUM_LAYERS
    global NUM_ITERATIONS
    global NUM_IGNORED_ITERATIONS
    global EXPERIMENT_NAME
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
    parser.add_argument('-e', '--experiment_name', type=str,
                        help='The name of this experiment.',
                        default='Experiment')
    args = parser.parse_args()
    BATCH_SIZE = int(args.batch_size)
    WEIGHT_SIZE = int(args.weight_size)
    NUM_LAYERS = int(args.num_layers)
    NUM_ITERATIONS = int(args.num_iterations)
    NUM_IGNORED_ITERATIONS = int(args.num_ignored_iterations)
    EXPERIMENT_NAME = args.experiment_name
    if args.single_machine:
        print("No split")
        SingleMKL(1, False)
        for i in [2, 4, 8]:
            print("Split by %d with add" % i)
            SingleMKL(i, True)
            print("Split by %d without add" % i)
            SingleMKL(i, False)
    else:
        pass


if __name__ == "__main__":
    main()
