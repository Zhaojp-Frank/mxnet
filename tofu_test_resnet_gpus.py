# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 30
cold_skip = 5

rng = np.random.RandomState(seed=32)

has_activation = True

def Activation(data, **kwargs):
    if has_activation:
        return mx.sym.Activation(data=data, **kwargs)
    else:
        return data

expand_shortcut = True
def Expand(data, num_skips):
    if expand_shortcut:
        for i in range(num_skips):
            data = mx.sym.identity(data=data)
        return data
    else:
        return data

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        act1 = Activation(data=data, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        act2 = Activation(data=conv1, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        act3 = Activation(data=conv2, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = Expand(data, num_skips=6)
        else:
            shortcut = Expand(mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc'), num_skips=4)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        act1 = Activation(data=data, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        act2 = Activation(data=conv1, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = Expand(data, num_skips=4)
        else:
            shortcut = Expand(mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc'), num_skips=2)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    relu1 = Activation(data=body, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1', no_bias=True)
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)

def feed_args(net, arg_arrays):
    names = net.list_arguments()
    for name, arr in zip(names, arg_arrays):
        if not name.endswith('label'):
            # create random data
            arr[:] = 0.1 * rng.randn(*(arr.shape))

def test_mlp():
    has_mpi = False
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("Tofu MLP test code")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--num_layers', type=int, default=50, help='Number of hidden layers')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('-f', '--host_file', type=str,
                        help='Host file that contains addresses of all workers.')

    args = parser.parse_args()
    group2ctx = {'group:%d' % i : mx.gpu(i) for i in range(args.num_gpus)}
    if args.num_gpus == 1:
      default_ctx = mx.gpu(0)
    else:
      default_ctx = mx.cpu(0)

    image_shape = (3, 224, 224)
    num_classes = 1000
    net = get_symbol(num_classes, args.num_layers, image_shape)

    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    in_shapes = {}
    #in_shapes = {'fc%d_weight' % i: (args.hidden_size, args.hidden_size) for i in range(args.num_layers)}
    in_shapes['data'] = (args.batch_size, ) + image_shape
    in_types = {}
    #in_types = {'fc%d_weight' % i : mx.base.mx_real_t for i in range(args.num_layers)}
    in_types['data'] = mx.base.mx_real_t
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**in_shapes)
    arg_types, out_types, aux_types = net.infer_type(**in_types)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, default_ctx, dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))
    # create gradient ndarray for all parameters.
    args_grad = {name : mx.nd.zeros(shape, default_ctx, dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 if name != 'data' and not name.endswith('label')}
    print('Argument grads: ', args_grad.keys())

    executor = net.bind(ctx=default_ctx,
                        args=arg_arrays,
                        args_grad=args_grad,
                        grad_req='write',
                        group2ctx=group2ctx)
    return

    feed_args(net, arg_arrays)
    all_time = []
    for i in range(num_loops):
        print('=> loop %d' % i);
        st_l = time.time()
        if i == cold_skip:
            t0 = time.time()
        outputs = executor.forward()
        executor.backward()
        for name, grad in args_grad.items():
            #print(name, grad.asnumpy())
            grad.wait_to_read()
        # XXX(minjie): Currently, the last output is used to synchronize all send nodes.
        # Send nodes may not appear on the dependency path of the local graph.
        # We need make sure all send nodes have finished before the end of the iteration.
        if len(outputs) > 0:
            outputs[-1].wait_to_read()
        ed_l = time.time()
        print('=> loop duration %f' % float(ed_l - st_l))
        if (i >= cold_skip):
             all_time.append(float(ed_l - st_l))
    t1 = time.time()

    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))
    print('std : %f' % np.asarray(all_time).std())


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_mlp()
