import mxnet as mx

def add_args(parser):
    parser.add_argument('--num_filter', type=int, default=1024, help='Resnet unit num_filters')
    parser.add_argument('--image_size', type=int, default=7, help='Image size')

def get_symbol(args, workspace=256):
    data = mx.sym.Variable('data')
    conv0 = mx.sym.Convolution(data=data, num_filter=int(args.num_filter),
            kernel=(1,1), stride=(1,1), pad=(0,0),
            no_bias=True, workspace=workspace, name='conv0')

    act1 = mx.sym.Activation(data=conv0, act_type='relu', name='relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(args.num_filter),#,int(args.num_filter*0.25),
            kernel=(1,1), stride=(1,1), pad=(0,0),
            no_bias=True, workspace=workspace, name='conv1')
    act2 = mx.sym.Activation(data=conv1, act_type='relu', name='relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(args.num_filter),#,int(args.num_filter*0.25),
            kernel=(3,3), stride=(1,1), pad=(1,1),
            no_bias=True, workspace=workspace, name='conv2')
    act3 = mx.sym.Activation(data=conv2, act_type='relu', name='relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=int(args.num_filter),
            kernel=(1,1), stride=(1,1), pad=(0,0),
            no_bias=True, workspace=workspace, name='conv3')

    #net = conv0 + conv3
    net = conv3

    #net = mx.sym.Flatten(net)
    #net = mx.sym.FullyConnected(net, name='fc', num_hidden=1000, no_bias=True)
    #net = mx.sym.SoftmaxOutput(net)
    return net, (args.num_filter, args.image_size, args.image_size), None
