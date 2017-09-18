import mxnet as mx

def add_args(parser):
  parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden layer size')
  parser.add_argument('--num_layers', type=int, default=5, help='Number of hidden layers')

# symbol net
def get_symbol(args):
  batch_size = args.batch_size
  hidden_size = args.hidden_size
  num_layers = args.num_layers
  print('Batch size: %d, Hidden size: %d' % (batch_size, hidden_size))
  net = mx.symbol.Variable('data')
  for i in range(num_layers):
    #weight = mx.symbol.Variable('fc%d_weight' % i)
    #net = mx.symbol.dot(net, weight)
    net = mx.symbol.FullyConnected(net, name='fc%d' % i, num_hidden=hidden_size, no_bias=True)
    net = mx.symbol.Activation(net, act_type='sigmoid')
  net = mx.symbol.SoftmaxOutput(net)
  return net, (hidden_size,), hidden_size
