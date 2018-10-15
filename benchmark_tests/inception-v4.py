import mxnet as mx



def add_args(parser):
  return parser



def Conv(data, num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), name=None, suffix=''):

  conv_data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
    no_bias=True, name='%s%s_conv'%(name,suffix))
  #bn = mx.sym.BatchNorm(data=conv_data, name='%s%s_batchnorm'%(name,suffix), fix_gamma=True)
  act = mx.sym.Activation(data=conv_data, act_type='relu', name='%s%s_relu'%(name,suffix))

  return act



def Inception_stem(data, name=None):
  c = Conv(data, num_filter=32, kernel=(3,3), stride=(2,2), name='%s_conv1_3*3'%name)
  c = Conv(c, num_filter=32, kernel=(3,3), name='%s_conv2_3*3'%name)
  c = Conv(c, num_filter=64, kernel=(3,3), pad=(1,1), name='%s_conv3_3*3'%name)

  pooling = mx.sym.Pooling(c, kernel=(3,3), stride=(2,2), pool_type='max', name='%s_pool_1'%name)
  c1 = Conv(c, num_filter=96, kernel=(3,3), stride=(2,2), name='%s_conv4_3*3'%name)
  concat = mx.sym.Concat(*[pooling, c1], name='%s_concat_1'%name)

  c2 = Conv(concat, num_filter=64, kernel=(1,1), name='%s_conv5_1*1'%name)
  c2 = Conv(c2, num_filter=96, kernel=(3,3), name='%s_conv6_3*3'%name)

  c1 = Conv(concat, num_filter=64, kernel=(1, 1), name='%s_conv7_1*1' %name)
  c1 = Conv(c1, num_filter=64, kernel=(7,1), pad=(3,0), name='%s_conv8_7*1'%name)
  c1 = Conv(c1, num_filter=64, kernel=(1,7), pad=(0,3), name='%s_conv9_1*7'%name)
  c1 = Conv(c1, num_filter=96, kernel=(3,3), name='%s_conv10_3*3'%name)

  concat = mx.sym.Concat(*[c2, c1], name='%s_concat_2'%name)

  c2 = Conv(concat, num_filter=192, kernel=(3,3), stride=(2,2), name='%s_conv11_3*3'%name)
  pooling = mx.sym.Pooling(concat, kernel=(3,3), stride=(2,2), pool_type='max', name='%s_pooling_2'%name)
  concat = mx.sym.Concat(*[c2, pooling], name='%s_concat_3'%name)

  return concat



def InceptionA(data, name=None):
  pool = mx.sym.Pooling(data, kernel=(3,3), pad=(1,1), pool_type='avg', name='%s_avgpool_1'%name)
  c1 = Conv(pool, 96, name='%s_conv1_1*1'%name)
  c2 = Conv(data, 96, name='%s_conv2_1*1'%name)
  c3 = Conv(data, 64, name='%s_conv3_1*1'%name)
  c4 = Conv(data, 64, name='%s_conv4_1*1'%name)

  c3 = Conv(c3, 96, kernel=(3,3), pad=(1,1), name='%s_conv5_3*3'%name)
  c4 = Conv(c4, 96, kernel=(3,3), pad=(1,1), name='%s_conv6_3*3'%name)
  c4 = Conv(c4, 96, kernel=(3,3), pad=(1,1), name='%s_conv7_3*3'%name)

  concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1'%name)

  return concat



def InceptionB(data, name=None):
  pool = mx.sym.Pooling(data, kernel=(3,3), pad=(1,1), pool_type='avg', name='%s_avgpool_1'%name)
  c1 = Conv(pool, 128, name='%s_conv1_1*1'%name)
  c2 = Conv(data, 384, name='%s_conv2_1*1'%name)
  c3 = Conv(data, 192, name='%s_conv3_1*1'%name)

  # c4 can use initial c3
  c4 = Conv(c3, 192, kernel=(1,7), pad=(0,3), name='%s_conv_4_1*7'%name)
  c3 = Conv(c3, 224, kernel=(1,7), pad=(0,3), name='%s_conv_5_1*7'%name)
  c3 = Conv(c3, 256, kernel=(7,1), pad=(3,0), name='%s_conv_6_7*1'%name)

  c4 = Conv(c4, 224, kernel=(7,1), pad=(3,0), name='%s_conv_7_7*1'%name)
  c4 = Conv(c4, 224, kernel=(1,7), pad=(0,3), name='%s_conv_8_1*7'%name)
  c4 = Conv(c4, 256, kernel=(7,1), pad=(3,0), name='%s_conv_9_7*1'%name)

  concat= mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1'%name)

  return concat



def InceptionC(data, name=None):
  pool = mx.sym.Pooling(data, kernel=(3,3), pad=(1,1), pool_type='avg', name='%s_avgpool_1'%name)
  c1 = Conv(pool, 256, name='%s_conv1_1*1'%name)
  c2 = Conv(data, 256, name='%s_conv2_1*1' %name)
  c3 = Conv(data, 384, name='%s_conv3_1*1' %name)

  c3_1 = Conv(c3, 256, kernel=(1, 3), pad=(0, 1), name='%s_conv4_3*1' %name)
  c3_2 = Conv(c3, 256, kernel=(3, 1), pad=(1, 0), name='%s_conv5_1*3' %name)

  c4 = Conv(data, 384, name='%s_conv6_1*1' %name)
  c4 = Conv(c4, 448, kernel=(1, 3), pad=(0, 1), name='%s_conv7_1*3' %name)
  c4 = Conv(c4, 512, kernel=(3, 1), pad=(1, 0), name='%s_conv8_3*1' %name)
  c4_1 = Conv(c4, 256, kernel=(3, 1), pad=(1, 0), name='%s_conv9_1*3' %name)
  c4_2 = Conv(c4, 256, kernel=(1, 3), pad=(0, 1), name='%s_conv10_3*1' %name)

  concat = mx.sym.Concat(*[c1, c2, c3_1, c3_2, c4_1, c4_2], name='%s_concat'%name)

  return concat
  


def ReductionA(data, name=None):
  pool = mx.sym.Pooling(data, kernel=(3,3), stride=(2,2), pool_type='max', name='%s_maxpool_1'%name)
  c1 = Conv(data, 384, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' %name)

  c2 = Conv(data, 192, name='%s_conv2_1*1' %name)
  c2 = Conv(c2, 224, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' %name)
  c2 = Conv(c2, 256, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name='%s_conv4_3*3' %name)

  concat = mx.sym.concat(*[pool, c1, c2], name='%s_concat_1'%name)

  return concat



def ReductionB(data, name=None):
  pool = mx.sym.Pooling(data, kernel=(3,3), stride=(2,2), pool_type='max', name='%s_maxpool_1'%name)
  c1 = Conv(data, 192, name='%s_conv1_1*1' %name)
  c1 = Conv(c1, 192, kernel=(3, 3), stride=(2, 2), name='%s_conv2_3*3' %name)

  c2 = Conv(data, 256, name='%s_conv3_1*1' %name)
  c2 = Conv(c2, 256, kernel=(1, 7), pad=(0, 3), name='%s_conv4_1*7' %name)
  c2 = Conv(c2, 320, kernel=(7, 1), pad=(3, 0), name='%s_conv5_7*1' %name)
  c2 = Conv(c2, 320, kernel=(3, 3), stride=(2, 2), name='%s_conv6_3*3' %name)

  concat = mx.sym.concat(*[pool, c1, c2], name='%s_concat_1'%name)

  return concat



def inception_v4(image_shape=(3,299,299), num_classes=1000):
  data = mx.sym.Variable(name='data')
  r = Inception_stem(data, name='stem')

  for i in range(4):
    r = InceptionA(r, name='inceptionA%d'%(i+1))
  r = ReductionA(r, name='reductionA')

  for i in range(7):
    r = InceptionB(r, name='inceptionB%d'%(i+1))
  r = ReductionB(r, name='reductionB')

  for i in range(3):
    r = InceptionC(r, name='inceptionC%d'%(i+1))

  r = mx.sym.Pooling(r, kernel=(8,8), pad=(1,1), pool_type='avg', name='global_vagpool')
  r = mx.sym.Dropout(r, p=0.2)
  flatten = mx.sym.Flatten(r, name='flatten')
  fc = mx.sym.FullyConnected(flatten, num_hidden=num_classes, name='fc')
  softmax = mx.sym.SoftmaxOutput(fc, name='softmax')

  return softmax



def get_symbol(args):
  image_shape = (3, 299, 299)
  num_classes = 1000
  net = inception_v4(image_shape, num_classes)
  return net, image_shape, num_classes




