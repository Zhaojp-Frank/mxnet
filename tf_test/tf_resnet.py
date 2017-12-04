import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import time

from tensorflow.python.client import timeline
FLAGS = None

has_activation = True
def Activation(data, **kwargs):
    if has_activation:
        return tf.nn.relu(data)
    else:
        return data

expand_shortcut = False
def Expand(data, num_skips):
    if expand_shortcut:
        for i in range(num_skips):
            data = tf.identity(data)
        return data
    else:
        return data

DF = 'NCHW'
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
        act1 = Activation(data=data)
        conv1 = tf.nn.conv2d(act1, tf.Variable(tf.ones([1, 1, act1.shape[1], int(num_filter*0.25)], tf.float32)), [1, 1, 1, 1], padding='SAME', data_format=DF)
        print 3, conv1.shape
        act2 = Activation(data=conv1)
        conv2 = tf.nn.conv2d(act2, tf.Variable(tf.ones([3, 3, act2.shape[1], int(num_filter*0.25)], tf.float32)), (1, 1) + stride, padding='SAME', data_format=DF)
        print 4, conv2.shape
        act3 = Activation(data=conv2)
        conv3 = tf.nn.conv2d(act3, tf.Variable(tf.ones([1, 1, act2.shape[1], num_filter], tf.float32)), [1, 1, 1, 1], padding='SAME', data_format=DF)
        print 5, conv3.shape
        if dim_match:
            shortcut = Expand(data, num_skips=6)
        else:
            conv4 = tf.nn.conv2d(act1, tf.Variable(tf.ones([1, 1, act1.shape[1], num_filter], tf.float32)), (1, 1) + stride, padding='SAME', data_format=DF)
            print 6, conv4.shape
            shortcut = Expand(conv4, num_skips=4)
        if memonger:
            assert False
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        assert False
        act1 = Activation(data=data)
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        act2 = Activation(data=conv1)
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = Expand(data, num_skips=4)
        else:
            shortcut = Expand(mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc'), num_skips=2)
        if memonger:
            assert False
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(batch_size, units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
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
    data = tf.placeholder(tf.float32, (batch_size,) + image_shape)
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        assert False
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = tf.nn.conv2d(data, tf.Variable(tf.ones([7, 7, 3, filter_list[0]], tf.float32)), [1, 1, 2, 2], padding='SAME', data_format=DF)
        print 1, body.shape
        body = Activation(data=body)
        body = tf.nn.max_pool(body, ksize=[1, 1, 3, 3], strides=[1, 1, 2, 2], padding='SAME', data_format=DF)
        print 2, body.shape

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    relu1 = Activation(data=body)
    # Although kernel is not used here when global_pool=True, we should put one
    print 97, relu1.shape
    pool1 = tf.nn.avg_pool(relu1, ksize=[1, 1, 7, 7], strides=[1, 1, 1, 1], padding='VALID', data_format=DF)
    print 98, pool1.shape
    flat = tf.reshape(pool1, [-1, int(np.prod(pool1.get_shape()[1:]))])
    print 99, pool1.shape
    kernel = tf.Variable(tf.ones([2048, num_classes], tf.float32))
    fc1 = tf.matmul(flat, kernel)
    prob = tf.nn.softmax(fc1)
    return data, prob

def get_symbol(batch_size, num_layers=50, conv_workspace=256):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = (3, 224, 224)
    num_classes = 1024
    num_layers = 50
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
            #filter_list = [i * 2 for i in filter_list]
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

    return resnet(batch_size  = batch_size,
                  units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace), (3, 224, 224), 1024


def get_mpi_hosts(FLAGS):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    task_index = comm.Get_rank() / 2
    job_name = "ps" if comm.Get_rank() % 2 == 0 else "worker"
    if job_name == "ps":
        os.environ["KMP_AFFINITY"]="verbose,explicit,granularity=thread,proclist=[4,5,6,7] "
    num_tasks = comm.Get_size() / 2
    ps_hosts = ""
    worker_hosts = ""
    with open(FLAGS.host_file) as fp:
        for line in fp:
            line = line.strip()
            ps_hosts += line + ":5555,"
            worker_hosts += line + ":6666,"

    FLAGS.task_index = task_index
    FLAGS.job_name = job_name 
    FLAGS.ps_hosts = ps_hosts[:-1]
    FLAGS.worker_hosts = worker_hosts[:-1]
    FLAGS.num_tasks = num_tasks


def main(_):
    data_format='NCHW'
    cold_start = 5
    get_mpi_hosts(FLAGS)
    loops = 300 * FLAGS.num_tasks
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    print(os.environ)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index, protocol='grpc+verbs')

    print ("Server type is %s" % FLAGS.job_name)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        batch = 128 / FLAGS.num_tasks
        print("Batch size is %d" % batch)
        # Assigns ops to the local worker by default.
        def _load_fn(unused_op):
            return 1
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)): #, ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(FLAGS.num_tasks, tf.contrib.training.byte_size_load_fn))):

            # Build model...
            (data, loss), _, _ = get_symbol(batch)
            global_step = tf.contrib.framework.get_or_create_global_step()

            optimizer = tf.train.GradientDescentOptimizer(0.01)
            #optimizer = tf.train.SyncReplicasOptimizer(
                           #optimizer, replicas_to_aggregate=FLAGS.num_tasks,
                           #total_num_replicas=FLAGS.num_tasks)
            #train_op = optimizer.minimize(loss, global_step = global_step)
            train_op = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(train_op, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=loops),]
               #optimizer.make_session_run_hook(FLAGS.task_index == 0)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        if data_format == 'NHWC':
            x = np.random.rand(batch, 224, 224, 3)
        else:
            x = np.random.rand(batch, 3, 224, 224)
        durations = []
        ops = tf.GraphOptions(build_cost_model=0)
        config = tf.ConfigProto(graph_options=ops)
        config.placement_period = 100000
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 1
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               config=config,
                                               save_checkpoint_secs=None,
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                begin = time.time()
                mon_sess.run(train_op, feed_dict={data: x})
                end = time.time()
                duration = end - begin
                print "Duration: ", ((len(durations) + 1), loops, duration)
                durations.append(duration)
                #if len(durations) == 5:
                   #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                   #with open('timeline.ctf.json', 'w') as fp:
                       #fp.write(trace.generate_chrome_trace_format())
                if len(durations) >= 10:
                   print("average : %f " % (sum(durations[cold_start:]) / (len(durations) - cold_start)))
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
            "--host_file",
            type=str,
            default="",
            help="The path to the host file.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
