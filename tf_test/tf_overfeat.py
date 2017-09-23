import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import time

from tensorflow.python.client import timeline
FLAGS = None

def overfeat(data, data_format, num_tasks=1):
    #1
    kernel = tf.Variable(tf.ones([7, 7, 3, 96], tf.float32))
    strides = [1, 1, 2, 2]
    net = tf.nn.conv2d(data, kernel, strides, padding='VALID', data_format=data_format)
    net = tf.nn.relu(net)
    print net.shape
    ksize=[1, 1, 3, 3]
    strides=[1, 1, 3, 3]
    net = tf.nn.max_pool(net, ksize=ksize, strides=strides, padding='SAME', data_format=data_format)
    print net.shape
    #2
    kernel = tf.Variable(tf.ones([7, 7, 96, 256], tf.float32))
    strides = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, kernel, strides, padding='VALID', data_format=data_format)
    net = tf.nn.relu(net)
    print net.shape
    ksize=[1, 1, 2, 2]
    strides=[1, 1, 2, 2]
    net = tf.nn.max_pool(net, ksize=ksize, strides=strides, padding='SAME', data_format=data_format)
    print net.shape
    # 3
    kernel = tf.Variable(tf.ones([3, 3, 256, 512], tf.float32))
    strides = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, kernel, strides, padding='SAME', data_format=data_format)
    net = tf.nn.relu(net)
    print net.shape
    # 4
    kernel = tf.Variable(tf.ones([3, 3, 512, 512], tf.float32))
    strides = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, kernel, strides, padding='SAME', data_format=data_format)
    net = tf.nn.relu(net)
    print net.shape
    # 5
    kernel = tf.Variable(tf.ones([3, 3, 512, 1024], tf.float32))
    strides = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, kernel, strides, padding='SAME', data_format=data_format)
    net = tf.nn.relu(net)
    print net.shape
    # 6
    kernel = tf.Variable(tf.ones([3, 3, 1024, 1024], tf.float32))
    strides = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, kernel, strides, padding='SAME', data_format=data_format)
    net = tf.nn.relu(net)
    print net.shape
    ksize=[1, 1, 3, 3]
    strides=[1, 1, 3, 3]
    net = tf.nn.max_pool(net, ksize=ksize, strides=strides, padding='SAME', data_format=data_format)
    print net.shape
    # 7
    net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
    if num_tasks > 1:
        kernels = []
        for i in range(num_tasks):
            kernels.append(tf.Variable(tf.ones([25600/num_tasks, 4096], tf.float32)))
        kernel1 = tf.concat(kernels, 0)
    else:
        kernel1 = tf.Variable(tf.ones([25600, 4096]), tf.float32)
    net = tf.matmul(net, kernel1)
    net = tf.nn.relu(net)
    print net.shape
    # 8
    if num_tasks > 1:
        kernels = []
        for i in range(num_tasks):
            kernels.append(tf.Variable(tf.ones([4096/num_tasks, 4096], tf.float32)))
        kernel1 = tf.concat(kernels, 0)
    else:
        kernel1 = tf.Variable(tf.ones([4096, 4096]), tf.float32)
    net = tf.matmul(net, kernel1)
    net = tf.nn.relu(net)
    print net.shape
    # 9
    kernel1 = tf.Variable(tf.ones([4096, 1024]), tf.float32)
    net = tf.matmul(net, kernel1)
    net = tf.nn.relu(net)

    prob = tf.nn.softmax(net)
    return prob


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
    cold_start = 10
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
            if data_format == 'NHWC':
                data = tf.placeholder(tf.float32, (batch , 221, 221, 3))
            else:
                data = tf.placeholder(tf.float32, (batch , 3, 221, 221))
            #loss = overfeat(data, data_format)
            loss = overfeat(data, data_format, FLAGS.num_tasks)
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
            x = np.random.rand(batch, 221, 221, 3)
        else:
            x = np.random.rand(batch, 3, 221, 221)
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
                if len(durations) >= 100:
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
