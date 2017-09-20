import argparse
import sys

import tensorflow as tf
import numpy as np
import time

FLAGS = None

def alexnet(data):
    kernel = tf.Variable(tf.ones([11, 11, 3, 64], tf.float32))
    net = tf.nn.conv2d(data, kernel, [1, 4, 4, 1], padding='VALID')
    print kernel.shape
    print net.shape
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    print net.shape
    kernel = tf.Variable(tf.ones([5, 5, 64, 192], tf.float32))
    net = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    kernel = tf.Variable(tf.ones([3, 3, 192, 384], tf.float32))
    net = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)
    kernel = tf.Variable(tf.ones([3, 3, 384, 256], tf.float32))
    net = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)
    kernel = tf.Variable(tf.ones([3, 3, 256, 256], tf.float32))
    net = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    print net.shape, net.dtype, int(np.prod(net.get_shape()[1:]))
    net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
    net = tf.nn.relu_layer(net, tf.Variable(tf.ones([6400, 4096]), tf.float32),
                                            tf.Variable(tf.ones([4096,]), tf.float32))
    net = tf.nn.relu_layer(net, tf.Variable(tf.ones([4096, 4096]), tf.float32),
                                            tf.Variable(tf.ones([4096,]), tf.float32))
    net = tf.nn.relu_layer(net, tf.Variable(tf.ones([4096, 1024]), tf.float32),
                                            tf.Variable(tf.ones([1024,]), tf.float32))
    prob = tf.nn.softmax(net)
    return prob


def get_mpi_hosts(FLAGS):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    task_index = comm.Get_rank() / 2
    job_name = "ps" if comm.Get_rank() % 2 == 0 else "worker"
    num_tasks = comm.Get_size()
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


def main(_):
    get_mpi_hosts(FLAGS)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index, protocol='grpc+mpi')

    print ("Server type is %s" % FLAGS.job_name)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            data = tf.placeholder(tf.float32, (128, 224, 224, 3))
            loss = alexnet(data)
            global_step = tf.contrib.framework.get_or_create_global_step()

            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
                    loss, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=15)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        x = np.random.rand(128, 224, 224, 3)
        with tf.train.MonitoredTrainingSession(master=server.target, 
                                               is_chief=(FLAGS.task_index == 0), 
                                               #checkpoint_dir="/tmp/train_logs", 
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
                print "Duration: ", duration


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
