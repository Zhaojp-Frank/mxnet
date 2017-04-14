#!/bin/bash
source ./mpi_env.sh

export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE

export MKL_DYNAMIC=FALSE
#export MKL_NUM_THREADS=8
#export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,1,2,3,4,5,6,7]"
export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,1,2,3]"

export NNVM_EXEC_MATCH_RANGE=0
export MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host
export MXNET_P2PNET_ZMQ_IO_THREADS=0
export MXNET_CPU_PRIORITY_NTHREADS=1
export MXNET_P2PNET_DEBUG=0
#export MXNET_P2PNET_MPI_SLEEP_DURATION=1
export MXNET_P2PNET_MAIN_AFFINITY=7
export MXNET_P2PNET_MPI_TEST_METHOD=1

NAME=$1  # name of the experiment
NP=$2    # number of processes to run
BATCH_SIZE=$3
HIDDEN_SIZE=$4

LOG_ROOT=/home/tofu/mxnet/log_minjie/

set -e
set -x

export LOG_DIR=$LOG_ROOT/${NAME}_`date "+%j-%H-%M-%S"`
export MASTER="worker-0"

mkdir -p $LOG_DIR
exec &> >(tee "$LOG_DIR/script.out")
env > $LOG_DIR/env.txt

if [ $NP == 1 ]; then
  # Single machine
  python tofu_test_mlp.py --address=127.0.0.1:9000 --batch_size=$BATCH_SIZE --hidden_size=$HIDDEN_SIZE -i 0 2>&1 | tee $LOG_DIR/single.log
else
  # Multi machines
  export I_MPI_DEBUG_OUTPUT=$LOG_DIR/mpi_debug
  mpirun -np $NP -envall -f host -ppn 1 \
         ./mpi_wrapper.sh python tofu_test_mlp.py -f host --batch_size=$BATCH_SIZE --hidden_size=$HIDDEN_SIZE
#--num_layers=50
fi
