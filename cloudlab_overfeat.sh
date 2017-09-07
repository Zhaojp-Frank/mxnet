NP=$1
AFFINITY=$2 
B=$3
H=$4

HOST=/local/host.$NP
OPTIONS="--mca btl openib,self,sm -n $NP -pernode -hostfile $HOST --bind-to none"
OPTIONS=$OPTIONS" -x LD_LIBRARY_PATH=/opt/intel/lib/intel64 "
OPTIONS=$OPTIONS" -x PYTHONPATH=/local/mxnet/python "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_HOST_PATH=$HOST "
OPTIONS=$OPTIONS" -x KMP_AFFINITY=\"explicit,granularity=fine,proclist=[$AFFINITY]\" "
OPTIONS=$OPTIONS" -x OMP_NUM_THREADS=14 "
#OPTIONS=$OPTIONS" -x MKL_NUM_THREADS=8 "
#OPTIONS=$OPTIONS" -x MKL_DOMAIN_NUM_THREADS=16 "
OPTIONS=$OPTIONS" -x OMP_DYNAMIC=FALSE "
OPTIONS=$OPTIONS" -x MKL_DYNAMIC=FALSE "
#OPTIONS=$OPTIONS" -x MXNET_CPU_PRIORITY_NTHREADS=1 "
#OPTIONS=$OPTIONS" -x MXNET_P2PNET_ZMQ_IO_THREADS=0 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_MPI_POLLING_TIME=0 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_MAIN_THREAD_AFFINITY=15 "
#OPTIONS=$OPTIONS" -x MXNET_P2PNET_MPI_TEST_METHOD=1 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_COMMUNICATION_METHOD=MPI "
OPTIONS=$OPTIONS" -x NNVM_EXEC_MATCH_RANGE=0 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_INTERNAL_POLLING=1 "
OPTIONS=$OPTIONS" -x TOFU_NO_COMPUTATION=0 "
OPTIONS=$OPTIONS" -x TOFU_FAKE_VAR_SPLIT_CONCAT=1 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_USE_MPI_BARRIER=1 "
CMD="python tofu_test_overfeat.py --batch_size=${B} "

echo "Doing $NP $B $H single"
#mpirun $OPTIONS -output-filename log_overfeat_single -x MXNET_P2PNET_DEBUG=0 /local/mxnet/env.sh
mpirun $OPTIONS -output-filename log_overfeat_single -x MXNET_P2PNET_DEBUG=0 $CMD --address=127.0.0.1 -i 0

echo "Doing $NP $B $H without communication"
mpirun $OPTIONS -output-filename log_overfeat_without_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=2 -x TOFU_TILING_TYPE=kcuts $CMD -f $HOST 

echo "Doing $NP $B $H with communication"
mpirun $OPTIONS -output-filename log_overfeat_with_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=0 -x TOFU_TILING_TYPE=kcuts $CMD -f $HOST

echo "Doing $NP $B $H dp without communication"
mpirun $OPTIONS -output-filename log_overfeat_dp_without_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=2 -x TOFU_TILING_TYPE=datapar $CMD -f $HOST

echo "Doing $NP $B $H dp with communication"
mpirun $OPTIONS -output-filename log_overfeat_dp_with_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=0 -x TOFU_TILING_TYPE=datapar $CMD -f $HOST
