NP=$1
AFFINITY=$2 
B=$3
H=$4
NCORES=$5
NOTDO_SINGLE=$6

HOST=/local/host.$NP
OPTIONS="--mca btl openib,self,sm -n $NP -pernode -hostfile $HOST --bind-to none"
OPTIONS=$OPTIONS" -x LD_LIBRARY_PATH=/opt/intel/lib/intel64 "
OPTIONS=$OPTIONS" -x PYTHONPATH=/local/mxnet/python "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_HOST_PATH=$HOST "
OPTIONS=$OPTIONS" -x KMP_AFFINITY=verbose,explicit,granularity=thread,proclist=[$AFFINITY] "
OPTIONS=$OPTIONS" -x OMP_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x MKL_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x MKL_DOMAIN_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x OMP_DYNAMIC=FALSE "
OPTIONS=$OPTIONS" -x MKL_DYNAMIC=FALSE "
OPTIONS=$OPTIONS" -x MXNET_CPU_PRIORITY_NTHREADS=1 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_ZMQ_IO_THREADS=0 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_MPI_POLLING_TIME=1 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_MAIN_THREAD_AFFINITY=7 "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_COMMUNICATION_METHOD=MPI "
OPTIONS=$OPTIONS" -x MXNET_P2PNET_INTERNAL_POLLING=1 "
OPTIONS=$OPTIONS" -x TOFU_NO_COMPUTATION=0 "
OPTIONS=$OPTIONS" -x TOFU_FAKE_VAR_SPLIT_CONCAT=1 "
OPTIONS=$OPTIONS" -x MXNET_CPU_TEMP_COPY=16 "
OPTIONS=$OPTIONS" -x NNVM_EXEC_MATCH_RANGE=0 "
OPTIONS=$OPTIONS" -x MXNET_EXEC_ENABLE_INPLACE=1 "
#OPTIONS=$OPTIONS" -x OMP_PROC_BIND=TRUE "
#OPTIONS=$OPTIONS" -x OMP_SCHEDULE=static "
#OPTIONS=$OPTIONS" -x OMP_NESTED=TRUE "
OPTIONS=$OPTIONS" -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so "
#OPTIONS=$OPTIONS" -x LD_PRELOAD=/usr/lib/libtcmalloc_minimal.so.4 "
CMD="python tofu_test_vgg.py --batch_size=${B} "

if [[ "$NOTDO_SINGLE" != '1' ]];
then
	echo "Doing $NP $B $H single"
	#mpirun $OPTIONS -output-filename log_vgg_single -x MXNET_P2PNET_DEBUG=0 /local/mxnet/env.sh
	mpirun $OPTIONS -output-filename log_vgg_single_${B} -x MXNET_P2PNET_DEBUG=0 $CMD --address=127.0.0.1 -i 0
fi

echo "Doing $NP $B $H without communication"
mpirun $OPTIONS -output-filename log_vgg_without_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=2 -x TOFU_TILING_TYPE=kcuts $CMD -f $HOST 

echo "Doing $NP $B $H with communication"
mpirun $OPTIONS -output-filename log_vgg_with_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=0 -x TOFU_TILING_TYPE=kcuts -x MXNET_P2PNET_USE_MPI_BARRIER=1 $CMD -f $HOST

echo "Doing $NP $B $H dp without communication"
mpirun $OPTIONS -output-filename log_vgg_dp_without_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=2 -x TOFU_TILING_TYPE=datapar $CMD -f $HOST

echo "Doing $NP $B $H dp with communication"
mpirun $OPTIONS -output-filename log_vgg_dp_with_comm_${NP}_${B}_${H} -x MXNET_P2PNET_DEBUG=0 -x TOFU_TILING_TYPE=datapar -x MXNET_P2PNET_USE_MPI_BARRIER=1 $CMD -f $HOST
