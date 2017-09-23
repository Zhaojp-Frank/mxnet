NP=$1
AFFINITY=$2 
NCORES=$3
BATCH_SIZE=$4
WEIGHT_SIZE=$5

HOST=/local/host.$NP
OPTIONS="--mca btl openib,self,sm -n $(( NP * 2)) -npernode 2 -hostfile $HOST --bind-to none"
OPTIONS=$OPTIONS" --display-map "
#OPTIONS=$OPTIONS" -x LD_LIBRARY_PATH=/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64 "
#OPTIONS=$OPTIONS" -x KMP_AFFINITY=verbose,explicit,granularity=thread,proclist=[$AFFINITY] "
OPTIONS=$OPTIONS" -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 "
OPTIONS=$OPTIONS" -x OMP_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x MKL_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x MKL_DOMAIN_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x OMP_DYNAMIC=FALSE "
OPTIONS=$OPTIONS" -x MKL_DYNAMIC=FALSE "
OPTIONS=$OPTIONS" -x KMP_SETTINGS=TRUE "
OPTIONS=$OPTIONS" -x KMP_BLOCKTIME=1 "
OPTIONS=$OPTIONS" -x MPI_OPTIMAL_PATH=1 "

CMD="python tf_mlp.py --host_file=$HOST --batch_size=$BATCH_SIZE --WEIGHT_SIZE=$WEIGHT_SIZE"

mpirun $OPTIONS -output-filename tf_mlp_comm_${NP} $CMD
