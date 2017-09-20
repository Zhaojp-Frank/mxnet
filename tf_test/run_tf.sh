NP=$1
AFFINITY=$2 
NCORES=$3

HOST=/local/host.$NP
OPTIONS="--mca btl openib,self,sm -n $(( NP * 2)) -npernode 2 -hostfile $HOST --bind-to none"
OPTIONS=$OPTIONS" --display-map "
OPTIONS=$OPTIONS" -x LD_LIBRARY_PATH=/opt/intel/lib/intel64 "
OPTIONS=$OPTIONS" -x KMP_AFFINITY=verbose,explicit,granularity=thread,proclist=[$AFFINITY] "
OPTIONS=$OPTIONS" -x OMP_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x MKL_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x MKL_DOMAIN_NUM_THREADS=$NCORES "
OPTIONS=$OPTIONS" -x OMP_DYNAMIC=FALSE "
OPTIONS=$OPTIONS" -x MKL_DYNAMIC=FALSE "

CMD="python tf_alexnet.py --host_file=$HOST"

mpirun $OPTIONS -output-filename tf_alexnet_comm_${NP} $CMD
