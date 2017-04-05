export NNVM_EXEC_MATCH_RANGE=0
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
#export MKL_NUM_THREADS=12
#export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22]"
export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,1,2,3,4,5,6,7,8,9,10,11]"

#set -x

#python tofu_test_mlp.py --address=$ADDRESS -i $1 --batch_size=4096 --hidden_size=4096

# Single
#python tofu_test_mlp.py -address=216.165.108.108:9000 --batch_size=$1 --hidden_size=$2 -i 0

# NoComm
NUM_PROCS=2

export MXNET_P2PNET_DEBUG=3
ADDRESS=127.0.0.1:9000
for i in `seq 1 $(($NUM_PROCS - 1))`;
do
  PORT=$((9000 + $i))
  ADDRESS="$ADDRESS,127.0.0.1:$PORT"
done
echo $ADDRESS
set -x
python tofu_test_conv.py --address=$ADDRESS -i 0 --batch_size=$1 --channel_size=$2 --num_layers=3
