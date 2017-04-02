ADDRESS=216.165.108.108:9000,216.165.108.109:9000

export NNVM_EXEC_MATCH_RANGE=0
export MKL_DYNAMIC=FALSE
#export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=12
#export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22]"

set -x

#python tofu_test_mlp.py --address=$ADDRESS -i $1 --batch_size=4096 --hidden_size=4096

# Single
#python tofu_test_mlp.py -address=216.165.108.108:9000 --batch_size=$1 --hidden_size=$2 -i 0

# NoComm cut2
#export MXNET_P2PNET_DEBUG=3
#python tofu_test_mlp.py --address=$ADDRESS -i 0 --batch_size=$1 --hidden_size=$2

# NoComm cut4
#export MXNET_P2PNET_DEBUG=3
#ADDRESS=216.165.108.108:9000,216.165.108.109:9000,216.165.108.109:9001,216.165.108.109:9002
#python tofu_test_mlp.py --address=$ADDRESS -i 0 --batch_size=$1 --hidden_size=$2

# NoComm cut8
#export MXNET_P2PNET_DEBUG=3
#ADDRESS=216.165.108.108:9000,216.165.108.109:9000,216.165.108.109:9001,216.165.108.109:9002,216.165.108.109:9003,216.165.108.109:9004,216.165.108.109:9005,216.165.108.109:9006
#python tofu_test_mlp.py --address=$ADDRESS -i 0 --batch_size=$1 --hidden_size=$2

# NoComm cut16
export MXNET_P2PNET_DEBUG=3
ADDRESS=216.165.108.108:9000,216.165.108.109:9000,216.165.108.109:9001,216.165.108.109:9002,216.165.108.109:9003,216.165.108.109:9004,216.165.108.109:9005,216.165.108.109:9006,216.165.108.108:9007,216.165.108.109:9008,216.165.108.109:9009,216.165.108.109:9010,216.165.108.109:9011,216.165.108.109:9012,216.165.108.109:9013,216.165.108.109:9014
python tofu_test_mlp.py --address=$ADDRESS -i 0 --batch_size=$1 --hidden_size=$2
