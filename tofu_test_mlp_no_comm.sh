export NNVM_EXEC_MATCH_RANGE=0
export MXNET_P2PNET_DEBUG=3
export PYTHONPATH=$PWD/python

NUM_PROCS=2

ADDRESS=127.0.0.1:9000
for i in `seq 1 $(($NUM_PROCS - 1))`;
do
  PORT=$((9000 + $i))
  ADDRESS="$ADDRESS,127.0.0.1:$PORT"
done
echo $ADDRESS
set -x
python tofu_test_mlp.py --address=$ADDRESS -i 0 --batch_size=$1 --hidden_size=$2
