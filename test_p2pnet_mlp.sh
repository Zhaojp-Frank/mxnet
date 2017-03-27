batch=8192
weight=8192
layers=5
iterations=20
ignored=5
export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]"
export NNVM_EXEC_MATCH_RANGE=0
export MXNET_P2PNET_DEBUG=1
python tests/python/unittest/test_p2pnet_mlp.py -f all_private_ips -b $batch -w $weight -l $layers -n $iterations -g $ignored -i $1 -e l${layers}_comm_$1 | tee l${layers}_comm_$1
#export MXNET_P2PNET_DEBUG=3
#python tests/python/unittest/test_p2pnet_mlp.py -f all_private_ips -b $batch -w $weight -l $layers -n $iterations -g $ignored -i $1 -e l${layers}_no_comm_$1 | tee l${layers}_no_comm_$1
#export MXNET_P2PNET_DEBUG=1
#python tests/python/unittest/test_p2pnet_mlp.py -s -b $batch -w $weight -l $layers -n $iterations -g $ignored -e l${layers}_single_$1 | tee l${layers}_single_$1
