addresses=216.165.108.105:9000,216.165.108.104:9000
#,216.165.108.103:9000,216.165.108.102:9000
batch=8192
weight=8192
layers=1
iterations=105
ignored=5
export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]"
export MXNET_P2PNET_DEBUG=1
python tests/python/unittest/test_p2pnet_mlp.py --address=$addresses -b $batch -w $weight -l $layers -n $iterations -g $ignored -i $1 -e m2_l1_comm_$1_ > m2_l1_comm_$1
export MXNET_P2PNET_DEBUG=3
python tests/python/unittest/test_p2pnet_mlp.py --address=$addresses -b $batch -w $weight -l $layers -n $iterations -g $ignored -i $1 -e m2_l1_no_comm_$1 > m2_l1_no_comm_$1
export MXNET_P2PNET_DEBUG=1
python tests/python/unittest/test_p2pnet_mlp.py -s -b $batch -w $weight -l $layers -n $iterations -g $ignored -e m2_l1_single_$1 > m2_l1_single_$1
