export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22]"
export NNVM_EXEC_MATCH_RANGE=0
python test_dot.py -m 4096 -n 4096 -k 4096
