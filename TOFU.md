This document assumes that you are using either NYU NEWS cluster or Azure cluster. Note that for NYU NEWS cluster only geeker-1 to geeker-4 are supported.

Install
-------
1. git clone https://github.com/fegin/mxnet
2. git checkout p2p\_net (You should be already done these two steps because you are reaeaing me ;)).
3. Replace "https://github.com/dmlc/nnvm" in .gitmodules with "https://github.com/fegin/nnvm".
4. git submodule update --recursive --init
5. Modify make/config.mk (Please refer to /home/fegin/workspace/mxnet/make/config.mk on beakers).
6. make -j 8

Execution
---------
1. Export following environment variables.
export MXNET\_CPU\_WORKER\_NTHREADS=1
export NNVM\_EXEC\_MATCH\_RANGE=0
export LD\_LIBRARY\_PATH=/home/fegin/intel/lib/intel64:$LD\_LIBRARY\_PATH
export PYTHONPATH=/home/fegin/workspace/mxnet/python
export KMP\_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]"

2. Create a file called host and fill in all worker's IPs, one IP per line. For NYU NEWS cluster, only geeker-1 to geeker-4 are supported
3. mpirun -outfile-pattern output\_stdout -errfile-pattern output\_stderr -l -np 2 -ppn 1 -hostfile all\_ips -print-rank-map -genv MXNET\_P2PNET\_MPI\_SLEEP\_DURATION=1 -genv MXNET\_P2PNET\_DEBUG=1 -genv MXNET\_P2PNET\_HOST\_PATH=/home/fegin/workspace/mxnet/all\_ips -genv MXNET\_P2PNET\_MAIN\_AFFINITY=3 -genv MXNET\_P2PNET\_ZMQ\_IO\_THREADS=0 -genv KMP\_AFFINITY="explicit,granularity=fine,proclist=[0,1,2,3,12,13,14,15]" python tofu\_test\_mlp.py -f host --batch\_size=8
