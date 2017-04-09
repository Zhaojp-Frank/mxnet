export I_MPI_FABRICS=shm:dapl
export I_MPI_DAPL_PROVIDER=ofa-v2-ib0
export I_MPI_DYNAMIC_CONNECTION=0
export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/tofu/mxnet/python
export NNVM_EXEC_MATCH_RANGE=0
export OMP_NUM_THREADS=8
export TOFU_TILING_TYPE=kcuts
