This document assumes that you are using either **NYU NEWS cluster** or **Azure cluster**. Note that for NYU NEWS cluster only **geeker-1** to **geeker-4** are supported.

Install
-------
1. Clone the repository: `git clone https://github.com/fegin/mxnet`.
2. Switch to p2p\_net branch: `git checkout p2p_net`.
  - You should be already done the first two steps because you are reading this document :).
3. Replace `https://github.com/dmlc/nnvm` in **.gitmodules** with `https://github.com/fegin/nnvm`.
4. Update the repository: `git submodule update --recursive --init`.
5. Modify the following fields in make/config.mk:
  - `export CC = mpicc`
  - `export CXX = mpicxx`
  - `ADD_CFLAGS = -DP2PNET_MPI`
  - `USE_OPENCV = 0`
  - `MKLML_ROOT = /home/fegin/.local`
  - `USE_INTEL_PATH = /home/fegin/intel`
    - Above two paths will only work on NEWS cluster. Change them accordingly.
  - `USE_MKL2017 = 1`
  - `USE_BLAS = mkl`
  - `USE_DIST_KVSTORE = 1`
6. Build it! `make -j 8`

Execution
---------
1. Export the following environment variables:
  - `export MXNET_CPU_WORKER_NTHREADS=1`
  - `export NNVM_EXEC_MATCH_RANGE=0`
  - `export KMP_AFFINITY="explicit,granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]"`
  - `export LD_LIBRARY_PATH=/home/fegin/intel/lib/intel64:$LD_LIBRARY_PATH`
    - Above path will only work on NEWS cluster. Change it accordingly.
  - `export PYTHONPATH=/home/fegin/workspace/mxnet/python`
    - Please change this path to match your own working path.
2. Create a file called **host** and fill in all worker's IPs, one IP per line.
3. That's run a script to test if everything is done. 
```mpirun -outfile-pattern output_stdout -errfile-pattern output_stderr -l -np 2 -ppn 1 -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=1 -genv MXNET_P2PNET_DEBUG=1 -genv MXNET_P2PNET_HOST_PATH=/home/fegin/workspace/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=3 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[0,1,2,3,12,13,14,15]" python tofu_test_mlp.py -f host --batch_size=8```
  -  Remember to change **MXNET_P2PNET_HOST_PATH** accordingly.

Using ZeroMQ
------------
P2PNet version always use ZeroMQ as the communication method within a worker internally. As a result, the MPI version described in the previous sections can also be run without invoking mpirun. When doing so, the backend will choose ZeroMQ as the communication method accross workers.

### Install
If your environment supports Intel MPI, you only need to follow the **Install** section above. If not, you only need to change following fields in make/config.mk. Other steps remain the same.
  - `export CC = gcc`
  - `export CXX = gxx`
  - `ADD_CFLAGS = `

### Execution
TBD
