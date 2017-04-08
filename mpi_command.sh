NP=$1
AFFINITY=$2 
# 5 layers matrix multiplication single machines
mpirun -outfile-pattern log_stdout_mm_single -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=5 -genv MXNET_P2PNET_DEBUG=0 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python /home/tofu/mxnet/tests/python/unittest/test_p2pnet_mlp.py -b 8192 -w 8192 -l 5 -n 55 -g 5 -s

# 5 layers matrix multiplication without communication
mpirun -outfile-pattern log_stdout_mm_without_comm -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=30000 -genv MXNET_P2PNET_DEBUG=2 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python /home/tofu/mxnet/tests/python/unittest/test_p2pnet_mlp.py -f host -b 8192 -w 8192 -l 5 -n 55 -g 5

# 5 layers matrix multiplication with communication
mpirun -outfile-pattern log_stdout_mm_with_comm -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=5 -genv MXNET_P2PNET_DEBUG=0 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python /home/tofu/mxnet/tests/python/unittest/test_p2pnet_mlp.py -f host -b 8192 -w 8192 -l 5 -n 55 -g 5

# 5 layers MLP single machine
mpirun -outfile-pattern log_stdout_mlp_single -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=5 -genv MXNET_P2PNET_DEBUG=0 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python tofu_test_mlp.py --batch_size=8192 --hidden_size=8192 --num_layers=5 --addresses=127.0.0.1 -i 0

# 5 layers MLP without communication
mpirun -outfile-pattern log_stdout_mlp_without_comm -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=30000 -genv MXNET_P2PNET_DEBUG=2 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python tofu_test_mlp.py -f host --batch_size=8192 --hidden_size=8192 --num_layers=5

# 5 layers MLP with communication
mpirun -outfile-pattern log_stdout_mlp_with_comm -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=5 -genv MXNET_P2PNET_DEBUG=0 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python tofu_test_mlp.py -f host --batch_size=8192 --hidden_size=8192 --num_layers=5

# 3 layers Conv single machine
mpirun -outfile-pattern log_stdout_conv_single -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=5 -genv MXNET_P2PNET_DEBUG=0 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python tofu_test_conv.py --batch_size=256 --channel_size=512 --filter_size=512 --num_layers=3 --addresses=127.0.0.1 -i 0

# 3 layers Conv without communication
mpirun -outfile-pattern log_stdout_conv_without_comm -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=30000 -genv MXNET_P2PNET_DEBUG=2 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python tofu_test_conv.py -t host --batch_size=256 --channel_size=512 --filter_size=512 --num_layers=3

# 3 layers Conv with communication
mpirun -outfile-pattern log_stdout_conv_with_comm -l -np $NP -ppn 1  -hostfile host -print-rank-map -genv MXNET_P2PNET_MPI_SLEEP_DURATION=5 -genv MXNET_P2PNET_DEBUG=0 -genv MXNET_P2PNET_HOST_PATH=/home/tofu/mxnet/host -genv MXNET_P2PNET_MAIN_AFFINITY=1 -genv MXNET_P2PNET_ZMQ_IO_THREADS=0 -genv KMP_AFFINITY="explicit,granularity=fine,proclist=[$AFFINITY]" python tofu_test_conv.py -t host --batch_size=256 --channel_size=512 --filter_size=512 --num_layers=3
