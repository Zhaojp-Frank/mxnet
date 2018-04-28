s=20

export TOFU_ENABLED=1
export TOFU_USE_BFS_LEVEL=1
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export MXNET_EXEC_INPLACE_GRAD_SUM_CAP=1
export TOFU_FUSED_CONVERSION=1
#export TOFU_IGNORE_GPU_COMM=1
export TOFU_CACHE_TEMP_MEMORY=1

#echo "Single: "
#for b in 1 2 4 8 16 32 64 128 256
#do
#  for h in 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_lstm_benchmark.py --batch_size=$b --num_layers=$l \
#      --hidden_size=$h --seq_len=$s --num_loops=15 \
#      2>&1 | tee lstm${l}-${s}_b${b}_h${h}.log | grep average
#  done
#done

for l in 4 6 8 10
do
  echo "LSTM-${l} 8GPUs TOFU: "
  for b in 8 16 32 64 128 256 512
  do
    for h in 4096 6144 8192 10240
    do
      echo "Batch: $b, Hidden: $h"
      stdbuf -oL python ../tofu_annotated_lstm.py --batch_size=$b --num_layers=$l \
        --hidden_size=$h --seq_len=$s --num_loops=15 --num_gpus=8 \
        2>&1 | tee lstm${l}-${s}_b${b}_h${h}_tofu_8.log | grep average
    done
  done
done

export TOFU_ENABLED=0
for l in 4 6 8 10
do
  echo "LSTM-${l} 8GPUs graph partition: "
  for b in 8 16 32 64 128 256 512
  do
    for h in 4096 6144 8192 10240
    do
      echo "Batch: $b, Hidden: $h"
      stdbuf -oL python ../tofu_lstm_benchmark.py --batch_size=$b --num_layers=$l \
        --hidden_size=$h --seq_len=$s --num_loops=15 --num_gpus=8 \
        2>&1 | tee lstm${l}-${s}_b${b}_h${h}_gp_8.log | grep average
    done
  done
done
