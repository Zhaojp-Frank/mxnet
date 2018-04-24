l=8
s=20

export TOFU_ENABLED=0

#echo "Single: "
#for b in 512
#do
#  for h in 4096 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_lstm_benchmark.py --batch_size=$b --num_layers=$l \
#      --hidden_size=$h --seq_len=$s --num_loops=15 \
#      2>&1 | tee lstm${l}-${s}_b${b}_h${h}.log | grep average
#  done
#done

#echo "2GPUs: "
#for b in 1 2 4 8 16 32 64 128 256
#do
#  for h in 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_lstm_benchmark.py --batch_size=$b --num_layers=$l \
#      --hidden_size=$h --seq_len=$s --num_gpus=2 \
#      2>&1 | tee lstm${l}-${s}_b${b}_h${h}_2.log | grep average
#  done
#done
#
echo "4GPUs: "
for b in 1 2 4 8 16 32 64 128 256 512
do
  for h in 8192
  do
    echo "Batch: $b, Hidden: $h"
    python ../tofu_lstm_benchmark.py --batch_size=$b --num_layers=$l \
      --hidden_size=$h --seq_len=$s --num_loops=15 --num_gpus=4 \
      2>&1 | tee lstm${l}-${s}_b${b}_h${h}_4.log | grep average
  done
done

#echo "8GPUs: "
#for b in 512 1024 2048
#do
#  for h in 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_lstm_benchmark.py --batch_size=$b --num_layers=$l \
#      --hidden_size=$h --seq_len=$s --num_loops=15 --num_gpus=8 \
#      2>&1 | tee lstm${l}-${s}_b${b}_h${h}_8.log | grep average
#  done
#done
