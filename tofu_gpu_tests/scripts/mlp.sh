n=2

#echo "Single: "
#for b in 512 1024 2048 4096 8192
#do
#  for h in 512 1024 2048 4096 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_benchmark.py mlp \
#      --hidden_size=$h --batch_size=$b \
#      2>&1 | tee mlp_b${b}_h${h}.log | grep average
#  done
#done

#echo "Tofu: #GPUs=$n"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 512 1024 2048 4096 8192
#do
#  for h in 512 1024 2048 4096 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_benchmark.py mlp \
#      --hidden_size=$h --batch_size=$b --num_gpus=$n \
#      2>&1 | tee mlp_b${b}_h${h}_tofu_${n}.log | grep average
#  done
#done
#
#echo "DP: #GPUs=$n"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=datapar
#for b in 512 1024 2048 4096 8192
#do
#  for h in 512 1024 2048 4096 8192
#  do
#    echo "Batch: $b, Hidden: $h"
#    python ../tofu_benchmark.py mlp \
#      --hidden_size=$h --batch_size=$b --num_gpus=$n \
#      2>&1 | tee mlp_b${b}_h${h}_dp_${n}.log | grep average
#  done
#done

echo "Tofu-no-comm: #GPUs=$n"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
export TOFU_IGNORE_GPU_COMM=1
for b in 512 1024 2048 4096 8192
do
  for h in 512 1024 2048 4096 8192
  do
    echo "Batch: $b, Hidden: $h"
    python ../tofu_benchmark.py mlp \
      --hidden_size=$h --batch_size=$b --num_gpus=$n \
      2>&1 | tee mlp-no-comm_b${b}_h${h}_tofu_${n}.log | grep average
  done
done

echo "DP-no-comm: #GPUs=$n"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=datapar
export TOFU_IGNORE_GPU_COMM=1
for b in 512 1024 2048 4096 8192
do
  for h in 512 1024 2048 4096 8192
  do
    echo "Batch: $b, Hidden: $h"
    python ../tofu_benchmark.py mlp \
      --hidden_size=$h --batch_size=$b --num_gpus=$n \
      2>&1 | tee mlp-no-comm_b${b}_h${h}_dp_${n}.log | grep average
  done
done

#echo "Half-batch-single: "
#python ../tofu_benchmark.py miniresnet \
#  --num_gpus=1 --batch_size=$(($b/2)) --num_filter=$f --image_size=$i \
#  2>&1 | grep average
#
#echo "Tofu-no-comm-no-conversion: "
#TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 \
#  python ../tofu_benchmark.py miniresnet \
#  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
#  2>&1 | grep average
#
#echo "DP-no-comm-no-conversion: "
#TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1\
#  python ../tofu_benchmark.py miniresnet \
#  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
#  2>&1 | grep average
#
#echo "Tofu-system-overhead: "
#TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 TOFU_NO_COMPUTATION=1 \
#  python ../tofu_benchmark.py miniresnet \
#  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
#  2>&1 | grep average
#
#echo "DP-system-overhead: "
#TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 TOFU_NO_COMPUTATION=1 \
#  python ../tofu_benchmark.py miniresnet \
#  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
#  2>&1 | grep average
