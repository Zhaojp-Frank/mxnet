l=50

export TOFU_USE_BFS_LEVEL=1
export MXNET_GPU_COPY_NTHREADS=2

#echo "Single: "
#for b in 128
#do
#  echo "Batch: $b"
#  python ../tofu_benchmark.py resnet \
#    --batch_size=$b --num_layers=$l \
#    2>&1 | tee resnet${l}_b${b}.log | grep average
#done
#
echo "Tofu:"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
for b in 128 256 512
do
  for n in 8
  do
    echo "Batch: $b, #GPUs: $n"
    python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --num_gpus=$n \
      2>&1 | tee resnet${l}_b${b}_tofu_${n}.log | grep average
  done
done

echo "DP:"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=datapar
for b in 128 256 512
do
  for n in 8
  do
    echo "Batch: $b, #GPUs: $n"
    python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --num_gpus=$n \
      2>&1 | tee resnet${l}_b${b}_dp_${n}.log | grep average
  done
done

export TOFU_IGNORE_GPU_COMM=1
echo "Tofu-no-comm:"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
for b in 128 256 512
do
  for n in 8
  do
    echo "Batch: $b, #GPUs: $n"
    python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --num_gpus=$n \
      2>&1 | tee resnet${l}-no-comm_b${b}_tofu_${n}.log | grep average
  done
done

echo "DP-no-comm:"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=datapar
for b in 128 256 512
do
  for n in 8
  do
    echo "Batch: $b, #GPUs: $n"
    python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --num_gpus=$n \
      2>&1 | tee resnet${l}-no-comm_b${b}_dp_${n}.log | grep average
  done
done

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
