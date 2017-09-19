b=32
l=50

#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export TOFU_USE_BFS_LEVEL=1

echo "ResNet-$l"
echo "Single: "
python ../tofu_benchmark.py resnet \
  --num_gpus=1 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "Tofu: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "DP: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "Tofu-no-comm: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "DP-no-comm: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "Half-batch-single: "
python ../tofu_benchmark.py resnet \
  --num_gpus=1 --batch_size=$(($b/2)) --num_layers=$l \
  2>&1 | grep average

echo "Single-system-overhead: "
TOFU_NO_COMPUTATION=1 python ../tofu_benchmark.py resnet \
  --num_gpus=1 --batch_size=$(($b/2)) --num_layers=$l \
  2>&1 | grep average

echo "Tofu-no-comm-no-conversion: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "DP-no-comm-no-conversion: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1\
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "Tofu-system-overhead: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 TOFU_NO_COMPUTATION=1 \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average

echo "DP-system-overhead: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 TOFU_NO_COMPUTATION=1 \
  python ../tofu_benchmark.py resnet \
  --num_gpus=2 --batch_size=$b --num_layers=$l \
  2>&1 | grep average
