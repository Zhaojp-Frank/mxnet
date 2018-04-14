b=32
f=2048
i=14

export TOFU_ENABLED=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

echo "ResUnit b=$b, f=$f, i=$i"
echo "Single: "
python ../tofu_benchmark.py miniresnet \
  --num_gpus=1 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "Tofu: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "DP: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "Tofu-no-comm: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "DP-no-comm: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "Half-batch-single: "
python ../tofu_benchmark.py miniresnet \
  --num_gpus=1 --batch_size=$(($b/2)) --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "Tofu-no-comm-no-conversion: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "DP-no-comm-no-conversion: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1\
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "Tofu-system-overhead: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=kcuts TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 TOFU_NO_COMPUTATION=1 \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average

echo "DP-system-overhead: "
TOFU_FAKE_VAR_SPLIT_CONCAT=1 TOFU_TILING_TYPE=datapar TOFU_IGNORE_GPU_COMM=1 TOFU_IGNORE_CONVERSION=1 TOFU_NO_COMPUTATION=1 \
  python ../tofu_benchmark.py miniresnet \
  --num_gpus=2 --batch_size=$b --num_filter=$f --image_size=$i \
  2>&1 | grep average
