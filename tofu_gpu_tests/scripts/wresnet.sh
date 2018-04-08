l=50

export TOFU_USE_BFS_LEVEL=1

#echo "Single: "
#for b in 1 2 4 8 16 32
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w \
#      2>&1 | tee resnet${l}_b${b}_w${w}.log | grep average
#  done
#done

#echo "Tofu-2GPUs:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 2 4 8 16 32 64
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=2 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_tofu_2.log | grep average
#  done
#done

#echo "Tofu-4GPUs:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 4 8 16 32 64 128
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=4 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_tofu_4.log | grep average
#  done
#done

#export TOFU_IGNORE_GPU_COMM=1
#echo "Tofu-2GPUs-no-comm:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 2 4 8 16 32 64
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=2 \
#      2>&1 | tee resnet${l}-no-comm_b${b}_w${w}_tofu_2.log | grep average
#  done
#done
#
#echo "Tofu-4GPUs-no-comm:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 4 8 16 32 64 128
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=4 \
#      2>&1 | tee resnet${l}-no-comm_b${b}_w${w}_tofu_4.log | grep average
#  done
#done

#export TOFU_TILINE_TYPE="k-equal-cuts"
#echo "Tofu-4GPUs-k-equal-cut:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 4 8 16 32 64 128
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=4 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_equal-cut_4.log | grep average
#  done
#done

export TOFU_TILINE_TYPE="spartan"
echo "Tofu-2GPUs-spartan:"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
for b in 2 4 8 16 32 64
do
  for w in 1 2 4 8
  do
    echo "Batch: $b, Wide: $w"
    python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=2 \
      2>&1 | tee resnet${l}_b${b}_w${w}_spartan_2.log | grep average
  done
done

echo "Tofu-4GPUs-spartan:"
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
for b in 4 8 16 32 64 128
do
  for w in 1 2 4 8
  do
    echo "Batch: $b, Wide: $w"
    python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --wide_scale=$w --num_gpus=4 \
      2>&1 | tee resnet${l}_b${b}_w${w}_spartan_4.log | grep average
  done
done

#echo "DP:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=datapar
#for b in 128 256 512
#do
#  for n in 2 4 8
#  do
#    echo "Batch: $b, #GPUs: $n"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --num_gpus=$n \
#      2>&1 | tee resnet${l}_b${b}_dp_${n}.log | grep average
#  done
#done
#
#export TOFU_IGNORE_GPU_COMM=1
#echo "Tofu-no-comm:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 128 256 512
#do
#  for n in 2 4 8
#  do
#    echo "Batch: $b, #GPUs: $n"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --num_gpus=$n \
#      2>&1 | tee resnet${l}-no-comm_b${b}_tofu_${n}.log | grep average
#  done
#done
#
#echo "DP-no-comm:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=datapar
#for b in 128 256 512
#do
#  for n in 2 4 8
#  do
#    echo "Batch: $b, #GPUs: $n"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --num_gpus=$n \
#      2>&1 | tee resnet${l}-no-comm_b${b}_dp_${n}.log | grep average
#  done
#done

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
