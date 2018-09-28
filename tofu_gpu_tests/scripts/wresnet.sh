l=152

export TOFU_USE_BFS_LEVEL=1
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export TOFU_FUSED_CONVERSION=1
export TOFU_CACHE_TEMP_MEMORY=1

#export TOFU_ENABLED=0
#echo "Single: "
#for b in 4 8 16 32 64
#do
#  for w in 4 6 8 10
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_ideal.log | grep average
#  done
#done

export TOFU_ENABLED=1

#echo "Tofu-2GPUs:"
#for b in 2 4 8 16 32 64
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=2 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_tofu_2.log | grep average
#  done
#done
#
#echo "Tofu-4GPUs:"
#for b in 4 8 16 32 64
#do
#  for w in 4 6 8 10
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=4 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_tofu_4.log | grep average
#  done
#done

echo "Tofu-8GPUs:"
for b in 8 16 32 64
do
  for w in 4 6 8 10
  do
    echo "Batch: $b, Wide: $w"
    stdbuf -oL python ../tofu_benchmark.py resnet \
      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=8 \
      2>&1 | tee resnet${l}_b${b}_w${w}_tofu_8.log | grep average
  done
done

#echo "Tofu-4GPUs-k-equal-cut:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=k-equal-cuts
#for b in 4 8 16 32 64 128
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=4 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_equal-cut_4.log | grep average
#  done
#done
#
#echo "Tofu-8GPUs-k-equal-cut:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=k-equal-cuts
#for b in 8 16 32 64 128 256
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=8 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_equal-cut_8.log | grep average
#  done
#done

#echo "Tofu-2GPUs-spartan:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=spartan
#for b in 2 4 8 16 32 64
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=2 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_spartan_2.log | grep average
#  done
#done
#
#echo "Tofu-4GPUs-spartan:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=spartan
#for b in 4 8 16 32 64 128
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=4 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_spartan_4.log | grep average
#  done
#done
#
#echo "Tofu-8GPUs-spartan:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=spartan
#for b in 8 16 32 64 128 256
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=8 \
#      2>&1 | tee resnet${l}_b${b}_w${w}_spartan_8.log | grep average
#  done
#done

export TOFU_IGNORE_GPU_COMM=1
#
#echo "Tofu-2GPUs-no-comm:"
#export TOFU_FAKE_VAR_SPLIT_CONCAT=1
#export TOFU_TILING_TYPE=kcuts
#for b in 2 4 8 16 32 64
#do
#  for w in 1 2 4 8
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=2 \
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
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=4 \
#      2>&1 | tee resnet${l}-no-comm_b${b}_w${w}_tofu_4.log | grep average
#  done
#done
#
#echo "Tofu-8GPUs-no-comm:"
#for b in 8 16 32 64
#do
#  for w in 4 6 8 10
#  do
#    echo "Batch: $b, Wide: $w"
#    python ../tofu_benchmark.py resnet \
#      --batch_size=$b --num_layers=$l --wide_scale=$w --num_loops=15 --num_gpus=8 \
#      2>&1 | tee resnet${l}-no-comm_b${b}_w${w}_tofu_8.log | grep average
#  done
#done
