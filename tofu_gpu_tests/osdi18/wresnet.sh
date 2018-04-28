export TOFU_USE_BFS_LEVEL=1
export TOFU_FAKE_VAR_SPLIT_CONCAT=1
export TOFU_TILING_TYPE=kcuts
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export TOFU_FUSED_CONVERSION=1
export TOFU_CACHE_TEMP_MEMORY=1
export TOFU_IGNORE_WEIGHT_REDUCTION=1

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

for l in 50 101 152
do
  echo "ResNet-${l} 8GPUs Tofu:"
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
done
