LAYERS=152
WIDE_SCALE=8
export MXNET_SWAP_NO_COPY=1
export MXNET_SWAP_THRESHOLD_MULTIPLIER=512
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 32 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_32_${WIDE_SCALE}_no_copy
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 16 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_16_${WIDE_SCALE}_no_copy
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 8 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_8_${WIDE_SCALE}_no_copy
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 4 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}_no_copy
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 2 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_2_${WIDE_SCALE}_no_copy
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 1 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_1_${WIDE_SCALE}_no_copy
export MXNET_SWAP_NO_COPY=0
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 32 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_32_${WIDE_SCALE}
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 16 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_16_${WIDE_SCALE}
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 8 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_8_${WIDE_SCALE}
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 4 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 2 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_2_${WIDE_SCALE}
python tofu_benchmark.py --num_layers=${LAYERS} --batch_size 1 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_1_${WIDE_SCALE}
