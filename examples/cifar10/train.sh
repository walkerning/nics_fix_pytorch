#!bin/sh

gpu=${GPU:-0}
result_dir=$(date +%Y%m%d-%H%M%S)
mkdir -p ${result_dir}

CUDA_VISIBLE_DEVICES=$gpu python main.py \
		    --save-dir ${result_dir} \
		    $@ 2>&1 | tee ${result_dir}/train.log

