#!bin/sh

#arch="vgg11_web"
#arch="vgg11_ugly"
arch="vgg11_elegant"

gpu=0
data=$(date +%m%d)
method="auto8_ele"+$date

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --arch $arch \
    --batch-size 128 \
    --test-batch-size 1000 \
    --save-dir save_fix \
    2>&1 | tee logs/log-auto8_1222.log
