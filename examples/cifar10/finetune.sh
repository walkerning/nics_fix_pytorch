#!bin/sh
#arch="vgg11_web"
#arch="vgg11_elegant"
arch="vgg11_ugly"

gpu=1
data=$(date +%m%d)
method="8_conv1233'44'55'_6"
lr=0.0001
# model="save_fix/checkpoint_auto8_90.730.tar"
model="save_fix/checkpoint_8_conv1233'44'5_6_90.820.tar"
epoches=30

CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --pretrained $model \
    --arch $arch \
    --lr $lr \
    --epoches $epoches \
    --prefix $method \
    --batch-size 128 \
    --test-batch-size 1000 \
    --save-dir save_fix \
    2>&1 | tee logs/log-"$method"_1222.log
