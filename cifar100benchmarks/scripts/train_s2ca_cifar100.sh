python train.py \
    --in-dataset CIFAR-100\
    --id_loc datasets/CIFAR100 \
    --gpu 0 \
    --seed 7 \
    --model resnet34 \
    --loss supcon \
    --epochs 50 \
    --proto_m 0.5 \
    --feat_dim 128 \
    --w 1 \
    --batch-size 512 \
    --cosine