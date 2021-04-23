#!/bin/bash

python main.py --batch_size 4 --loss 1*L1_Charbonnier_loss_color --lr 1e-4 --end_epoch 500 --activation relu --dataset gopro_lmdb_event --data_root /home/ma-user/work/dvs/datasets/gopro_compact/ --num_gpus 1 --thread 8 --trainer_mode dp --lr_scheduler lr_bit --model Event_ESTRNN --frames 10
