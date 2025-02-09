/home/zyserver/miniconda3/envs/Torch2.0/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
/data1/YHC/QiTanBasecall/train_basecall_parallel.py  \
/data1/YHC/Model_Save \
/data1/YHC/QiTanTrain/QiTan_basecall_train_all_3 \
--batch_size \
64 \
--num_epochs \
80 \
--learning_rate \
0.0003 \
--clip \
0.7 \
--save_tag \
TF_0208 \
--load_previous \
/data1/YHC/Model_Save/TF_0113_epoch:13_loss:0.351935_lr:0.000282_device-1_model.pt
