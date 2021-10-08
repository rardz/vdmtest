# vdmtest

## Training

To start training:

```bash
python -m torch.distributed.launch --nproc_per_node=n_gpu [NUMBER OF GPUS FOR TRAINING] train.py --data_dir [DATASET PATH]
```
