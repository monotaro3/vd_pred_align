<prerequisites>

chainer v7
cupy v7
chainercv v0.13.1
opencv
(for multi-GPU: chainermn v7)

<data prepration (source domain)>
download COWC dataset
configure process_COWC_dirs.txt (file name can be changed arbitrarily)
run process_COWC.py

<experiments>
pretraining:
python3 SSD_training.py --model ssd300 --resolution 0.3 --batchsize 32 --iteration 40000 --gpu --out [path of output dir] --snapshot_interval 1000 --datadir [path of source domain dataset] --lrdecay_schedule 28000 35000

prediction alignment(fine tuning):
python3 Adv_training.py --method proposed --batchsize 32 --max_iter 10000 --gpu 0 --out [path of output dir] --snapshot_interval 1000 --source_dataset [path of source domain dataset] --target_dataset [path of target domain dataset] --ssdpath  [path of pretrained model] --bufsize 128

If memory issue occurs, specify --cls_batch_split. This option splits a minibatch into multiple sub-minibatches whose size equals to cls_batch_split and accumulates gradients of sub-minibatches.
Or, use multi-GPU:
mpiexec --allow-run-as-root -n 2 python3 Adv_training.py --use_mn --method proposed --batchsize 32 --max_iter 10000 --gpu 0 --out [path of output dir] --snapshot_interval 1000 --source_dataset [path of source domain dataset] --target_dataset [path of target domain dataset] --ssdpath  [path of pretrained model] --bufsize 128
