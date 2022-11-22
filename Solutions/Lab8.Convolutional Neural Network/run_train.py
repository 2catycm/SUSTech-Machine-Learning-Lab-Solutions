
# !wget -c "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
# !tar -zxvf "cifar-10-binary.tar.gz"
# !mv "cifar-10-batches-bin" "./datasets/cifar-10-batches-bin"
# %%
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset.transforms.transforms import TypeCast
import mindspore.dataset.engine as de # 数据增强引擎。
from mindspore import context
# from mindspore.communication.management import init, get_rank, get_group_size
# batch_size = 32
batch_size = 64
#%%
def create_context():
    '''GPU单机多卡训练目前只支持图模式'''
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # init("nccl") # 使能NCCL通信，
    # rank_id = get_rank()
    # '''device_num为mpirun中设置的GPU的使用个数'''
    # device_num = get_group_size()
    # '''设置数据并行模式（DATA_PARALLEL）'''
    # context.set_auto_parallel_context(device_num=device_num, gradients_mean=True,
    #                     parallel_mode=ParallelMode.DATA_PARALLEL)

def create_cifar_dataset(dataset_path, do_train, batch_size=batch_size, image_size=(224, 224), rank_size=1, rank_id=0):
    '''数据集中需要设置num_shards和shard_id'''
    dataset = ds.Cifar10Dataset(dataset_path, shuffle=do_train,
                                num_shards=rank_size, shard_id=rank_id)

    # define map operations
    trans = []
    # 训练的时候增强数据
    if do_train:
        trans += [
            # vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            # vision.RandomHorizontalFlip(prob=0.5)
            vision.AutoAugment(vision.AutoAugmentPolicy.CIFAR10),# 这是经过实验得出的最优策略
        ]

    trans += [
        vision.Resize(image_size),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    type_cast_op = TypeCast(ms.int32)

    data_set = dataset.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=trans, input_columns="image")

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=do_train)
    return data_set

# %%
# import sys
# sys.argv = []

# %%
# import mindspore_hub as mshub
# # model = "mindspore/1.9/res2net50_cifar10" # 这个模型也有，不过就没有意思了
# model = "mindspore/1.9/resnet50_imagenet2012" # 我们体验一下迁移学习
# # model = "mindspore/1.6/googlenet_cifar10"
# network = mshub.load(model, include_top=False, activation="Sigmoid", num_classes=10)
# network.set_train(False) # 这个API比pytorch直观很多

# %%
# !wget -N https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/source-codes/resnet.py

# %%
from resnet import resnet50

# %%
import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits

# %%
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import load_checkpoint, load_param_into_net
import os
from mindspore import Model
from mindspore.train.callback import SummaryCollector


def main():
    create_context()  # 创建上下文    
    path = 'datasets/cifar-10-batches-bin'
    train_loader, test_loader = [create_cifar_dataset(path, do_train) for do_train in [True, False]]

    network = resnet50(batch_size=batch_size, num_classes=10)
    # network.set_train(True) 
    # print(list(network.get_parameters()))
    interrupt_point = "./checkpoints/ms/network_interrupt.ckpt"
    if os.path.exists(interrupt_point):
        print('checkoutpoint detected. ')
        param_dict = load_checkpoint(interrupt_point)
        not_loaded = load_param_into_net(network, param_dict)
        if len(not_loaded)==0:
            print("Load checkpoint success!")
        else:
            print(f"Load checkpoint failed!{len(not_loaded)} params are not loaded. ")
    # 这个损失函数的名字比torch清晰一些
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), 0.01, 0.9)
    # opt = nn.Momentum(network.get_parameters(), 0.01, 0.9)

    model = Model(network, loss_fn=ls, optimizer=opt, metrics={'acc'}, 
                    amp_level='O2',) # GPU 用O2精度比较好
                    # boost_level ='O1') # 准确率不变的情况下，加速训练
    
    steps_per_epoch = train_loader.get_dataset_size()
    config_ck = ms.CheckpointConfig(save_checkpoint_steps=steps_per_epoch, 
                                    keep_checkpoint_max=16)
    ckpt_cb = ms.ModelCheckpoint(prefix='CIFAR-10-resnet50', 
                                directory='./checkpoints/ms', config=config_ck)
    locc_monitor = ms.LossMonitor(1) # 每个step打印一次loss
    time_monitor = TimeMonitor(steps_per_epoch)
    # %%
    summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1) # 如果下沉到GPU，这个参数要改成1

    # %%
    # try:
    model.fit(epoch=100, 
                train_dataset=train_loader, 
                valid_dataset=test_loader,
                valid_frequency=2,
                callbacks=[summary_collector, ckpt_cb, locc_monitor, time_monitor], 
                dataset_sink_mode=True, # 下沉到GPU。 
                # dataset_sink_mode=False, #  不下沉。 
                ) 
    # except KeyboardInterrupt:
    #     print('Interrupted')
    #     mindspore.save_checkpoint(network, interrupt_point)
if __name__ == '__main__':
    main() # 必须定义，不然报错freeze support https://blog.csdn.net/shenfuli/article/details/103969964