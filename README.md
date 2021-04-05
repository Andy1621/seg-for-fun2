# seg-for-fun2

基于PaddleSeg2动态图开发的遥感地块分割解决方案，部分代码基于静态图版本的[seg-for-fun](https://github.com/Andy1621/seg-for-fun)。

## note
本仓库实现部分由静态图到动态图的迁移代码，仅供参考，exp目录下训练与测试脚本无法在该版本下运行，请自行修改。

### 迁移参考
可参考文档[add new model](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/add_new_model.md)添加model、loss、backbone、dataset、transform等：

- backbone修改：
    `paddleseg/models/backbones/hrnet.py`中添加了scSE attention，如下所示
    ```python
    class sSELayer(nn.Layer):
        def __init__(self, num_channels, name=None):
            super(sSELayer, self).__init__()
            self.excitation = nn.Conv2D(num_channels, 1, 1, padding='same', bias_attr=False)

        def forward(self, x):
            excitation = self.excitation(x)
            excitation = F.sigmoid(excitation)
            out = x * excitation
            return out


    class scSELayer(nn.Layer):
        def __init__(self, num_channels, num_filters, name=None):
            super(scSELayer, self).__init__()
            self.sSE = sSELayer(num_channels, name=name + '_sSE')
            self.cSE = SELayer(num_channels, num_filters=num_filters, reduction_ratio=2, name=name + '_cSE')
        
        def forward(self, x):
            x_sse = self.sSE(x)
            x_cse = self.cSE(x)
            return x_sse + x_cse 
    ```
    只需要相应添加has_scse字段即可，预训练模型可直接使用ImageNet-pretrained的backbone，参数加载时会忽略无法加载的字段。
    读者可参考[合集：基于Paddle2.0的含有注意力机制的卷积网络](https://aistudio.baidu.com/aistudio/projectdetail/1562506?channelType=0&channel=0)实现更多attention。
- 增加dataset：
    `paddleseg/datasets/remote_sensing.py`中添加了遥感影像数据集类，如下所示
    ```python
    @manager.DATASETS.add_component
    class RemoteSensing(Dataset):
    """
    Args:
        transforms (list): Transforms for image.
        train_dataset_root (str): The training dataset directory. Default: None
        test_dataset_root (str): The training dataset directory. Default: None
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 7

    def __init__(self,
                 train_dataset_root=None,
                 test_dataset_root=None,
                 negetive_ratio=0,
                 positive_train_dataset_list=None,
                 negetive_train_dataset_list=None,
                 transforms=None,
                 mode='train',
                 edge=False):
        self.train_dataset_root = train_dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if mode == 'train':
            file_path = os.path.join(self.train_dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.train_dataset_root, 'val_list.txt')
        else:
            file_path = os.path.join(self.test_dataset_root, 'test_list.txt')

        
        if mode == 'train' and   negetive_ratio != 0:
            positive_file_path = os.path.join(self.train_dataset_root, positive_train_dataset_list)
            negetive_file_path = os.path.join(self.train_dataset_root, negetive_train_dataset_list)
            with open(positive_file_path, 'r') as f:
                lines = f.readlines()
                positive_lines = [line for line in lines]
                positive_length = len(positive_lines)
            with open(negetive_file_path, 'r') as f:
                lines = f.readlines()
                negetive_lines = [line for line in lines]
                negetive_length = len(negetive_lines)
            if int(positive_length * negetive_ratio) < negetive_length:
                negetive_length = int(positive_length * negetive_ratio)
            sample_lines = positive_lines + random.sample(negetive_lines, int(negetive_length))
            for line in sample_lines:
                items = line.strip().split()
                image_path = os.path.join(self.train_dataset_root, items[0])
                grt_path = os.path.join(self.train_dataset_root, items[1])
                self.file_list.append([image_path, grt_path])
            print(f"{positive_length} positive data from :", negetive_train_dataset_list)
            print(f"Add {negetive_length} negetive data from :", negetive_train_dataset_list)
            print(f"Total data for {mode} : {len(self.file_list)}")
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    items = line.strip().split()
                    if len(items) != 2:
                        if mode == 'train' or mode == 'val':
                            raise Exception(
                                "File list format incorrect! It should be"
                                " image_name label_name\\n")
                        image_path = os.path.join(self.test_dataset_root, items[0])
                        grt_path = None
                    else:
                        image_path = os.path.join(self.train_dataset_root, items[0])
                        grt_path = os.path.join(self.train_dataset_root, items[1])
                    self.file_list.append([image_path, grt_path])
            print(f"Total data for {mode} : {len(self.file_list)}")
    ```
    这里设置了`positive_train_dataset_list`、`positive_train_dataset_list`，主要用于多阶段增加负样本比例`negetive_ratio`，在静态图版本中通过修改`reader.py`实现。

- 增加transform：
    `paddleseg/transforms/transforms.py`中增加了新的transform，如下所示：
    ```python
    @manager.TRANSFORMS.add_component
    class MyRandomRotate90:
    """RandomRotate 90/180/270 for the input image.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im, label=None):
        aug = RandomRotate90(p=self.p)
        aug_img = aug(image=im, mask=label)
        return (aug_img['image'], aug_img['mask'])
    ```
    代码直接调用了[albumentations](https://github.com/albumentations-team/albumentations)中实现的快速transform函数，可自行借鉴增加。

- 多模型投票与形态学后处理：
  PaddleSeg2的`val.py`实现了较多的测试技巧，包括TTA(test-time augmentation)等，在静态图版本[seg-for-fun](https://github.com/Andy1621/seg-for-fun)中，笔者也手动实现了TTA。不同的是，笔者实现的为“硬投票”，即生成图片后，对多张图片进行投票处理，而PaddleSeg2实现的为“软投票”，对sofotmax分数进行求和投票，各有利弊。
  多模型投票的基本思路为：
  1. 通过变换生成不同size、不同旋转角度数据集
  2. 对不同数据集进行测试
  3. 对测试结果进行变换，得到相同size相同角度的测试结果
  4. 对测试结果进行投票加权
  `tools`目录下提供了部分代码参考，第1步直接对测试数据进行transform即可，第2步需调用`predict.py`进行结果预测，第3步可参考`invert_binary/multi_class_results.py`对结果进行变换，第4步可参考`binary/multi_class_voting.py`对结果进行投票。
  后处理涉及较多的图像形态学处理，包括腐蚀膨胀、骨架提取等，读者可参考`tools/post_processing.py`进行学习。

## 代码运行
- 下载数据到`raw_data`目录下，运行`main.sh`会调用`exp/prepare_dataset.sh`生成数据。
    1. 首先调用`exp/create_txt.sh `解压数据，并生成训练集与验证集到`data/rs_data/train_data`目录下，若有测试集，会生成到`data/rs_data`目录下。
    2. 此外，还会调用`tools/generate_my_dataset.py`生成一些类别增强数据集以及二分类数据集，如不需要，注释即可。
- 运行如下命令，会启动简易训练脚本：
    ```shell
    python train.py --config configs/quick_start/se_hrnet_remote_sensing_256x256_1k.yml
    ```
    `exp/model_config`中提供了一些静态图训练参数，读者可参考自行修改参数。
    