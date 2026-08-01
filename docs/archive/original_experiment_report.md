# 基于卷积神经网络的视网膜OCT图像分类

## 一、 实验目的

1. **掌握深度学习分类全流程**： 熟练运用 PyTorch 框架完成数据加载、预处理、模型构建及训练评估，深入理解卷积神经网络(CNN)在图像分类中的应用。
2. **构建现代卷积网络架构**： 自主搭建集成 **ConvNeXt 模块**与 **SE 注意力机制**的特征提取网络，掌握现代 CNN 设计理念与通道增强技术。
3. **医学影像分析实战**： 基于视网膜 OCT 数据集实现 8 种眼底疾病的自动分类，探索计算机辅助诊断技术在医学场景中的应用。
4. **应用高级优化策略**： 综合运用高级数据增强（如 RandomErasing）、**AdamW 优化器**、**标签平滑**及**混合精度训练 (AMP)**，提升模型的泛化能力与训练效率。
5. **多维度性能评估**： 利用混淆矩阵、精确率、召回率及 F1-score 等指标，对模型分类性能进行全面的量化分析与可视化。

## 二、 实验原理

### 2.1 现代CNN架构设计

​	本实验构建了一个自定义的 `OCTNet`，其核心组件借鉴了 ConvNeXt [1]的设计理念，旨在优化传统残差网络（ResNet）的性能。`ConvNeXtBlock` 采用了**倒残差结构（Inverted Bottleneck）和深度可分离卷积**，具体流程如下：

1. **深度卷积 (Depthwise Conv)**：首先使用 $7 \times 7$ 的大卷积核进行分组卷积，扩大感受野并减少参数量。

2. **归一化 (Group Normalization)**：代码中使用 `GroupNorm` 替代了传统的 `BatchNorm`，这在小批量训练时通常能提供更稳定的性能。

3. **逐点卷积 (Pointwise Conv) 与 激活**：通过 $1 \times 1$ 卷积将通道数扩展为原来的 4 倍，经过 `GELU` 激活函数后，再通过另一个 $1 \times 1$ 卷积映射回原通道数。

4. **残差连接 (Residual Connection)**：将输入直接加到输出上，防止梯度消失，利于深层网络训练。

### 2.2 通道注意力机制

为了增强模型对关键特征的提取能力，网络在深层特征提取阶段（Stage 3-5）引入了 SE 注意力模块[2]。其分为3个步骤：

1. **压缩 (Squeeze)**：通过全局平均池化（`AdaptiveAvgPool2d`）将特征图的空间维度压缩为 $1 \times 1$，提取通道的全局分布信息。

2. **激励 (Excitation)**：通过两个全连接层（中间层带有降维比率 reduction=16）学习通道间的相关性，最后使用 `Sigmoid` 函数生成每个通道的权重系数。

3. **重标定 (Scale)**：将生成的权重乘回原始特征图，从而抑制无用特征，强调对分类有帮助的特征通道。

### 2.3 标签平滑

实验使用了带有 `label_smoothing=0.1` 的交叉熵损失函数。

**原理**：传统的 One-hot 编码标签（如 [0, 1, 0]）会迫使模型对正确类别的预测概率无限趋近于 1，这容易导致模型过分自信（Overconfidence）并引发过拟合。

**实现**：标签平滑将硬标签转换为软标签（例如 [0.033, 0.933, 0.033]），增加标签分布的不确定性，迫使模型学习更鲁棒的特征聚类，从而提升泛化能力[3]。

### 2.3 优化算法与学习率调度

**AdamW 优化器**：相比于标准的 Adam，AdamW 将权重衰减（Weight Decay）与梯度更新解耦[4]。这能更有效地规范模型权重，防止过拟合，是目前训练 Transformer 和现代 CNN 的主流选择。

**余弦退火调度 (Cosine Annealing LR)**：实验采用了 `CosineAnnealingLR`，学习率在训练过程中按照余弦函数曲线从初始值下降到最小值[5]。这种非线性的下降方式能帮助模型在训练后期更平滑地收敛到损失函数的局部极小值。

### 2.5 数据增强

为了解决医学图像数据量相对有限的问题，实验在预处理阶段引入了多种增强策略：

1. **几何变换**：随机水平翻转 (`RandomHorizontalFlip`)、随机旋转 (`RandomRotation`)。

2. **光度变换**：颜色抖动 (`ColorJitter`)，改变图像的亮度与对比度，模拟不同成像条件。

3. **随机擦除 (Random Erasing)**：在图像中随机选取矩形区域并赋予随机值，模拟遮挡情况，迫使网络利用局部特征进行识别，提高模型的鲁棒性[6]。

## 三、 实验内容

### 3.1 数据准备与预处理

- **数据集加载**：使用 `RetinalOCT_Dataset` 数据集，包含 8 个视网膜疾病类别：AMD（年龄相关性黄斑变性）、CNV（脉络膜新生血管）、CSR（中心性浆液性脉络膜视网膜病变）、DME（糖尿病黄斑水肿）、DR（糖尿病视网膜病变）、DRUSEN（玻璃膜疣）、MH（黄斑裂孔）和 NORMAL（正常）。
- **数据集划分**：将数据集按文件夹结构划分为训练集（18,400张）、验证集（2,800张）和测试集（2,800张）。
- **图像预处理与增强**：
    - 统一将图像尺寸调整为 $224 \times 224$ 像素，使用 BICUBIC 插值。
    - 对所有数据进行归一化处理（Mean=[0.210], Std=[0.182]）。
    - **训练集增强策略**：应用随机水平翻转 (`RandomHorizontalFlip`)、随机旋转 15 度 (`RandomRotation`)、亮度与对比度抖动 (`ColorJitter`) 以及随机擦除 (`RandomErasing`, p=0.4)，以增强数据多样性并防止过拟合。

### 3.2 自定义CNN搭建

- **模块设计**：
    - **SEBlock**：实现压缩与激励机制，通过全连接层学习通道权重（Reduction=16），增强关键特征响应。
    - **ConvNeXtBlock**：构建基于 ConvNeXt 风格的残差块，包含 $7 \times 7$ 深度可分离卷积、`GroupNorm` 归一化、$1 \times 1$ 逐点卷积扩展维度（倍率4）以及 `GELU` 激活函数。
- **整体架构**：
    - 构建 `OCTNet` 类，包含 5 个特征提取阶段（Stage），通道数依次为 32, 64, 128, 256, 512。
    - 在深层网络（Stage 3, 4, 5）中集成 `SEBlock`。
    - 分类头（Classifier）包含 `AdaptiveAvgPool2d`、全连接层（512 -> 1024 -> 8）、`GELU` 激活以及 `Dropout`（比率 0.5）。
- **模型复杂度**：输入为 $224 \times 224$ 时，模型计算量（FLOPs）约为 2.32 G，参数量（Params）约为 3.60 M。

### 3.3 实验环境与超参数设置

- **硬件环境**：使用 CUDA 加速训练。
- **损失函数**：使用带有标签平滑 (`Label Smoothing = 0.1`) 的交叉熵损失函数 (`CrossEntropyLoss`)。
- **优化器**：使用 `AdamW` 优化器，初始学习率设为 `5e-4`，权重衰减 (`weight_decay`) 设为 `3e-5`。
- **学习率调度**：采用余弦退火策略 (`CosineAnnealingLR`)，周期 $T_{max}=20$，最小学习率 $1e-6$。
- **训练策略**：启用混合精度训练 (`torch.amp.GradScaler`) 以减少显存占用并加速计算。Batch Size 设为 64，训练轮次 (Epochs) 设为 20。

### 3.4 模型训练与验证流程

编写训练函数 `train_model`，在每个 Epoch 中交替进行训练和验证。

- **训练阶段**：前向传播计算损失，利用 Scaler 进行反向传播和参数更新，并在每个 Epoch 结束后更新学习率。

- **验证阶段**：在验证集上评估模型性能，仅保存验证集准确率（Val Accuracy）最高的模型权重作为最佳模型。

记录并保存每个 Epoch 的训练损失、训练准确率、验证损失和验证准确率。

### 3.5 结果可视化与测试评估

**曲线绘制**：使用 Matplotlib 绘制训练集与验证集的 Loss 曲线和 Accuracy 曲线，分析模型收敛情况及是否存在过拟合。

**测试集评估**：

- 加载训练好的最佳模型权重，在独立的测试集上进行推理。
- 计算并输出**分类报告 (Classification Report)**，包含每个类别的精确率 (Precision)、召回率 (Recall) 和 F1-score。
- 绘制**混淆矩阵 (Confusion Matrix)** 热力图，直观展示模型在各类别间的预测分布与混淆情况。
- 计算测试集的全局准确率 (Global Accuracy)。

## 四、 实验结果与分析

### 4.1 训练过程分析

<img src="D:\Typora Images\image-20251126201919631.png" alt="image-20251126201919631" style="zoom:80%;" />

通过观察训练过程中的损失（Loss）和准确率（Accuracy）变化曲线并结合训练日志，可以得出以下分析：

- **收敛趋势**：
    - 模型在训练初期（Epoch 1-5）收敛迅速，训练集准确率从 39.45% 快速提升至 85.63%，验证集准确率也同步提升至 88.29%。这表明 ConvNeXt 风格的卷积块能够快速提取有效的底层视觉特征。
    - 在中后期（Epoch 10-20），损失函数下降趋于平缓，训练集和验证集的准确率稳步上升。
- **最佳模型**：
    - 训练过程在第 **15** 个 Epoch 达到了最佳验证集准确率 **96.64%**（Val Loss: 0.5467）。
- **泛化能力**：
    - 值得注意的是，在整个训练过程中，验证集的准确率（Val Acc）始终略高于或紧贴训练集准确率（Train Acc）。例如在第 15 轮，Train Acc 为 94.80%，而 Val Acc 为 96.64%。
    - **分析**：这种现象通常归功于训练集中使用了较强的数据增强策略（如 `RandomErasing`, `ColorJitter`, `RandomRotation`）以及 `Dropout` 和 `Label Smoothing`。这些正则化手段增加了训练难度，但有效防止了过拟合，使得模型在未见过的验证数据上表现优异。

### 4.2 测试集定量评估

加载最佳模型权重后，在独立的测试集（2800张图像）上进行了最终评估，结果如下：

**全局准确率 (Global Accuracy)**：**97.04%**：

​	模型在测试集上的表现优于验证集（96.64%），证明了 OCTNet 架构具有极强的鲁棒性和泛化能力，能够很好地适应未知的医学影像数据。

**分类报告性能分析 (Classification Report)：**

​	下图展示了模型在 8 个类别上的详细性能指标：

<img src="D:\Typora Images\image-20251126202652322.png" alt="image-20251126202652322" style="zoom:50%;" />

**性能解读**：

- **完美分类类别**：模型在 **AMD** 、**CSR**、**DR**和 **MH** 这四个类别上实现了 **100% 的精确率和召回率**。这说明这些疾病在 OCT 图像上的病理特征（如黄斑裂孔的断裂结构）非常显著，且 SE 注意力模块成功捕捉到了这些关键特征。
- **易混淆类别**：
    - **CNV**与 **DRUSEN ** 的 F1-score 相对较低（约 0.93）。
    - **NORMAL ** 类别的召回率很高 (97.71%)，但精确率稍低 (93.19%)。这意味着模型极少漏诊正常样本，但偶尔会将其他病变样本误判为正常，或者将某些轻微病变误判为正常。

### 4.3 混淆矩阵分析

<img src="D:\Typora Images\image-20251126203213282.png" alt="image-20251126203213282" style="zoom: 50%;" />

通过观察混淆矩阵（Confusion Matrix）的热力图分布：

- **对角线主导**：矩阵的对角线区域颜色最深，数值最大，代表绝大多数样本被正确分类。
- **误判分布**：
    - 主要的混淆发生在 DME 、 DRUSEN 和 CNV 之间。这些疾病在 OCT 图像上都可能表现为视网膜层的隆起或层间积液，纹理特征具有一定的相似性，导致 CNN 在区分细微差别时面临挑战。

### 4.4 实验小结

实验结果表明，自主搭建的 **OCTNet**（集成 ConvNeXt 块与 SE 注意力机制）在视网膜 OCT 图像分类任务上达到了 **97.04%** 的高准确率。

- **SE 模块的作用**：全 100% 识别率的类别证明了通道注意力机制在提取显著病理特征方面的有效性。
- **优化策略的有效性**：Label Smoothing 和 AdamW 优化器配合 CosineAnnealingLR 调度，使得模型训练曲线平滑，收敛稳定，且未出现过拟合现象。

## 五、 实验结论

1. **模型架构设计的有效性验证**： 本实验自主设计的 **OCTNet** 卷积神经网络，成功结合了 **ConvNeXt** 的现代化设计（大核深度可分离卷积、倒残差结构）与 **SE (Squeeze-and-Excitation)** 通道注意力机制。实验结果显示，该模型在视网膜 OCT 图像 8 分类任务上取得了 **97.04%** 的测试集准确率。这一高精度证明了该架构能够有效提取医学影像中细微且关键的病理特征，证明了现代 CNN 设计理念在特定领域任务中的优越性。
2. **注意力机制对特征提取的增强作用**： 通过引入 SE 模块，模型在 **AMD、CSR、DR 和 MH** 四个类别上实现了完美的分类效果（F1-score 均为 1.0000）。这表明通道注意力机制有效地帮助网络聚焦于具有显著病理特征的通道，显著提升了模型对关键病灶的辨识能力，验证了在卷积网络深层引入注意力机制的必要性。
3. **正则化与优化策略的成功应用**： 实验中采用的 **Label Smoothing（标签平滑）** 损失函数配合 **RandomErasing** 等强数据增强策略，有效地抑制了模型的过拟合倾向。在训练全过程中，验证集准确率始终紧跟甚至略高于训练集准确率，表明模型学到了具有强泛化能力的特征表示，而非简单地记忆训练样本。此外，**AdamW** 优化器与 **余弦退火学习率调度** 的组合确保了模型训练的稳定收敛。
4. **局限性与改进方向**： 尽管整体性能优异，但混淆矩阵显示模型在 **DME**与 **DRUSEN** 等纹理特征相似的类别间仍存在少量误判。未来的改进方向可以包括：
    - 引入空间注意力机制（Spatial Attention）以关注病灶的位置信息。
    - 尝试使用迁移学习（Transfer Learning），利用在大规模数据集（如 ImageNet）上预训练的更深层模型（如 ResNet-50 或 ConvNeXt-T）进行微调。
    - 利用 Grad-CAM 等可视化技术分析模型关注区域，提高模型的可解释性。



# 参考文献

[1] Liu, Zhuang, et al. "A convnet for the 2020s." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. 2022. 

[2] Hu, Jie, et al. "Squeeze-and-excitation networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2018.

[3] Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016.

[4] Loshchilov, Ilya, and Frank Hutter. "Decoupled weight decay regularization." *International Conference on Learning Representations (ICLR)*. 2019. 

[5] Loshchilov, Ilya, and Frank Hutter. "SGDR: Stochastic gradient descent with warm restarts." *International Conference on Learning Representations (ICLR)*. 2017. 

[6] Zhong, Zhun, et al. "Random erasing data augmentation." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 34. No. 07. 2020. 



# 代码附录

## 环境配置


```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from torchvision.transforms import InterpolationMode
from thop import profile
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```

## 数据预处理


```python
img_size = 224
norm_mean = [0.210, 0.210, 0.210]
norm_std = [0.182, 0.182, 0.182]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.4)      
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
}

data_dir = 'RetinalOCT_Dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloaders = {
    x: DataLoader(
        image_datasets[x], 
        batch_size=64, 
        shuffle=(x == 'train'), 
        num_workers=4,       
        pin_memory=True,               
        persistent_workers=True,       
        prefetch_factor=4
    )
    for x in ['train', 'val', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"设备: {device}")
print(f"类别：{class_names}")
print("训练/验证/测试集数量:", dataset_sizes)
```

## CNN搭建


```python
# SE 注意力模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```


```python
# ConvNeXtBlock
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度可分离卷积 (Depthwise Conv)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 这里改用 GroupNorm 代替 LayerNorm，效果近似且速度快
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim) 
        # 1x1 卷积代替 Linear，避免 permute
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x
```


```python
class OCTNet(nn.Module):
    def __init__(self, num_classes=8):
        super(OCTNet, self).__init__()

        def simple_block(in_ch, out_ch, use_se=False):
            layers = []
            # 如果通道数变化，先用一个 1x1 卷积调整通道
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))            
            # 核心：ConvNeXt Block
            layers.append(ConvNeXtBlock(out_ch))            
            # 可选：SE Block
            if use_se:
                layers.append(SEBlock(out_ch))            
            # 下采样
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            # Stage 1: 浅层特征，不用 SE，保持简单
            simple_block(3, 32, use_se=False),
            # Stage 2
            simple_block(32, 64, use_se=False),
            # Stage 3
            simple_block(64, 128, use_se=True),
            # Stage 4
            simple_block(128, 256, use_se=True),
            # Stage 5
            simple_block(256, 512, use_se=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

model = OCTNet(num_classes=8).to(device)
```


```python
inputs = torch.randn(1, 3, 224, 224).float()
flops, params = profile(model, inputs=(inputs, ))
print('FLOPs: {:.2f} G'.format(flops / 1e9))
print('Params: {:.2f} M'.format(params / 1e6))
```

## 训练配置


```python
criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=3e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-6
)

scaler = torch.amp.GradScaler('cuda')
```

## 训练与验证循环


```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):

    best_acc = 0.0
    best_wts = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\n=========== Epoch {epoch+1}/{num_epochs} ===========')

        for phase in ['train', 'val']:
            model.train() if phase=='train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(dataloaders[phase], desc=f"{phase} {epoch+1}")

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 混合精度
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                scheduler.step()
            else:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = model.state_dict()

    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history

model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=20)
```

## 结果可视化与测试集评估


```python
# 绘制 Loss 和 Accuracy 曲线
def plot_training_history(history):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_history(history)
```


```python
# 测试集评估与混淆矩阵
def evaluate_test_set(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    # 1. 预测过程
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 2. 生成分类报告 
    print("分类报告 (Classification Report):")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # 3. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 4. 计算总准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f'Test Set Global Accuracy: {accuracy:.4f}')

evaluate_test_set(model, dataloaders['test'], class_names)
```