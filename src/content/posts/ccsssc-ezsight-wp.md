---
title: 2025年软件系统安全赛 ezsight 部分思路
published: 2025-03-23
description: ''
image: ''
tags: ['AI','CTF','Python','Bkcrack']
category: 'CTF Writeup'
draft: false 
lang: ''
---

# 吐槽

我这个华南赛区二号赛场8:30开赛恨不得12点网络才恢复，开局及落后，一道题也没写出来，这是赛后写的

![image-20250323223414379](https://gu-blog-img.gmoe.cc/20250323223414436.png)

# 题目分析

拿到压缩包有三个文件，里面有一个加密的`workspace.zip`压缩包![image-20250323223828079](https://gu-blog-img.gmoe.cc/20250323223828133.png)

这时候打开加密压缩包发现 `公告.txt` 和给的题目压缩包内的`公告.txt`是一个文件

这时候可以通过`明文攻击`来实现破解压缩包

------

## 什么是明文攻击

ZIP 文件里的内容可以用一种叫 ZipCrypto 的加密方式保护，靠密码生成一串随机字节，跟文件内容“混在一起”变成加密后的数据。它的核心是个由三个数字组成的小机器，先用密码启动，然后边加密边更新。但这方法有个弱点：如果有人知道加密后的内容和至少 12 个字节的原文，就能破解这个小机器的内部状态。掌握了状态，就能解开所有用同一密码加密的内容，还能试着猜密码，难度大概是“字符种类数 × 密码长度 - 6”。简单说，就是不够安全，容易被攻破。

------

# 解题思路(部分)

## 1.通过`bkcrack`工具对压缩包进行攻击

刚好`公告.txt`大小为`694KB`可以使用该工具对压缩包进行 `明文攻击`

通过

```bash
 bkcrack -C .\workspace.zip -c 公告.txt -p 公告.txt
```

可以获得3个KEY

```bash
Keys: ffe9e9e9 d65f814a f3c468c9
```

![image-20250323230333459](https://gu-blog-img.gmoe.cc/20250323230333511.png)

## 2.通过指令生成一个新的压缩包

拿到3个Key后通过指令生成一个新的压缩包，因为不知道密码多少位逆向回去也特别费时间

```bash
bkcrack -C  workspace.zip -k ffe9e9e9 d65f814a f3c468c9 -U workspace_new.zip gugugu
```

上面这段代码就通过Key生成了个新的压缩包，密码为`gugugu`这样就能拿到压缩包内的文件

## 3.查看解密后压缩包内容

压缩包内有一个`password.pt`文件和一个`flag`文件夹下有一堆.bmp图片文件

![image-20250323230854481](https://gu-blog-img.gmoe.cc/20250323230854538.png)

结合公告内容

```
各位员工：

为了提升公司的安全管理水平，从即日起，我司将引入AI技术对通行密码进行管理。相关的密码图片内容已整理并放入压缩包中，压缩包的密码将由各部门负责组织发放，请大家留意部门通知。

请注意公司内部AI模型的使用规范：
1.除最后一层外与池化层外其他隐藏层输出均需要通过激活函数
2.至少需要通过两次池化层
3.注意隐藏之间输出数据格式的匹配，必要时对数据张量进行重塑
4.为保证模型准确性，输入图片应转换为灰度图


感谢大家的配合与支持。如有疑问，请随时与人事部联系。

此致
```

应该是写一个代码调用该模型然后输出密码，但是我这后面怎么试flag都不对，等大佬的WP吧

## 4.不可行代码

通过[Netron](https://netron.app/)工具读取这个.pt模型文件内容发现模型为这样的

![image-20250323231320393](https://gu-blog-img.gmoe.cc/20250323231320441.png)

刚好和公告是对的上的

```
1.除最后一层外与池化层外其他隐藏层输出均需要通过激活函数
2.至少需要通过两次池化层
3.注意隐藏之间输出数据格式的匹配，必要时对数据张量进行重塑
4.为保证模型准确性，输入图片应转换为灰度图
```

以下内容来自GROK解释

------

### 1. 模型概述

SimpleCNN 是一个典型的卷积神经网络，设计用于处理二维输入数据（例如图像）。从结构上看，它包含以下层：

- 两个卷积层（Conv2d）
- 一个最大池化层（MaxPool2d）
- 两个全连接层（Linear）

这种架构常用于图像分类任务，例如手写数字识别（MNIST 数据集）或简单的图像分类问题。

------

### 2. 逐层解释

#### **第一层：Conv2d**

- 参数
  - weight (32x1x3x3)：表示有 32 个卷积核（filters），输入通道数为 1（例如灰度图像），每个卷积核的大小是 3x3。
  - bias (32)：每个卷积核有一个偏置项，总共 32 个。
- 作用
  - 这一层对输入图像进行卷积操作，提取低级特征（如边缘、纹理等）。
  - 输入通道为 1，可能是因为输入是灰度图像（例如 MNIST 数据集的 28x28 灰度图像）。
  - 输出通道为 32，意味着这一层会生成 32 个特征图（feature maps）。

#### **第二层：MaxPool2d**

- **参数**：没有显示具体参数，但通常 MaxPool2d 会有池化窗口大小（例如 2x2）和步幅（stride）。
- 作用
  - 最大池化层用于下采样（downsampling），减少特征图的空间维度（宽和高），从而降低计算量并提取更显著的特征。
  - 假设池化窗口是 2x2，步幅为 2，那么特征图的宽和高会减半。例如，如果输入特征图是 28x28，经过池化后会变成 14x14。

#### **第三层：Conv2d**

- 参数
  - weight (64x32x3x3)：表示有 64 个卷积核，输入通道数为 32（来自上一层的输出），卷积核大小为 3x3。
  - bias (64)：每个卷积核有一个偏置项，总共 64 个。
- 作用
  - 这一层继续提取更高级的特征，输入是上一层的 32 个特征图，输出 64 个特征图。
  - 经过这一层，特征图的空间维度可能会进一步减小（取决于是否有 padding）。如果没有 padding，假设上一层输出是 14x14，经过 3x3 卷积后会变成 12x12（因为 14 - 3 + 1 = 12）。

#### **第四层：Linear**

- 参数
  - weight (128x3136)：表示全连接层有 128 个神经元，输入维度是 3136。
  - bias (128)：每个神经元有一个偏置项，总共 128 个。
- 作用
  - 在全连接层之前，特征图需要被展平（flatten）为一维向量。
  - 输入维度 3136 是如何计算的呢？假设上一层输出的特征图是 12x12x64（64 个通道，空间维度 12x12），那么展平后就是 12 * 12 * 64 = 9216。但这里是 3136，说明可能有额外的池化层（未显示）或空间维度计算不同。一种可能是特征图被进一步下采样（例如通过另一个池化层变成 7x7），那么 7 * 7 * 64 = 3136。
  - 这一层将特征映射到 128 维的向量，用于后续分类。

#### **第五层：Linear**

- 参数
  - weight (10x128)：表示全连接层有 10 个神经元，输入维度是 128（上一层的输出）。
  - bias (10)：每个神经元有一个偏置项，总共 10 个。
- 作用
  - 这是输出层，10 个神经元通常对应 10 个类别（例如 MNIST 数据集的 0-9 数字分类）。
  - 这一层将 128 维的特征向量映射到 10 维的输出，通常会通过 softmax 激活函数转换为概率分布，表示每个类别的预测概率。

------

### 3. 模型的整体流程

假设输入是一张 28x28 的灰度图像（例如 MNIST 数据集的图像），以下是数据通过模型的流动过程：

1. **输入**：28x28x1（宽 x 高 x 通道数）。
2. **第一层 Conv2d**：经过 32 个 3x3 卷积核，输出 28x28x32（如果有 padding="same"）或 26x26x32（无 padding）。
3. **MaxPool2d**：假设池化窗口是 2x2，输出 14x14x32（如果输入是 28x28）或 13x13x32（如果输入是 26x26）。
4. **第二层 Conv2d**：输入 14x14x32，经过 64 个 3x3 卷积核，输出 12x12x64（无 padding）。
5. **展平**：12x12x64 = 9216（可能有额外的池化层未显示，最终展平为 3136）。
6. **第一层 Linear**：输入 3136，输出 128。
7. **第二层 Linear**：输入 128，输出 10（对应 10 个类别）。

------

### 4. 可能的用途

这个模型的结构非常适合小型图像分类任务，例如：

- **MNIST 手写数字识别**：输入是 28x28 的灰度图像，输出是 0-9 的 10 个类别。
- **CIFAR-10 数据集**：如果输入通道改为 3（RGB 图像），也可以用于 CIFAR-10 的分类任务，但可能需要调整模型结构以适应 32x32x3 的输入。

------

通过这些让AI写一个调用该模型的代码识别flag/*.bmp文件输出后放到`flag.py`里面后不可行，所以等大佬的wp吧

## 5.调用模型识别数字

代码如下

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.serialization import safe_globals
import os

# 根据模型实际结构定义SimpleCNN类
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积层 (1 -> 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二层卷积层 (32 -> 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # 第二层卷积 + 激活 + 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # 重塑张量
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
def load_model(model_path):
    # 添加SimpleCNN到安全全局变量列表中
    safe_globals_list = [SimpleCNN]
    
    # 使用safe_globals上下文管理器来安全加载模型
    with safe_globals(safe_globals_list):
        model = torch.load(model_path, weights_only=False)
    
    model.eval()
    return model

# 对现有图像进行预测
def predict_images(model_path, image_paths):
    # 加载模型
    model = load_model(model_path)
    
    results = []
    for img_path in image_paths:
        try:
            # 加载图像并转换为灰度图
            img = Image.open(img_path).convert('L')
            # 调整图像大小为28x28
            img = img.resize((28, 28))
            # 转换为张量
            img_tensor = torch.tensor(np.array(img)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
            
            # 进行预测
            with torch.no_grad():
                output = model(img_tensor)
                prob = F.softmax(output, dim=1)
                pred_class = torch.argmax(prob, dim=1).item()
                confidence = prob[0][pred_class].item()
            
            results.append({
                'image': os.path.basename(img_path),
                'prediction': pred_class,
                'confidence': confidence
            })
            
            print(f"图像 {os.path.basename(img_path)} 预测为: {pred_class}, 置信度: {confidence:.4f}")
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    return results

# 主函数
def main():
    model_path = "./password.pt"
    
    # 对已有的0-13.bmp图像进行预测
    image_paths = [f"./flag/{i}.bmp" for i in range(14)]
    print("分析数字图像...")
    predictions = predict_images(model_path, image_paths)
    
    # 打印预测结果摘要
    print("\n预测结果摘要:")
    predicted_digits = ""
    for p in predictions:
        predicted_digits += str(p['prediction'])
    print(f"数字序列: {predicted_digits}")
    
    # 尝试将数字序列转换为ASCII字符
    try:
        ascii_text = ""
        for i in range(0, len(predicted_digits), 2):
            if i+1 < len(predicted_digits):
                char_code = int(predicted_digits[i:i+2])
                if 32 <= char_code <= 126:  # 可打印ASCII范围
                    ascii_text += chr(char_code)
        if ascii_text:
            print(f"可能的ASCII文本: {ascii_text}")
    except Exception as e:
        print(f"转换ASCII时出错: {e}")

if __name__ == "__main__":
    main()
```

运行结果如下

![image-20250323232511230](https://gu-blog-img.gmoe.cc/20250323232511313.png)

## 总结

这题标的难度`简单`，可真ez啊
