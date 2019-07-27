# poem in rhyme - 押韵古诗

这是随手写的一个炼丹demo，使用tensorflow(2.0, nightly ,GPU)编写

用tf的原因很简单，因为不明原因我的pyTorch跟MXNet都炸掉了

V0.3，终于找到MXNet炸掉的原因，用N天修好&&熟悉了MXNet的用法，之后把这个炼丹demo迁移到了MXNet
新模型在训练时候做到了1.25X faster
至于预测？？tensorflow的预测速度就是个渣渣

## Apologize

Since most of the potential users are Chinese, Here I will mainly provide Chinese rather than English to explain what I have done.

## Requirements

Python 3.6

```
python -m pip install numpy
python -m pip install tf-nightly-gpu-2.0-preview
python -m pip install pypinyin
```

## Usage（ver 0.0）

自己打开.py文件，逐行往python命令行里面贴，在这个过程中可能会报错。由于我能力有限，你或许只能根据报错内容自行修改程序……

值得一提的是，这个.py文件是对命令行友好的，你可以将其中任意整段代码复制粘贴到命令行而不必担心复制粘贴带来的不一致性

## 样例（ver 0.1-0.3)

已经有了初步续写古诗的能力，在开启复读机模式（……）之后，李白给出了一个神奇的输出：
```
>>> print(generate_text(model, start_string=u"静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n",temperature =.5))
……

静夜思 李白

床前明月光，疑是地上霜。
举头望明月，低头思故乡。

相思不相见，深夜空独忘。
昔时欲来去，秋风仍自伤。
>>> _=(generate_text(model, "生命的意义 ",.5,100));del _#ver 0.2
生命的意义 白居易

何处堪归老，相逢不自来。
自无名是累，不是死生胎。

天将离白发，真见喜并开。
自是浮名改，无因用不裁。
>>> generate_text(net, "静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n",.5,100)#ver 0.3
静夜思 李白

床前明月光，疑是地上霜。
举头望明月，低头思故乡。

楚云带霜积，秋草白猿翔。
却忆旧游日，相思寒夜芳。
>>> generate_text(net, "静夜思 李白\r\n\r\n床前明月光，疑是地上霜。\r\n举头望明月，低头思故乡。\r\n\r\n",.5,100,return_str=True)
静夜思 李白

床前明月光，疑是地上霜。
举头望明月，低头思故乡。

美人摘芳草，婉娈扣清商。
今日花开院，蜉蝣意还长。
```
# MXNet版

## 说明

使用了mxboard, tqdm进行记录，如果不想加入相应的import语句，请自行删除……
使用pickle保存了预处理的数据，若觉得pickle保存的数据有问题……可以选择从完整脚本自行生成

## 使用方法

解压[poem-in-rhyme-MXNet-minimal.7z](https://github.com/Neutron3529/poem-in-rhyme/releases/download/0/poem-in-rhyme-MXNet-minimal.7z)
在解压目录下运行`python poem-in-rhyme-MXNet-minimal.py`
如果希望生成其他古诗（比如续写《静夜思》），可以改最后一句generate_text函数里面的参数
有时候运行结果并不能令人满意……如果你觉得古诗不通顺，可以多执行几次generate_text函数，总有一首诗是你想要的……

# Tensorflow版

## 原理

在最开始，用pyPinYin获取汉字的声母、韵母跟音调，获取不到的留空。

在之后，使用老掉牙的LSTM搭模型（或许未来会换成BART甚至XLNet）

在预测结果上，我做了一点小花招：

```
#首先定义：
delim0=vocab_size          #在这之后是对声母的预测，而[:,:,:delim0]是针对字符的预测结果
delim1=delim0+shengMu_size #在这之后是对韵母的预测
delim2=delim1+yunMu_size   #在这之后是对声调的预测
delim3=delim2+tone_size    #[:,:,delim3]是损失函数
#在神经网络中，另有一个小网络验证我们的预测结果是否与预测的声母韵母声调相对应
#这样，LSTM就会捕捉到押韵这一特征。
#目前模型的参数并不完善，接下来我还会继续调整loss函数的各个参数，以获得更好的预测结果
```

模型会训练10个EPOCH，并在训练结束之后在当前目录下新建training_checkpoints文件夹，在其中存储相应参数，请确保剩余磁盘空间足够。

# Notice

本文的代码有一半以上都是从 https://tensorflow.google.cn/beta/tutorials/text/text_generation 获取的

