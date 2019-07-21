# poem in rhyme - 押韵古诗

这是随手写的一个炼丹demo，使用tensorflow(2.0, nightly ,GPU)编写

用tf的原因很简单，因为不明原因我的pyTorch跟MXNet都炸掉了

## Apologize

Since most of the potential users are Chinese, Here I will mainly provide Chinese rather than English to explain what I have done.

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

