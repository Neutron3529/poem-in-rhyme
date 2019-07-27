import numpy as np
import mxnet as mx
import pickle
try:#预处理&&文件保存
  text_as_int,vocab_as_pyint,idx2char=np.load('t2i.npy',allow_pickle=True)
  with open('c2f.pic','rb') as f: char2f=pickle.load(f)
except:
  path_to_file="D:\\pen\\(torch)\\char-rnn-master\\chinese\\全唐诗-utf8-wash.txt"
  text = open(path_to_file, 'rb').read().decode(encoding='utf8')
  print ('Length of text: {} characters'.format(len(text)))
  vocab = sorted(set(text))
  print ('{} unique characters'.format(len(vocab)))
  #制作char<->idx的映射
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)
  #启用拼音（这里使用pypinyin，目的是将拼音分成三部分，1为声母（可以为空），2为韵母，3为声调）
  from pypinyin import pinyin
  def myPinYin(char,style):
    try: 
      return pinyin(char,style=style,errors=lambda x: [''])[0][0]
    except IndexError:
      return ''
  uniqS=sorted(set(myPinYin(char,3) for char in vocab))
  uniqY=sorted(set(myPinYin(char,5) for char in vocab))
  uniqY+=['?i','?hi']
  s2idx={s:i for i,s in enumerate(uniqS)}
  y2idx={y:i for i,y in enumerate(uniqY)}
  vocab_size   = len(vocab)
  shengMu_size = len(uniqS)
  yunMu_size   = len(uniqY)
  tone_size    = 5
  def processPinYin(char):
    #简单区分声母与韵母
    py=myPinYin(char,8)
    shengMu=s2idx[myPinYin(char,3)]
    tone=int('0'+''.join(i for i in filter(str.isdigit, myPinYin(char,9))))
    yunMu=y2idx[myPinYin(char,5)]
    #对zhi chi shi zi ci si以及ri进行处理，这几个与其他的i相比并不押韵
    try:
      if py[0] in set(('z','x','c')) and 'i' in py[1:3]:
        if py[1]=='i': yunMu=y2idx['?i']
        else         : yunMu=y2idx['?hi']
      if py=='ri'    : yunMu=y2idx['?hi']
    except:pass
    return [char2idx[char],shengMu,yunMu,tone]
  char2f={char:processPinYin(char) for char in vocab}
  text_as_int = np.array([char2f[c] for c in text],dtype='int32')
  vocab_as_pyint=np.array([char2f[c] for c in vocab],dtype='int32')[:,1:]
  del char2idx
  with open('c2f.pic','wb') as f: pickle.dump(char2f,f)
  np.save('t2i',[text_as_int,vocab_as_pyint,idx2char])

text_as_int=mx.nd.array(text_as_int,dtype='int32')
vocab_as_pyint=mx.nd.array(vocab_as_pyint,dtype='int32')
# The maximum length sentence we want for a single input in characters
seq_length = 300
examples_per_epoch = len(text_as_int)//seq_length
BATCH_SIZE = 32

# The embedding dimension
embedding_dim = 256

pyembedding_dim = 16

# Number of RNN units
rnn_units = 1024
"""
A example of iter:
class g:
 def __init__(self,x):
  self.x=x
  self.reset()
 def __iter__(self):
  yield self.x
  yield self.x+1
  self.x+=2
  return
 def reset(self):
  self.x-=1
>>> a=g(1)
>>> [i for i in a]
[1, 2]
>>> [i for i in a]
[3, 4]
>>> [i for i in a]
[5, 6]
class myIter:#(会扔掉最后一个batch的内容（随机分布在模型中），同时，seq的开始与结束部分可能不完整)
  def __init__(self,seq_len,data,discard_len,batch_size=BATCH_SIZE):
    self.seq_len=seq_len+1#will 'eat' 1 extra data.
    self.data=data
    self.data_len=data.shape[0]
    self.take=mx.nd.arange(self.seq_len-1,dtype='int32').reshape((1,-1))#-1因为不能读答案……
    self.discard=mx.nd.arange(discard_len+1,self.seq_len,dtype='int32').reshape((1,-1))#会额外丢弃若干初始label以防止nn拟合不同Batch的衔接处导致五言诗便三言诗……
    self.maxstep=(self.data_len//self.seq_len)//batch_size
    self.maxstart=self.data_len-self.maxstep
    self.batch_size=batch_size
  def __iter__(self):
    self.reset()
    for u in self.idx:
      yield self.data.take(u+self.take), self.data.take(u+self.discard)
    return
  def reset(self):
    self.idx=mx.nd.array(np.random.permutation(self.batch_size*self.maxstep),dtype='int32').reshape((self.maxstep,self.batch_size,1))*self.seq_len+np.random.randint(self.maxstart)
"""
class myIter:#前面那个版本有问题，这个版本用wrap修正了循环问题，现在程序不会丢弃任何batch
  def __init__(self,seq_len,data,discard_len,batch_size=BATCH_SIZE,ctx=mx.gpu()):
    self.seq_len=seq_len+1#will 'eat' 1 extra data.
    self.data=data.as_in_context(ctx)
    self.data_len=data.shape[0]
    self.take=mx.nd.arange(self.seq_len-1,dtype='int32').reshape((-1,1)).as_in_context(ctx)#-1因为不能读答案……
    self.discard=mx.nd.arange(discard_len+1,self.seq_len,dtype='int32').reshape((-1,1)).as_in_context(ctx)#会额外丢弃若干初始label以防止nn拟合不同Batch的衔接处导致五言诗便三言诗……
    self.batch_actual_size=self.seq_len*batch_size
    self.maxstep=self.data_len//self.batch_actual_size
    self.batch_size=batch_size
    self.x=0
    self.ctx=ctx
  def __iter__(self):
    self.reset()
    for u in self.idx:
      yield self.data.take(u+self.take,mode='wrap'), self.data.take(u+self.discard,mode='wrap')
    return
  def reset(self):
    if self.x>0 : self.x-=self.data_len
    self.idx=mx.nd.array(np.random.permutation(self.batch_size*self.maxstep),dtype='int32',ctx=self.ctx).reshape((self.maxstep,1,-1))*self.seq_len+self.x
    self.x+=self.batch_actual_size

discard_length=0

ctx=mx.gpu()
it=myIter(seq_length,text_as_int,discard_length,BATCH_SIZE,ctx=ctx)
if True:
  from pypinyin import pinyin
  def myPinYin(char,style,heteronym=False):
    try: 
      if heteronym:
        return pinyin(char,style=style,errors=lambda x: [''],heteronym=True)[0]
      else:
        return pinyin(char,style=style,errors=lambda x: [''])[0][0]
    except IndexError:
      return ''
  uniqS=sorted(set(myPinYin(char,3) for char in idx2char))
  uniqY=sorted(set(myPinYin(char,5) for char in idx2char))
  uniqY+=['?i','?hi']
  s2idx={s:i for i,s in enumerate(uniqS)}
  y2idx={y:i for i,y in enumerate(uniqY)}
  vocab_size   = len(idx2char)
  shengMu_size = len(uniqS)
  yunMu_size   = len(uniqY)
  tone_size    = 5

'''
  ST=mx.nd.stack(*([sum(set(myPinYin(char,3,True))&i for i in SGroup if H in i) for H in uniqS] for char in idx2char))
  del SGroup,YGroup,TGroup,s2idx,y2idx,t2idx,uniqS,uniqY
'''
SGroup=(('b','p','m','f'),('d','t','n','l'),('zh','ch','sh','r'),('g','k','h'),('j','q','x'),('z','c','s'))#https://baike.baidu.com/item/声母
YGroup=(('a','ua','ia'),('o','uo'),('e',),('ie','ve'),('?hi','?i'),('er',),('i',),('ei','uei'),('ai','uai'),('u',),('v',),('ou','iou'),('ao','iao'),('an','ian','uan','van'),('en','in','uen','vn'),('ang','uang','iang'),('eng','ing','ueng'),('ong','iong'))#https://baike.baidu.com/item/押韵
TGroup=((1,2),(3,4))#阴平阳平，上声去声
t2idx=[i for i in range(5)]
SMOH=mx.nd.one_hot(vocab_as_pyint[:,0],shengMu_size)
YMOH=mx.nd.one_hot(vocab_as_pyint[:,1],yunMu_size)
TOH=mx.nd.one_hot(vocab_as_pyint[:,2],tone_size)
SE=np.eye(shengMu_size,shengMu_size)
YE=np.eye(yunMu_size,yunMu_size)
TE=np.eye(tone_size,tone_size)
if True:#处理押韵，将同类声母/韵母/声调算成一类，以减少类似XXXXXX天，XXXXXX天这样的重复韵字出现的概率。
 for x in SGroup:
  SE[(np.array([s2idx[x] for x in x]).reshape(-1,1),np.array([s2idx[x] for x in x]))]=1./len(x)
 for x in YGroup:
  YE[(np.array([y2idx[x] for x in x]).reshape(-1,1),np.array([y2idx[x] for x in x]))]=1./len(x)
 for x in TGroup:
  TE[(np.array([t2idx[x] for x in x]).reshape(-1,1),np.array([t2idx[x] for x in x]))]=1./len(x)
 ST=mx.nd.dot(SMOH,mx.nd.array(SE))
 YT=mx.nd.dot(YMOH,mx.nd.array(YE))
 TT=mx.nd.dot(TOH,mx.nd.array(TE))

del SMOH,YMOH,TOH,SE,YE,TE,SGroup,YGroup,TGroup,s2idx,y2idx,t2idx,uniqS,uniqY


def build_model(vocab_size, embedding_dim, rnn_units, batch_size,seq_len=seq_length,drop=False,_prefix='',keep_state=False):
  #rnn_units变量已经硬编码进模型了，于是这个参数现在并没有什么卵用……
  #STx=mx.sym.var('ST',shape=ST.shape)
  #YTx=mx.sym.var('YT',shape=YT.shape)
  #TTx=mx.sym.var('TT',shape=TT.shape)
  data  = mx.sym.var('data',shape=(batch_size,seq_len,4),dtype='int32')
  #aux_lstm10,aux_lstm11,aux_lstm20,aux_lstm21,aux_lstm30,aux_lstm31=mx.sym.var('la10',shape=(1,208),init=mx.init.Xavier()).broadcast_to((batch_size,208)),mx.sym.var('la11',shape=(1,208),init=mx.init.Xavier()).broadcast_to((batch_size,208)),mx.sym.var('la20',shape=(1,512),init=mx.init.Xavier()).broadcast_to((batch_size,512)),mx.sym.var('la21',shape=(1,512),init=mx.init.Xavier()).broadcast_to((batch_size,512)),mx.sym.var('la30',shape=(1,1024),init=mx.init.Xavier()).broadcast_to((batch_size,1024)),mx.sym.var('la31',shape=(1,1024),init=mx.init.Xavier()).broadcast_to((batch_size,1024))
  #ew1,ew2,ew3,ew4=mx.sym.var('ew1',shape=(vocab_size,embedding_dim),init=mx.init.Xavier()),mx.sym.var('ew2',shape=(shengMu_size,pyembedding_dim),init=mx.init.Xavier()),mx.sym.var('ew3',shape=(yunMu_size,pyembedding_dim),init=mx.init.Xavier()),mx.sym.var('ew4',shape=(tone_size,pyembedding_dim),init=mx.init.Xavier())
  slice_0,slice_1,slice_2,slice_3 = mx.sym.split(mx.sym.BlockGrad(data),4,axis=2,squeeze_axis=True)
  #se0     = mx.sym.Embedding(data=slice_0, weight=ew1, input_dim=vocab_size, output_dim=embedding_dim)
  #se1     = mx.sym.Embedding(data=slice_1, weight=ew2, input_dim=shengMu_size, output_dim=pyembedding_dim)
  #se2     = mx.sym.Embedding(data=slice_2, weight=ew3, input_dim=yunMu_size, output_dim=pyembedding_dim)
  #se3     = mx.sym.Embedding(data=slice_3, weight=ew4, input_dim=tone_size, output_dim=pyembedding_dim)
  se0     = mx.sym.Embedding(data=slice_0, input_dim=vocab_size, output_dim=embedding_dim)
  se1     = mx.sym.Embedding(data=slice_1, input_dim=shengMu_size, output_dim=pyembedding_dim)
  se2     = mx.sym.Embedding(data=slice_2, input_dim=yunMu_size, output_dim=pyembedding_dim)
  se3     = mx.sym.Embedding(data=slice_3, input_dim=tone_size, output_dim=pyembedding_dim)
  if drop:
    combined_1= mx.sym.Dropout(mx.sym.Concat(se0,se1,se2,se3, dim=-1),p=0.5)
  else:
    combined_1= mx.sym.Concat(se0,se1,se2,se3, dim=-1)
  st00,st10,st20,st01,st11,st21=mx.sym.var('st00',shape=(1,batch_size,208),dtype='float32'),mx.sym.var('st10',shape=(1,batch_size,512),dtype='float32'),mx.sym.var('st20',shape=(1,batch_size,1024),dtype='float32'),mx.sym.var('st01',shape=(1,batch_size,208),dtype='float32'),mx.sym.var('st11',shape=(1,batch_size,512),dtype='float32'),mx.sym.var('st21',shape=(1,batch_size,1024),dtype='float32')
  lstm_1,[st02,st03]= mx.gluon.rnn.LSTM(208,layout='TNC',input_size=304)(combined_1,[st00,st01])
  combined_2 = mx.sym.Concat(combined_1,lstm_1, dim=-1)
  lstm_2,[st12,st13]= mx.gluon.rnn.LSTM(512,layout='TNC',input_size=512)(combined_2,[st10,st11])
  combined_3 = mx.sym.Concat(combined_2,lstm_2, dim=-1)
  lstm_3,[st22,st23]= mx.gluon.rnn.LSTM(1024,layout='TNC',input_size=1024)(combined_3,[st20,st21])
  #out0    = mx.sym.softmax(mx.sym.FullyConnected(lstm_3,num_hidden=vocab_size))
  denseCore,denseBias=mx.sym.var(_prefix+'dense',shape=(1024,vocab_size),init=mx.init.Xavier()),mx.sym.var(_prefix+'dense_bias',shape=(1,1,vocab_size),init=mx.init.Xavier())
  out0   = mx.sym.broadcast_add(mx.sym.dot(lstm_3,denseCore),denseBias).softmax().clip(1e-15,1)
#  var_list=(ew1,ew2,ew3,ew4,denseCore,denseBias)
#  return mx.sym.MakeLoss(mx.sym.add_n(*(i.abs().mean() for i in var_list))+mx.sym.add_n(*(i.square().mean() for i in var_list))-out0.pick(lslice_0,2).log().mean()-out1.pick(lslice_1,2).log().mean()-out2.pick(lslice_2,2).log().mean()-out3.pick(lslice_3,2).log().mean())
  return mx.gluon.SymbolBlock([out0,st02,st03,st12,st13,st22,st23],[data,st00,st01,st10,st11,st20,st21])

#x.pick(mx.nd.array([[1,1,1],[2,2,2],[0,0,0]]),1)

with mx.gluon.nn.Block().name_scope() as b:
  net=build_model(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE,#seq_len=400,
  drop=False,_prefix=b._block._prefix)
  net.hybridize()
  net._prefix=b._block._prefix
  def _begin_state(batch_size=BATCH_SIZE,ctx=ctx):
    return [mx.nd.zeros((1,batch_size,i),ctx=ctx) for i in (208,208,512,512,1024,1024)]
  net.begin_state=_begin_state
  del _begin_state
  #net.collect_params().load('param.test',restore_prefix=b._block._prefix)

STg,YTg,TTg=ST.as_in_context(ctx),YT.as_in_context(ctx),TT.as_in_context(ctx)

net.initialize(mx.init.Uniform(scale=0.01), ctx=ctx,force_reinit=True)
trainer = mx.gluon.Trainer(
    params=net.collect_params(),
    optimizer='nadam',
    optimizer_params={'clip_gradient':2},
)
class myloss(mx.gluon.loss.Loss):
  def __init__(self,STs,YTs,TTs, weight=None, batch_axis=0, **kwargs):
    self.ST=STs
    self.YT=YTs
    self.TT=TTs
#    self.discard_length=discard_length
#    self.seq_len=seq_len
    super(myloss, self).__init__(weight, batch_axis, **kwargs)
  def hybrid_forward(self,F,out0,label, sample_weight=None):
    lslice_0,lslice_1,lslice_2,lslice_3 = label.split(4,axis=2,squeeze_axis=True)
    j=net.collect_params()
    out1    = mx.nd.dot(out0,self.ST)
    out2    = mx.nd.dot(out0,self.YT)
    out3    = mx.nd.dot(out0,self.TT)
    return (F.add_n(*(j[i].data().abs().mean() for i in j))+F.add_n(*(j[i].data().square().mean() for i in j)))-out0.pick(lslice_0,2).log().mean(1)-out1.pick(lslice_1,2).log().mean(1)*.5-out2.pick(lslice_2,2).log().mean(1)*2-out3.pick(lslice_3,2).log().mean(1)*4
    #return -out0.pick(lslice_0,2).log().mean(1)-out1.pick(lslice_1,2).log().mean(1)-out2.pick(lslice_2,2).log().mean(1)-out3.pick(lslice_3,2).log().mean(1)

loss_function=myloss(STg,YTg,TTg)
metric = mx.metric.Accuracy()
#loss_function(out0,out1,out2,out3, label.as_in_context(ctx))
epochs = 2
from time import time
from tqdm import tqdm
from mxboard import SummaryWriter
sw = SummaryWriter(logdir='./logs', flush_secs=5)
#使用mxboard
#tensorboard --logdir=./logs --host=127.0.0.1 --port=7000
#http://localhost:7000
#grads = [i.grad().asnumpy() for i in net.collect_params().values()]
newval = [i.data().asnumpy() for i in net.collect_params().values()]

for epoch in range(epochs):
    t=time()
    total_loss = 0
    i=0
    with tqdm(total=it.maxstep,ncols=130) as pbar:
      pbar.set_description('batch %i' % i)
      i+=1
      states=net.begin_state()
      for data, label in it:
        with mx.autograd.record():
            #out0= net(data).slice(begin=(None,discard_length,None),end=(None,seq_length,None))
#            out0,*_= net(data,*states)
            out0,*_= net(data,*states)
            loss = loss_function(out0, label)
        loss.backward()
        trainer.step(BATCH_SIZE)
        _loss=loss.asnumpy().mean()
        pbar.set_postfix(loss=str(_loss))
        pbar.update()
        metric.update(label[:,:,0].reshape(-3,1), out0.reshape((-3,-1)))
        sw.add_scalar(tag='cross_entropy', value=_loss, global_step=epoch)
    grads = [i.grad().asnumpy() for i in net.collect_params().values()]
    oldval = newval
    newval = [i.data().asnumpy() for i in net.collect_params().values()]
    diffval = [a-b for a,b in zip(newval,oldval)]
    for i, name in enumerate(net.collect_params().keys()):
      sw.add_histogram(tag='grad'+name, values=grads[i], global_step=epoch, bins=1000)
      sw.add_histogram(tag='val'+name, values=newval[i], global_step=epoch, bins=1000)
      sw.add_histogram(tag='diff'+name, values=diffval[i], global_step=epoch, bins=1000)
    sw.add_scalar(tag=name, value=acc, global_step=epoch)
    name, acc = metric.get()
    print('After epoch {}: {} = {}({}s),\n    final batch loss is {}'.format(epoch + 1, name, acc,time()-t,loss.asnumpy().mean()))
    metric.reset()
    t=time()

'''
net.collect_params().save('test2.param',net.prefix)
net.collect_params().load('test2.param',restore_prefix=b._block._prefix)#可以在这里指定ctx=ctx
with mx.gluon.Block().name_scope() as b:
  net_p=build_model(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE,#seq_len=400,
  drop=False,_prefix=b._block._prefix)#,keep_state=True)
  net_p._prefix=b._block._prefix
  net_p.collect_params().load('test2.param',restore_prefix=b._block._prefix)
'''

def generate_text(net, start_string,temperature = 1.0,num_generate = 1000,return_str=False,ctx=mx.gpu()):
  input_eval = mx.nd.array([char2f[s] for s in start_string],ctx=ctx).expand_dims(1)
  text_generated = []
  print(start_string,end='')
  # Here batch size == 1
  states=net.begin_state(1,ctx)
  for i in range(num_generate):
      predictions,*states = net(input_eval,*states)
      # remove the batch dimension
      predictions=predictions[-1,:,:]
      predictions = (mx.nd.log(predictions/((1-predictions)+1e-15))/ temperature).softmax()
      # using a categorical distribution to predict the word returned by the model
      predicted_id = mx.nd.random.multinomial(predictions).asnumpy()[0]
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = mx.nd.array([char2f[idx2char[predicted_id]]],ctx).expand_dims(0)
      text_generated.append(idx2char[predicted_id])
      print(idx2char[predicted_id],end='')
  print()
  if return_str:
    return (start_string + ''.join(text_generated))

generate_text(net, "生命的意义 ",.5,100,return_str=True)
