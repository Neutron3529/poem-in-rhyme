import numpy as np
import pickle
try:#预处理&&文件保存
  text_as_int=np.load('tai.npy',allow_pickle=True)
  vocab_as_pyint=np.load('vap.npy',allow_pickle=True)
  idx2char=np.load('i2c.npy',allow_pickle=True)
  with open('c2f.pic','rb') as f: char2f=pickle.load(f)
except:
  path_to_file="./全唐诗-utf8-wash.txt"
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
  np.save('tai',text_as_int)
  np.save('vap',vocab_as_pyint)
  np.save('i2c',idx2char)

import oneflow as flow
import oneflow.nn as nn
flow.backends.cudnn.enable_conv_heuristic_search_algo(False)

text_as_int=flow.from_numpy(text_as_int)
vocab_as_pyint=flow.from_numpy(vocab_as_pyint)
# The maximum length sentence we want for a single input in characters
seq_length = 300
examples_per_epoch = len(text_as_int)//seq_length
BATCH_SIZE = 28

# The embedding dimension
embedding_dim = 256

pyembedding_dim = 16

# Number of RNN units
rnn_units = 1024
generator = flow.Generator()
generator.manual_seed(0)
class myIter(flow.utils.data.IterableDataset):#前面那个版本有问题，这个版本用wrap修正了循环问题，现在程序不会丢弃任何batch
  def __init__(self,seq_len,data,discard_len,batch_size=BATCH_SIZE):
    super(myIter).__init__()
    self.seq_len=seq_len#will 'eat' 1 extra data.
    self.data=data
    self.data_len=data.shape[0]
    # self.take=mx.nd.arange(self.seq_len-1,dtype='int32').reshape((-1,1)).as_in_context(ctx)#-1因为不能读答案……
    # self.discard=mx.nd.arange(discard_len+1,self.seq_len+discard_len,dtype='int32').reshape((-1,1)).as_in_context(ctx)#会额外丢弃若干初始label以防止nn拟合不同Batch的衔接处导致五言诗便三言诗……
    self.take=flow.tensor(range(self.seq_len-discard_len-1)).reshape((-1,1))#(0,self.seq_len-discard_len-1)
    self.discard_len=discard_len
    self.discard=self.take+discard_len+1#(discard_len+1,self.seq_len)
    self.batch_actual_size=self.seq_len*batch_size
    self.now_step=self.data_len//self.batch_actual_size
    self.batch_size=batch_size
    self.x=0
    self.idx=None
    self.data_shape=(self.seq_len-discard_len-1,self.batch_size,-1)
  def __iter__(self):
    self.reset()
    for i in self.idx:
      # yield self.data.take(u+self.take,mode='wrap'), self.data.take(u+self.discard,mode='wrap')
      yield (
        self.data.index_select(0,(i+self.take).reshape(-1).fmod(self.data_len)).reshape(self.data_shape),
        self.data.index_select(0,(i+self.discard).reshape((-1)).fmod(self.data_len)).reshape(self.data_shape)
      )
    return
  def reset(self):
    now_len=self.data_len-self.x
    self.now_step=now_len//self.batch_actual_size
    # self.idx=mx.nd.array(np.random.permutation(self.batch_size*self.maxstep),dtype='int32',ctx=self.ctx).reshape((self.maxstep,1,-1))*self.seq_len+self.x
    self.idx=(flow.randperm(self.batch_size*self.now_step).cast(flow.int32)*self.seq_len+self.data_len+self.x).reshape((self.now_step,1,self.batch_size))
    self.x+=self.now_step*self.batch_actual_size - self.data_len
    print("data reset, actual len="+str(self.idx.shape[0])+", curr bias="+str(self.x)+", now_step="+str(self.now_step))

discard_len=0

it=myIter(seq_length,text_as_int,discard_len,BATCH_SIZE)
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
SMOH=flow.nn.functional.one_hot(vocab_as_pyint[:,0],shengMu_size)
YMOH=flow.nn.functional.one_hot(vocab_as_pyint[:,1],yunMu_size)
TOH=flow.nn.functional.one_hot(vocab_as_pyint[:,2],tone_size)
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
 ST=flow.matmul(SMOH.cast(flow.float32),flow.from_numpy(SE).cast(flow.float32))
 YT=flow.matmul(YMOH.cast(flow.float32),flow.from_numpy(YE).cast(flow.float32))
 TT=flow.matmul(TOH.cast(flow.float32),flow.from_numpy(TE).cast(flow.float32))

del SMOH,YMOH,TOH,SE,YE,TE,SGroup,YGroup,TGroup,s2idx,y2idx,t2idx,uniqS,uniqY


# def build_model(vocab_size, embedding_dim, rnn_units, batch_size,seq_len=seq_length,drop=False,_prefix='',keep_state=False):
#   #rnn_units变量已经硬编码进模型了，于是这个参数现在并没有什么卵用……
#   #STx=mx.sym.var('ST',shape=ST.shape)
#   #YTx=mx.sym.var('YT',shape=YT.shape)
#   #TTx=mx.sym.var('TT',shape=TT.shape)
#   data  = mx.sym.var('data',shape=(batch_size,seq_len,4),dtype='int32')
#   #aux_lstm10,aux_lstm11,aux_lstm20,aux_lstm21,aux_lstm30,aux_lstm31=mx.sym.var('la10',shape=(1,208),init=mx.init.Xavier()).broadcast_to((batch_size,208)),mx.sym.var('la11',shape=(1,208),init=mx.init.Xavier()).broadcast_to((batch_size,208)),mx.sym.var('la20',shape=(1,512),init=mx.init.Xavier()).broadcast_to((batch_size,512)),mx.sym.var('la21',shape=(1,512),init=mx.init.Xavier()).broadcast_to((batch_size,512)),mx.sym.var('la30',shape=(1,1024),init=mx.init.Xavier()).broadcast_to((batch_size,1024)),mx.sym.var('la31',shape=(1,1024),init=mx.init.Xavier()).broadcast_to((batch_size,1024))
#   #ew1,ew2,ew3,ew4=mx.sym.var('ew1',shape=(vocab_size,embedding_dim),init=mx.init.Xavier()),mx.sym.var('ew2',shape=(shengMu_size,pyembedding_dim),init=mx.init.Xavier()),mx.sym.var('ew3',shape=(yunMu_size,pyembedding_dim),init=mx.init.Xavier()),mx.sym.var('ew4',shape=(tone_size,pyembedding_dim),init=mx.init.Xavier())
#   slice_0,slice_1,slice_2,slice_3 = mx.sym.split(mx.sym.BlockGrad(data),4,axis=2,squeeze_axis=True)
#   #se0     = mx.sym.Embedding(data=slice_0, weight=ew1, input_dim=vocab_size, output_dim=embedding_dim)
#   #se1     = mx.sym.Embedding(data=slice_1, weight=ew2, input_dim=shengMu_size, output_dim=pyembedding_dim)
#   #se2     = mx.sym.Embedding(data=slice_2, weight=ew3, input_dim=yunMu_size, output_dim=pyembedding_dim)
#   #se3     = mx.sym.Embedding(data=slice_3, weight=ew4, input_dim=tone_size, output_dim=pyembedding_dim)
#   se0     = mx.sym.Embedding(data=slice_0, input_dim=vocab_size, output_dim=embedding_dim)
#   se1     = mx.sym.Embedding(data=slice_1, input_dim=shengMu_size, output_dim=pyembedding_dim)
#   se2     = mx.sym.Embedding(data=slice_2, input_dim=yunMu_size, output_dim=pyembedding_dim)
#   se3     = mx.sym.Embedding(data=slice_3, input_dim=tone_size, output_dim=pyembedding_dim)
#   if drop:
#     combined_1= mx.sym.Dropout(mx.sym.Concat(se0,se1,se2,se3, dim=-1),p=0.5)
#   else:
#     combined_1= mx.sym.Concat(se0,se1,se2,se3, dim=-1)
#   st00,st10,st20,st01,st11,st21=mx.sym.var('st00',shape=(1,batch_size,208),dtype='float32'),mx.sym.var('st10',shape=(1,batch_size,512),dtype='float32'),mx.sym.var('st20',shape=(1,batch_size,1024),dtype='float32'),mx.sym.var('st01',shape=(1,batch_size,208),dtype='float32'),mx.sym.var('st11',shape=(1,batch_size,512),dtype='float32'),mx.sym.var('st21',shape=(1,batch_size,1024),dtype='float32')
#   lstm_1,[st02,st03]= mx.gluon.rnn.LSTM(208,layout='TNC',input_size=304)(combined_1,[st00,st01])
#   combined_2 = mx.sym.Concat(combined_1,lstm_1, dim=-1)
#   lstm_2,[st12,st13]= mx.gluon.rnn.LSTM(512,layout='TNC',input_size=512)(combined_2,[st10,st11])
#   combined_3 = mx.sym.Concat(combined_2,lstm_2, dim=-1)
#   lstm_3,[st22,st23]= mx.gluon.rnn.LSTM(1024,layout='TNC',input_size=1024)(combined_3,[st20,st21])
#   #out0    = mx.sym.softmax(mx.sym.FullyConnected(lstm_3,num_hidden=vocab_size))
#   denseCore,denseBias=mx.sym.var(_prefix+'dense',shape=(1024,vocab_size),init=mx.init.Xavier()),mx.sym.var(_prefix+'dense_bias',shape=(1,1,vocab_size),init=mx.init.Xavier())
#   out0   = mx.sym.broadcast_add(mx.sym.dot(lstm_3,denseCore),denseBias).softmax().clip(1e-15,1)
# #  var_list=(ew1,ew2,ew3,ew4,denseCore,denseBias)
# #  return mx.sym.MakeLoss(mx.sym.add_n(*(i.abs().mean() for i in var_list))+mx.sym.add_n(*(i.square().mean() for i in var_list))-out0.pick(lslice_0,2).log().mean()-out1.pick(lslice_1,2).log().mean()-out2.pick(lslice_2,2).log().mean()-out3.pick(lslice_3,2).log().mean())
#   return mx.gluon.SymbolBlock([out0,st02,st03,st12,st13,st22,st23],[data,st00,st01,st10,st11,st20,st21])

#x.pick(mx.nd.array([[1,1,1],[2,2,2],[0,0,0]]),1)
from time import time
import oneflow.nn as nn
import oneflow.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size,shengMu_size,yunMu_size,tone_size,embedding_dim, batch_size,seq_len=seq_length,drop=False,_prefix='',keep_state=False):
        super().__init__()
        self.drop=drop
        self.se0=nn.Embedding(vocab_size,embedding_dim)
        self.se1=nn.Embedding(shengMu_size,pyembedding_dim)
        self.se2=nn.Embedding(yunMu_size,pyembedding_dim)
        self.se3=nn.Embedding(tone_size,pyembedding_dim)
        self.lstm1=nn.LSTM(304,208,bias=False)
        self.lstm2=nn.LSTM(512,512,2,bias=False)
        self.lstm3=nn.LSTM(512,512,2,bias=False)
        self.dense=nn.Linear(512,vocab_size)
        if self.drop :
          self.drop=nn.Dropout(0.5)
        # self.concat=nn.
        # se0 = mx.sym.Embedding(data=slice_0, input_dim=vocab_size, output_dim=embedding_dim)
        # se1 = mx.sym.Embedding(data=slice_1, input_dim=shengMu_size, output_dim=pyembedding_dim)
        # se2 = mx.sym.Embedding(data=slice_2, input_dim=yunMu_size, output_dim=pyembedding_dim)
        # se3 = mx.sym.Embedding(data=slice_3, input_dim=tone_size, output_dim=pyembedding_dim)
    def initial(self,batch_size=1):
      return [flow.zeros((i,batch_size,j)).cuda() for i,j in ((1,208),(1,208),(2,512),(2,512),(2,512),(2,512))]
    def forward(self, x, *status):
        st00,st01,st10,st11,st20,st21=status
        x1,x2,x3,x4=x.split(1,-1)
        xc=flow.cat([self.se0(x1.reshape(x1.shape[:-1])),self.se1(x2.reshape(x2.shape[:-1])),self.se2(x3.reshape(x3.shape[:-1])),self.se3(x4.reshape(x4.shape[:-1]))],-1)
        if self.drop :
          xc=self.drop(xc)
        l1,(st02,st03)=self.lstm1(xc,(st00,st01))
        L1=flow.cat([xc,l1],-1)
        l2,(st12,st13)=self.lstm2(L1,(st10,st11))
        L2=L1+l2
        out=self.dense(L2)
        l3,(st22,st23)=self.lstm3(L2,(st20,st21))
        out=self.dense(L2+l3)
        return out,(st02,st03,st12,st13,st22,st23)

net=Model(vocab_size,shengMu_size,yunMu_size,tone_size,embedding_dim,batch_size=BATCH_SIZE).cuda()


STg,YTg,TTg=[i.cuda() for i in [ST,YT,TT]]
class myloss(nn.Module):
    def __init__(self,STs,YTs,TTs,weight=None):
      super().__init__()
      self.ST=STs
      self.YT=YTs
      self.TT=TTs
      self.NLL=flow.nn.NLLLoss(weight)
    def forward(self,out0,label):
      lslice_0,lslice_1,lslice_2,lslice_3 = [label[:,i] for i in range(label.shape[1])]
      out1 = out0.matmul(self.ST)
      out2 = out0.matmul(self.YT)
      out3 = out0.matmul(self.TT)
      return self.NLL(out0.log_softmax(),lslice_0)+self.NLL(out1.log_softmax(),lslice_1)*.0625+self.NLL(out2.log_softmax(),lslice_2)*.25+self.NLL(out3.log_softmax(),lslice_3)*.5

#loss_fn = nn.CrossEntropyLoss().cuda()
loss_fn = myloss(STg,YTg,TTg).cuda()
NAG=True
if NAG:
  optimizer = flow.optim.SGD(net.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-5, nesterov=True)
else:
  optimizer = flow.optim.Adam([{
    "params": net.parameters(),
    "lr": 1e-3,
    "clip_grad_max_norm": 2.0,
    "clip_grad_norm_type": 2.0,
    "weight_decay": 1e-5,
  }])

class Graph(nn.Graph):
    def __init__(self,model,loss_fn,optimizer):
        super().__init__()
        #self.config.enable_amp(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        self.config.enable_compress_memory(True)
        self.config.enable_straighten_algorithm('MemoryFirst')
        self.model     = model
        self.loss_fn   = loss_fn
        self.add_optimizer(optimizer)
    def build(self, y,x,states):
        pred,states=self.model(x,*states)
        z=y.reshape((-1,4))
        out0=pred.reshape((z.shape[0],-1))
        loss = self.loss_fn(out0, z)
        loss.backward()
        return loss

graph=Graph(net,loss_fn,optimizer)
#graph.debug(3)
x,y=it.__iter__().__next__()
t=time()
out = graph(y.cuda(),x.cuda(),net.initial(BATCH_SIZE))
print('build graph cost '+str(time()-t)+'s')


from tqdm import tqdm
def train(iter, iterlen, graph):
  initial = net.initial(BATCH_SIZE)
  with tqdm() as pbar:
    flag=True
    for x, y in iter:
      if flag:
        pbar.reset(it.now_step)
        flag=False
      loss = graph(y.cuda(), x.cuda(), initial)
      pbar.set_postfix(loss='%20.16f'%loss.tolist())
      _ = pbar.update()


#### TODO! mxboard
# from mxboard import SummaryWriter
# sw = SummaryWriter(logdir='./logs', flush_secs=5)
# #使用mxboard
# #tensorboard --logdir=./logs --host=127.0.0.1 --port=7000
# #http://localhost:7000
# #grads = [i.grad().asnumpy() for i in net.collect_params().values()]
# newval = [i.data().asnumpy() for i in net.collect_params().values()]
#
# for epoch in range(epochs):
#     t=time()
#     total_loss = 0
#     i=0
#     with tqdm(total=it.maxstep,ncols=130) as pbar:
#       pbar.set_description('batch %i' % i)
#       i+=1
#       states=net.begin_state()
#       for data, label in it:
#         with mx.autograd.record():
#             #out0= net(data).slice(begin=(None,discard_length,None),end=(None,seq_length,None))
# #            out0,*_= net(data,*states)
#             out0,*_= net(data,*states)
#             loss = loss_function(out0, label)
#         loss.backward()
#         trainer.step(BATCH_SIZE)
#         _loss=loss.asnumpy().mean()
#         pbar.set_postfix(loss=str(_loss))
#         pbar.update()
#         metric.update(label[:,:,0].reshape(-3,1), out0.reshape((-3,-1)))
#         sw.add_scalar(tag='cross_entropy', value=_loss, global_step=epoch)
#     grads = [i.grad().asnumpy() for i in net.collect_params().values()]
#     oldval = newval
#     newval = [i.data().asnumpy() for i in net.collect_params().values()]
#     diffval = [a-b for a,b in zip(newval,oldval)]
#     for i, name in enumerate(net.collect_params().keys()):
#       sw.add_histogram(tag='grad'+name, values=grads[i], global_step=epoch, bins=1000)
#       sw.add_histogram(tag='val'+name, values=newval[i], global_step=epoch, bins=1000)
#       sw.add_histogram(tag='diff'+name, values=diffval[i], global_step=epoch, bins=1000)
#     sw.add_scalar(tag=name, value=acc, global_step=epoch)
#     name, acc = metric.get()
#     print('After epoch {}: {} = {}({}s),\n    final batch loss is {}'.format(epoch + 1, name, acc,time()-t,loss.asnumpy().mean()))
#     metric.reset()
#     t=time()

def generate_text(net, start_string,temperature = 1.0,num_generate = 1000,return_str=False):
  input_eval = flow.tensor([[char2f[s]] for s in start_string]).cuda()
  text_generated = []
  print(start_string,end='')
  # Here batch size == 1
  states=net.initial(1)
  for i in range(num_generate):
      predictions,states = net(input_eval.cuda(),*states)
      # remove the batch dimension
      predictions=predictions[-1,:,:]
      predictions = (predictions/temperature).softmax()
      # using a categorical distribution to predict the word returned by the model
      # predicted_id = mx.nd.random.multinomial(predictions).asnumpy()[0]
      predicted_id = flow.multinomial(predictions,1).tolist()[0][0]
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      # input_eval = mx.nd.array([char2f[idx2char[predicted_id]]],ctx).expand_dims(0)
      input_eval = flow.tensor([[char2f[idx2char[predicted_id]]]])
      text_generated.append(idx2char[predicted_id])
      print(idx2char[predicted_id],end='')
  print()
  if return_str:
    return (start_string + ''.join(text_generated))

#net.load_state_dict(flow.load('attampt1-30.flow'))
generate_text(net, "生命的意义 ",.5,100,return_str=True)

for i in (flow.nn.init.xavier_normal_(net.state_dict()[i]) if len(net.state_dict()[i].shape)>=2 else flow.nn.init.uniform_(net.state_dict()[i]) for i in net.state_dict()):
  pass

epochs = 200
for t in range(epochs):
  data=it.__iter__()
  train(data,it.now_step,graph)
  generate_text(net, "生命的意义 ",.5,100)
  if t%10 == 9:
    flow.save(net.state_dict(), "./model"+(str(t+1))[:-1]+".dict.flow")
