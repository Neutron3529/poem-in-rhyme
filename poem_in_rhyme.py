import tensorflow as tf
tf.config.optimizer.set_jit(True)
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import os
import time

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

uniqS=set(myPinYin(char,3) for char in vocab)
uniqY=set(myPinYin(char,5) for char in vocab)
uniqY.add('?i')
uniqY.add('?hi')


s2idx={s:i for i,s in enumerate(uniqS)}
y2idx={y:i for i,y in enumerate(uniqY)}

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


text_as_int = np.array([char2f[c] for c in text])
vocab_as_pyint=np.array([char2f[c] for c in vocab])[:,1:]


# The maximum length sentence we want for a single input in characters
seq_length = 640
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 8

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 20000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size   = len(vocab)
shengMu_size = len(uniqS)
yunMu_size   = len(uniqY)
tone_size    = 5
# The embedding dimension
embedding_dim = 256

pyembedding_dim = 16

# Number of RNN units
rnn_units = 1024


#SMOH=tf.expand_dims(tf.keras.backend.one_hot(vocab_as_pyint[:,0],shengMu_size),2)
#YMOH=tf.expand_dims(tf.keras.backend.one_hot(vocab_as_pyint[:,1],yunMu_size),2)
#TOH=tf.expand_dims(tf.keras.backend.one_hot(vocab_as_pyint[:,2],tone_size),2)
SMOH=tf.keras.backend.one_hot(vocab_as_pyint[:,0],shengMu_size)
YMOH=tf.keras.backend.one_hot(vocab_as_pyint[:,1],yunMu_size)
TOH=tf.keras.backend.one_hot(vocab_as_pyint[:,2],tone_size)
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
#  def spilt(x):
#    return x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3]
  inputs  = tf.keras.layers.Input(batch_shape=(batch_size,None,4),dtype='int32')
#  slice_0,slice_1,slice_2,slice_3 = tf.keras.layers.Lambda(spilt)(inputs)
  slice_0,slice_1,slice_2,slice_3 = [tf.keras.backend.squeeze(i,axis=-1) for i in tf.split(inputs,4,axis=-1)]
  se0     = tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None])(slice_0)
  se1     = tf.keras.layers.Embedding(vocab_size, pyembedding_dim,batch_input_shape=[batch_size, None])(slice_1)
  se2     = tf.keras.layers.Embedding(vocab_size, pyembedding_dim,batch_input_shape=[batch_size, None])(slice_2)
  se3     = tf.keras.layers.Embedding(vocab_size, pyembedding_dim,batch_input_shape=[batch_size, None])(slice_3)
  combined= tf.keras.layers.concatenate([se0,se1,se2,se3], axis=-1)
  lstm    = tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform')(combined)
  lstm2   = tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform')(lstm)
  out0    = tf.keras.layers.Dense(vocab_size,activation='softmax')(lstm2)
  out1    = tf.keras.layers.Dense(shengMu_size,activation='softmax')(lstm2)
  out2    = tf.keras.layers.Dense(yunMu_size,activation='softmax')(lstm2)
  out3    = tf.keras.layers.Dense(tone_size,activation='softmax')(lstm2)
#  loss0=tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits(slice_1,out01)+tf.nn.softmax_cross_entropy_with_logits(slice_2,out02)+tf.nn.softmax_cross_entropy_with_logits(slice_3,out03),-1)
  cce=tf.keras.losses.CategoricalCrossentropy(reduction='none',from_logits=True)
  aux     = tf.keras.backend.expand_dims(tf.keras.layers.add([
  cce(tf.einsum('abc,cd->abd',out0,SMOH),out1),
  cce(tf.einsum('abc,cd->abd',out0,YMOH),out2),
  cce(tf.einsum('abc,cd->abd',out0,TOH),out3)
			]),axis=-1)
  o= tf.keras.layers.concatenate([out0,out1,out2,out3,aux], axis=-1)
  return tf.keras.models.Model(inputs=inputs, outputs=o)

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


delim0=vocab_size
delim1=delim0+shengMu_size
delim2=delim1+yunMu_size
delim3=delim2+tone_size

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,0], logits[:,:,:delim0]
  )+ tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,1], logits[:,:,delim0:delim1]
  )+ tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,2], logits[:,:,delim1:delim2]
  )+ tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,3], logits[:,:,delim2:delim3]
  )+ logits[:,:,delim3]




_=''' test only!
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions[:,:,:delim0].shape, "# (batch_size, sequence_length, vocab_size)")
  print(example_batch_predictions[:,:,delim0:delim1].shape, "# (batch_size, sequence_length, shengMu_size)")
  print(example_batch_predictions[:,:,delim1:delim2].shape, "# (batch_size, sequence_length, yunMu_size)")
  print(example_batch_predictions[:,:,delim2:delim3].shape, "# (batch_size, sequence_length, tone_size)")
  print(example_batch_predictions[:,:,delim3:].shape, "# (batch_size, sequence_length, 1)(loss)")


sampled_indices = tf.random.categorical(example_batch_predictions[0,:,:delim0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print("Input: \n", repr("".join(idx2char[input_example_batch[0,:,0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

def dloss(labels, logits):
  return [tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,0], logits[:,:,:delim0]
  ), tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,1], logits[:,:,delim0:delim1]
  ), tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,2], logits[:,:,delim1:delim2]
  ), tf.keras.losses.sparse_categorical_crossentropy(labels[:,:,3], logits[:,:,delim2:delim3]
  ), logits[:,:,delim3]]


example_batch_loss  = dloss(target_example_batch, example_batch_predictions)

print("Prediction shape: ", example_batch_predictions[0].shape, " # (batch_size, sequence_length)")
print("scalar_loss:      ", example_batch_loss[0].numpy().mean())
print("scalar_loss:      ", example_batch_loss[1].numpy().mean())
print("scalar_loss:      ", example_batch_loss[2].numpy().mean())
print("scalar_loss:      ", example_batch_loss[3].numpy().mean())

''';del _

model.compile(optimizer='adam', loss=loss)


# Directory where the checkpoints will be saved
checkpoint_dir = '.\\training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


EPOCHS=10
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    period=EPOCHS)




#history = model.fit(dataset, epochs=EPOCHS)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))



def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)
  # Number of characters to generate
  num_generate = 1000
  # Converting our start string to numbers (vectorizing)
  input_eval = [char2f[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  # Empty string to store our results
  text_generated = []
  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0
  print(start_string,end='')
  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)[:,:,:delim0]
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      predictions = tf.math.log(predictions/(1-predictions))/ temperature
      # using a categorical distribution to predict the word returned by the model
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([char2f[idx2char[predicted_id]]], 0)
      text_generated.append(idx2char[predicted_id])
      print(idx2char[predicted_id],end='')
  print()
  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"生命的意义"))
