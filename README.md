# Text-Generator
Simple Text Generator
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Activation
from tensorflow.keras.optimizers import RMSprop


filepath = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath,'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c,i) for i,c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))

Seq_len = 40
Step_size = 3

sentences = []
next_char = []



for i in range(0,len(text) - Seq_len,Step_size):
  sentences.append(text[i: i+Seq_len])
  next_char.append(text[i+Seq_len])


x = np.zeros((len(sentences), Seq_len, len(characters)), dtype=np.float32)
y = np.zeros((len(sentences), len(characters)), dtype=np.float32)

for i,sentence in enumerate(sentences):
  for t, character in enumerate(sentence):
    x[i, t, char_to_index[character]]=1
  y[i, char_to_index[next_char[i]]]=1

model = Sequential()
model.add(LSTM(128, input_shape=(Seq_len, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

model.fit(x,y, batch_size=256 , epochs=4)
model.save('textgenerator.keras')

def sample(preds,temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds= np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)



def generate_text(length, temperature):
  start_index = random.randint(0,len(text)-Seq_len - 1)
  generated = ''
  sentence = text[start_index: start_index + Seq_len]
  generated += sentence
  for i in range(length):
    x = np.zeros((1,Seq_len, len(characters)))
    for t, character in enumerate(sentence):
      x[0,t, char_to_index[character]]=1
    
    predictions = model.predict(x, verbose=0)[0]
    next_index = sample(predictions, temperature)
    next_char = index_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char
  return generated


print('-------0.2--------')
print(generate_text(300, 0.2))
print('-------0.4--------')
print(generate_text(300, 0.4))
print('-------0.6--------')
print(generate_text(300, 0.6))
print('-------0.8--------')
print(generate_text(300, 0.8))
print('-------1--------')
print(generate_text(300, 1))
