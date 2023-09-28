import speck as sp

import numpy as np

from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

from constants import INV_NAME, NUM_PLAINTEXTS

bs = 5000;
wdir = f'./{INV_NAME}_freshly_trained_nets/'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,));
  rs = Reshape((2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('relu')(conv0);
  #add residual blocks
  shortcut = conv0;
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('relu')(conv1);
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('relu')(conv2);
    shortcut = Add()([shortcut, conv2]);
  #add prediction head
  flat1 = Flatten()(shortcut);
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('relu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('relu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  model = Model(inputs=inp, outputs=out);
  return(model);

def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1):
    #create the network
    net = make_resnet(num_blocks=NUM_PLAINTEXTS, depth=depth, reg_param=10**-5);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    #generate training and validation data
    X, Y = sp.make_train_data(10**7,num_rounds);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds);
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);

def pretrain_8_rounds(net7, diffA, diffB, num_epochs=10, batch_size=5000, lr=0.0001):
  X, Y = sp.make_train_data(n=10**7, nr=5, diffA=diffA, diffB=diffB)
  X_eval, Y_eval = sp.make_train_data(n=10**6, nr=5, diffA=diffA, diffB=diffB)
  net7.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['acc'])
  check = make_checkpoint(wdir+'bestpretrain8.h5')
  h = net7.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval), callbacks=[check])
  np.save(wdir+'hpretrain8.npy', h.history['val_acc']);
  np.save(wdir+'hpretrain8.npy', h.history['val_loss']);
  dump(h.history,open(wdir+'histpretrain8.p','wb'));
  print("Best validation accuracy: ", np.max(h.history['val_acc']));
  return(net7, h);

def each_train_8_rounds(net, i, lr, checkpoint, num_epochs=200, batch_size=10000):
  X, Y = sp.make_train_data(n=10**9, nr=8)
  X_eval, Y_eval = sp.make_train_data(n=10**6, nr=8)
  net.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['acc'])
  check = checkpoint
  if check is None:
    check = make_checkpoint(wdir+f'best8i{i}lr{lr}.h5')
  h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval), callbacks=[check])
  np.save(wdir+f'h8r_i{i}_lr{lr}.npy', h.history['val_acc']);
  np.save(wdir+f'h8r_i{i}_lr{lr}.npy', h.history['val_loss']);
  dump(h.history,open(wdir+f'hist8r_i{i}_lr{lr}.p','wb'));
  print("Best validation accuracy: ", np.max(h.history['val_acc']));
  return(net, h, check);

def train_speck_distinguisher_8_rounds(pretrained_net, num_epochs=200, batch_size=10000):
  net = pretrained_net
  h = None
  lr = [0.0001, 0.00001, 0.000001]
  max_val_acc = 0
  checkpoint = None
  for i in range(len(lr)):
    print(f"Training {i + 1}, lr: {lr[i]}")
    net, h, checkpoint = each_train_8_rounds(net, i, lr=lr[i], checkpoint=checkpoint, num_epochs=num_epochs, batch_size=batch_size)
    max_val_acc = max(max_val_acc, np.max(h.history['val_acc']))
  print("Final best validation accuracy: ", max_val_acc);
  return (net, h)
