import speck as sp
import train_nets as tn

from keras.models import load_model

from constants import INV_NAME

diffA = (32768, 33802)
diffB = (32832, 33088)

model_dir = f'{INV_NAME}_freshly_trained_nets'

net7 = load_model(f'{model_dir}/best7depth1.h5')
tn.pretrain_8_rounds(net7=net7, diffA=diffA, diffB=diffB, num_epochs=30)


