import speck as sp
import train_nets as tn

from keras.models import load_model

from constants import INV_NAME

diffAs = [(32768, 33802), (34304, 33290), (32768, 33802), (33280, 34314), (32780, 33798), (32772, 33806), (32772, 33806), (33280, 34314), (32768, 33802), (32768, 33802)]
diffBs = [(34752, 34496), (32832, 33088), (33728, 33472), (33216, 32960), (32832, 33088), (32960, 33216), (32832, 33088), (32832, 33088), (32960, 33216), (32832, 33088)]

model_dir = f'{INV_NAME}_freshly_trained_nets'

net7 = load_model(f'{model_dir}/best7depth1.h5')
tn.pretrain_speck_distinguisher_8_rounds(net7=net7, diffAs=diffAs, diffBs=diffBs, num_epochs=30)
