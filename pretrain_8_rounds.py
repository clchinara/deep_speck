import speck as sp
import train_nets as tn

import math
from keras.models import load_model

from constants import INV_NAME

model_dir = f'{INV_NAME}_freshly_trained_nets'

net7 = load_model(f'{model_dir}/best7depth1.h5')
# tn.pretrain_speck_distinguisher_8_rounds_new(net7=net7, diff_to_num_train_data=diff_to_num_train_data, num_epochs=20)

diffAs = [(32768, 33802)]
diffBs = [(32832, 33088)]
diffCs = [(12608, 57696)]

tn.pretrain_speck_distinguisher_8_rounds(net7=net7, diffAs=diffAs, diffBs=diffBs, diffCs=diffCs, file_name_suffix='12_12', num_epochs=20)