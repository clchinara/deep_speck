import speck as sp
import train_nets as tn

from keras.models import load_model

from constants import INV_NAME

model_dir = f'{INV_NAME}_freshly_trained_nets'
pretrained_net = load_model(f'{model_dir}/bestpretrain8i0_12_12.h5')

tn.train_speck_distinguisher_8_rounds(pretrained_net=pretrained_net, file_name_suffix='12_12_2', num_epochs=80)