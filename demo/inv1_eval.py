import inv1_speck as sp
import utils

import numpy as np
from keras.models import load_model

INV_NAME = 'inv1'

model_dir = f'{INV_NAME}_freshly_trained_nets'
output_file = f'{INV_NAME}_logs.txt'

def evaluate(net,X,Y):
    # print('X.shape', X.shape)
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    log_str = f'Accuracy: {acc}, TPR: {tpr}, TNR: {tnr}, MSE: {mse}\nPercentage of random pairs with score higher than median of real pairs: {100*high_random}'
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);
    return acc, tpr, tnr, mse

"""
ciphertexts = [0x00400000, 0x00400000, 0x00400000, 0x00400000]
"""
def demo_func(nr, ciphertexts=[]):
    net = load_model(f'{model_dir}/best{nr}.h5')
    np_ciphertexts = []
    for c in ciphertexts: # 32-bit each
        left, right = utils.extract_left_right(c)
        np_ciphertexts.append(np.array([left]))
        np_ciphertexts.append(np.array([right]))
    X = sp.convert_to_binary(np_ciphertexts)
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5)
    return Z[0], Zbin[0]


def eval_func():
    net5 = load_model(f'{model_dir}/best5.h5')
    net6 = load_model(f'{model_dir}/best6.h5')
    net7 = load_model(f'{model_dir}/best7.h5')
    net8 = load_model(f'{model_dir}/best8.h5')
    nets = [net5, net6, net7, net8]
    res = []
    for i, net in enumerate(nets):
        num_rounds = i + 5
        X, Y = sp.make_train_data(10**6, num_rounds) # X.shape = (1000000, 128)
        acc, tpr, tnr, mse = evaluate(net, X, Y)
        res.append({
            'rounds': str(num_rounds),
            'acc': str(acc),
            'tpr': str(tpr), 
            'tnr': str(tnr),
            'mse': str(mse)
        })
    return res
