import speck as sp
import numpy as np

from keras.models import load_model

from constants import INV_NAME

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
    return log_str

net5 = load_model(f'{model_dir}/best5depth10.h5')
net6 = load_model(f'{model_dir}/best6depth10.h5')
net7 = load_model(f'{model_dir}/best7depth1.h5')
net8 = load_model(f'{model_dir}/best8i1lr1e-05.h5')

nets = [net5, net6, net7, net8]

with open(output_file, 'w') as fn:
    for i, net in enumerate(nets):
        num_rounds = i + 5
        X, Y = sp.make_train_data(10**6, num_rounds)
        fn.write(f'{num_rounds} rounds:\n{evaluate(net, X, Y)}\n')
fn.close()
