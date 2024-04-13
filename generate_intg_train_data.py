import speck as sp
import numpy as np

def generate_intg_train_data(num_batch, batch_size, num_rounds, num_plaintexts):
  for i in range(num_batch):
    X_train, Y_train = sp.make_intg_train_data(batch_size, num_rounds, num_plaintexts)
    X_eval, Y_eval = sp.make_intg_train_data(batch_size // 10, num_rounds, num_plaintexts)
    np.save(f'X_train_{i}', X_train)
    np.save(f'Y_train_{i}', Y_train)
    np.save(f'X_eval_{i}', X_eval)
    np.save(f'Y_eval_{i}', Y_eval)

generate_intg_train_data(100, 100, 9, 65536)