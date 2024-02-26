import train_nets as tn

tn.train_intg_speck_distinguisher(num_samples=10**6, num_batch=1000, num_epochs=10, num_rounds=9)
# depth=1,reg_param=10**-8

# tn.train_intg_speck_distinguisher(num_samples=10, num_batch=2, num_epochs=1, num_rounds=9)