import numpy as np

# TODO possibly use jax again?
def symlog(x):
  return np.sign(x) * np.log(1 + np.abs(x))

def symexp(x):
  return np.sign(x) * (np.exp(np.abs(x)) - 1)