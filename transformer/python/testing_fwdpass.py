import jax
import jax.numpy as jnp
from jax import random
from layers import Transformer

key = random.PRNGKey(0)

num_layers = 2
d_model = 8 
d_ff = 16
num_heads = 4

batch = 1
src_len = 6
tgt_len = 5

k_src, k_tgt = random.split(key)

src = random.normal(k_src, (1,6,8))
tgt = random.normal(k_tgt, (1,5,8))

model = Transformer(num_layers, d_model, d_ff, num_heads, key)
out = model(src, tgt)

print(out.shape)



