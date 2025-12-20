import jax.numpy as jnp 
import jax 
from jax import random 
from selfattention import multihead_attn, masked_multihead_attn

def relu(x:jax.Array):
    return jnp.maximum(0,x)

class LayerNorm: 
    def __init__(self, dim, eps=1e-5): 
        self.eps = eps 
        self.gamma = jnp.ones((dim,))
        self.beta = jnp.zeros((dim,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis = -1, keepdims=True)
        x_norm = (x-mean)/jnp.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out 

class FeedForwardNN: 
    def __init__(self, d_model:int, d_ff:int, key:jax.Array):
        k1, k2, k3, k4 = random.split(key, 4)
        self.W1 = random.normal(k1, (d_model, d_ff)) * jnp.sqrt(2/d_model)
        self.W2 = random.normal(k2, (d_ff, d_model)) * jnp.sqrt(2/d_ff)
        self.B1 = jnp.zeros((d_ff,))
        self.B2 = jnp.zeros((d_model,))

    def __call__(self, x):
        layer1 = (x @ self.W1) + self.B1
        layer2 = relu(layer1)
        layer3 = (layer2 @ self.W2) + self.B2
        return layer3
    
class EncoderBlock: 
    def __init__(self, d_model:int, d_ff:int, num_heads:int, key:jax.Array):
        k1, k2, k3, k_ff = random.split(key, 4)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
        self.W_q = random.normal(k1, (d_model, d_model)) *jnp.sqrt(2.0/ d_model)
        self.W_k = random.normal(k2, (d_model, d_model)) * jnp.sqrt(2.0/d_model)
        self.W_v = random.normal(k3,(d_model, d_model)) * jnp.sqrt(2.0/d_model)
        
        self.ffnn = FeedForwardNN(d_model, d_ff, k_ff)
        
        self.d_model = d_model
        self.num_heads = num_heads
    

    def __call__(self, x): 
        x_norm = self.ln1(x)
        attn_out = multihead_attn(x_norm, x_norm, self.W_q, self.W_k, self.W_v)
        x = x + attn_out
        
        x_norm2 = self.ln2(x)
        ffn_out = self.ffnn(x_norm2)
        x = x + ffn_out 
        return x
    
class DecoderBlock:
    def __init__(self,d_model:int, d_ff:int, num_heads:int, key:jax.Array):
        k1, k2, k3, k4, k5, k6, k_ff = random.split(key, 7) 
            
        self.Wq_mask = random.normal(k1, (d_model, d_model)) *jnp.sqrt(2.0/ d_model)
        self.Wk_mask = random.normal(k2, (d_model, d_model)) * jnp.sqrt(2.0/d_model)
        self.Wv_mask = random.normal(k3,(d_model, d_model)) * jnp.sqrt(2.0/d_model)

        self.Wq_cross = random.normal(k4, (d_model, d_model)) *jnp.sqrt(2.0/ d_model)
        self.Wk_cross = random.normal(k5, (d_model, d_model)) * jnp.sqrt(2.0/d_model)
        self.Wv_cross = random.normal(k6,(d_model, d_model)) * jnp.sqrt(2.0/d_model)

        self.ffnn = FeedForwardNN(d_model, d_ff, k_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)
        
        self.d_model = d_model
        self.num_heads = num_heads
        
    def __call__(self, x, enc_out):
        batch_size, seq_len, _ = x.shape
        
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        x1 = self.ln1(x)
        m_attn_out = masked_multihead_attn(x1, self.Wq_mask, self.Wk_mask, self.Wv_mask, mask)
       
        x = x + m_attn_out
        x2 = self.ln2(x)
        c_attn_out = multihead_attn(x2, enc_out, self.Wq_cross, self.Wk_cross, self.Wv_cross)
        
        x = x + c_attn_out
        x3 = self.ln3(x)
        x = x + self.ffnn(x3)
        
        return x
    
class Encoder: 
    def __init__(self, num_layers:int,  d_model:int, d_ff:int, num_heads:int, key:jax.Array):
        keys = random.split(key, num_layers)
        self.layers = [EncoderBlock(d_model, d_ff, num_heads, k) for k in keys]
        self.ln_final = LayerNorm(d_model)

    def __call__(self, x):
        for layer in self.layers: 
            x = layer(x) 
        x = self.ln_final(x)
        return x
    
class Decoder: 
    def __init__(self, num_layers:int, d_model:int, d_ff:int, num_heads:int, key:jax.Array):
        keys = random.split(key, num_layers)
        self.layers = [DecoderBlock(d_model, d_ff, num_heads, k) for k in keys]
        self.ln_final = LayerNorm(d_model)


    def __call__(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        x = self.ln_final(x)
        return x 

class Transformer: 
    def __init__(self, num_layers: int, d_model:int, d_ff:int, num_heads:int, key:jax.Array): 
        k_enc, k_dec = random.split(key)
        self.encoder = Encoder(num_layers, d_model, d_ff, num_heads, k_enc)
        self.decoder = Decoder(num_layers, d_model, d_ff, num_heads, k_dec)

    def __call__(self, src, tgt): 
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        return dec_out
    

def init_transformer_params(key, num_layers, d_model, d_ff, num_heads):
    k1, k2, k3 = random.split(key, 3)
    # encoder weights
    enc_Wq = random.normal(k1, (d_model, d_model))
    enc_Wk = random.normal(k2, (d_model, d_model))
    enc_Wv = random.normal(k3, (d_model, d_model))
    # normally you’d add ffn weights etc.; this is just a start
    params = {
        "encoder": {
            "Wq": enc_Wq,
            "Wk": enc_Wk,
            "Wv": enc_Wv,
        },
        # you can mirror similar keys for the decoder
    }
    return params


def transformer_apply(params, src, tgt):
    # simple demo forward using those weights
    Wq, Wk, Wv = params["encoder"]["Wq"], params["encoder"]["Wk"], params["encoder"]["Wv"]
    Q = src @ Wq
    K = src @ Wk
    V = src @ Wv
    out = (Q + K + V) / 3.0
    return out










