import jax 
import jax.random as random
import jax.numpy as jnp 
from typing import Tuple
from jax import make_jaxpr, jit

from jax.typing import ArrayLike

def make_embeddings(key: jax.Array, vocab_size: int = 50257, d_model: int = 512) -> jnp.ndarray:
    return random.normal(key, (vocab_size, d_model))

def embed_tokens(embedding_matrix: jnp.ndarray, tokens_id:jnp.ndarray) -> jnp.ndarray: 
    return embedding_matrix[tokens_id]

def precompute_rope(max_seq_len: int, d_model:int) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    assert d_model % 2 == 0 
    dim_half = d_model // 2 

    # FIX: Corrected typo from jnp.arrange to jnp.arange
    dim_indices = jnp.arange(0, dim_half)
    
    # FIX: Corrected denominator calculation for standard RoPE (dim_indices * 2 / d_model)
    freqs = 1.0 / 10000**(dim_indices * 2 / d_model)

    positions = jnp.arange(max_seq_len)
    theta = positions[:, None] * freqs[None, :]
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    return cos, sin

import jax.numpy as jnp

def rotate_pairs_jax(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    orig_shape = x.shape
    d_model = orig_shape[-1]
    dim_half = d_model // 2

    assert d_model % 2 == 0, "d_model must be even"

    # 1. Reshape into pairs: (..., seq_len, dim_half, 2)
    # The -1 infers the missing batch/seq dimensions.
    x_pairs = x.reshape(*orig_shape[:-1], dim_half, 2)

    # 2. Split the pairs
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1] # x1/x2.shape is (..., seq_len, dim_half)

    # 3. Add batch dimension to cos/sin if input is batched (3D)
    # The rotation components need to broadcast over the batch axis.
    # We use jnp.where or a simple addition for the rotation.
    
    # cos/sin.shape is (seq_len, dim_half)
    # x1/x2.shape is (B, S, D/2) if batched, (S, D/2) otherwise.
    # JAX will automatically broadcast (S, D/2) across a leading batch axis (B, S, D/2).

    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos

    # 4. Stack and reshape back to original rank
    x_rot = jnp.stack([x1_new, x2_new], axis=-1)

    # Return to original shape (..., d_model)
    return x_rot.reshape(orig_shape)

def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:

    seq_len = x.shape[-2]
    cos_slice = cos[:seq_len]
    sin_slice = sin[:seq_len]
    return rotate_pairs_jax(x, cos_slice, sin_slice)


    # jitted entrypoints for inspection and speed
apply_rope_jit = jax.jit(apply_rope)
    
# A small convenience to apply RoPE to token ids -> final embeddings
@jax.jit
def embed_and_apply_rope(embedding_matrix: jnp.ndarray, token_ids: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:

    embs = embed_tokens(embedding_matrix, token_ids)
    return apply_rope(embs, cos, sin)


def print_jaxpr_and_hlo(jitted_fn, example_args):


    # 1. Get JAXPR (The high-level IR)
    try:
        # NOTE: We use the *unjitted* version of the function for jax.make_jaxpr
        # to trace the Python logic before it hits the JIT decorator.
        # If jitted_fn is defined with @jax.jit, we need to access the original
        # function if possible, or assume the user passes the @jit version and 
        # let JAX unwrap it (which it often can). A safer way is:
        # jaxpr = make_jaxpr(jitted_fn.lower(example_args).as_hlo_module().module.entry_function.to_text) 
        # But we will rely on jax's auto-unwrapping behavior for simplicity.
        
        # JAX often unwraps the JIT to get the raw function for tracing
        jaxpr = make_jaxpr(jitted_fn)(*example_args)
        print("="*20)
        print("  JAXPR (Intermediate Representation)  ")
        print("="*20)
        print(jaxpr)
    except Exception as e:
        print("Error: Failed to make jaxpr. Ensure the function is traceable.")
        print(e)
        return

    # 2. Get HLO / Compiler IR (The low-level XLA IR)
    print("\n" + "="*20)
    print("  XLA HLO (High-Level Optimizer IR)  ")
    print("="*20)
    
    try:
        # .lower() is called on the JIT-compiled function
        lowered = jitted_fn.lower(*example_args)
        
        # Get the compiler IR object
        ir = lowered.compiler_ir() 
        
        # Attempt to get HLO text (common method)
        hlo_text = ir.as_hlo_text()

        # Print the first 1000 characters (HLO can be long)
        print(hlo_text[:1000] + "\n[... truncated ...]")
        
    except AttributeError:
        # Fallback for older JAX versions or specific backends
        print("Could not retrieve HLO using .compiler_ir().as_hlo_text().")
        print(str(ir)[:1000] + "\n[... truncated ...]")
        
    except Exception as e:
        print("Error: Failed to lower and get HLO.")
        print(e)

if __name__ == "__main__":
    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    vocab_size = 50257
    d_model = 512
    max_seq_len = 1024

    # make embeddings
    emb = make_embeddings(subkey, vocab_size=vocab_size, d_model=d_model)

    # small toy sequence of length 10
    tokens = jnp.arange(10)
    seq_emb = embed_tokens(emb, tokens) # (10, 512)

    cos, sin = precompute_rope(max_seq_len=max_seq_len, d_model=d_model)

    # apply RoPE normally and jitted
    out = apply_rope(seq_emb, cos, sin)
    out_jit = apply_rope_jit(seq_emb, cos, sin)

    # print shapes
    print("out shape:", out.shape)
    print("out_jit shape:", out_jit.shape)

    # show jaxpr and HLO for jitted function
    print_jaxpr_and_hlo(apply_rope_jit, (seq_emb, cos, sin))

