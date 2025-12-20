import time
import jax
import jax.numpy as jnp
from jax import random

num_heads = 4

# ------------------------------
# Plain (non-jitted) baseline
# ------------------------------
def multihead_attn_impl(q_input: jnp.ndarray,
                        kv_input: jnp.ndarray,
                        W_q: jnp.ndarray,
                        W_k: jnp.ndarray,
                        W_v: jnp.ndarray) -> jnp.ndarray:
    batch_size, q_len, d_model = q_input.shape
    head_dim = d_model // num_heads

    Q = q_input @ W_q
    K = kv_input @ W_k
    V = kv_input @ W_v

    _, kv_len, _ = K.shape

    Q = Q.reshape(batch_size, q_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = jnp.matmul(Q, K.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(weights, V)

    out = out.transpose(0, 2, 1, 3).reshape(batch_size, q_len, d_model)
    return out


# ---------------------------------------
# Tiled matmul (same pattern as your pass)
# ---------------------------------------
def tiled_matmul_3d(x: jnp.ndarray,
                    W: jnp.ndarray,
                    tile_m: int = 16,
                    tile_n: int = 16,
                    tile_k: int = 16) -> jnp.ndarray:
    """
    x: (batch, M, K)
    W: (K, N)
    returns: (batch, M, N)
    """
    batch, M, K = x.shape
    K2, N = W.shape
    assert K == K2, "Incompatible matmul shapes"

    C = jnp.zeros((batch, M, N), dtype=x.dtype)

    # Python loops – cheap for small fixed ranges, and still reflect your tiling logic
    for b in range(batch):
        for i in range(0, M, tile_m):
            for j in range(0, N, tile_n):
                acc = jnp.zeros((tile_m, tile_n), dtype=x.dtype)
                for k in range(0, K, tile_k):
                    a_tile = x[b, i:i+tile_m, k:k+tile_k]      # (tile_m, tile_k)
                    b_tile = W[k:k+tile_k, j:j+tile_n]        # (tile_k, tile_n)
                    acc = acc + a_tile @ b_tile               # (tile_m, tile_n)
                C = C.at[b, i:i+tile_m, j:j+tile_n].add(acc)
    return C


def multihead_attn_tiled_impl(q_input: jnp.ndarray,
                              kv_input: jnp.ndarray,
                              W_q: jnp.ndarray,
                              W_k: jnp.ndarray,
                              W_v: jnp.ndarray) -> jnp.ndarray:
    batch_size, q_len, d_model = q_input.shape
    head_dim = d_model // num_heads

    Q = tiled_matmul_3d(q_input, W_q)
    K = tiled_matmul_3d(kv_input, W_k)
    V = tiled_matmul_3d(kv_input, W_v)

    _, kv_len, _ = K.shape

    Q = Q.reshape(batch_size, q_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = jnp.matmul(Q, K.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(weights, V)

    out = out.transpose(0, 2, 1, 3).reshape(batch_size, q_len, d_model)
    return out


# Jitted versions used only for timing
multihead_attn = jax.jit(multihead_attn_impl)
multihead_attn_tiled = jax.jit(multihead_attn_tiled_impl)


def bench(fn, args, iters=50):
    fn(*args).block_until_ready()  # compile
    t0 = time.time()
    for _ in range(iters):
        y = fn(*args)
    y.block_until_ready()
    return (time.time() - t0) / iters


if __name__ == "__main__":
    key = random.PRNGKey(0)

    # Bigger than (1,4,8), but not crazy
    batch, seq, d_model = 4, 64, 256
    x = random.normal(key, (batch, seq, d_model))

    kq, kk, kv = random.split(key, 3)
    W_q = random.normal(kq, (d_model, d_model))
    W_k = random.normal(kk, (d_model, d_model))
    W_v = random.normal(kv, (d_model, d_model))

    args = (x, x, W_q, W_k, W_v)

    # ---- correctness without jit ----
    print("===== Correctness =====")
    ref = multihead_attn_impl(*args)
    til = multihead_attn_tiled_impl(*args)
    print("max diff =", jnp.max(jnp.abs(ref - til)))

    # ---- CPU benchmark with jit ----
    print("\n===== CPU JIT benchmark =====")
    t_base = bench(multihead_attn, args)
    t_tiled = bench(multihead_attn_tiled, args)
    print(f"baseline = {t_base:.6f}s")
    print(f"tiled    = {t_tiled:.6f}s")
    print(f"speedup  = {t_base / t_tiled:.3f}x")
