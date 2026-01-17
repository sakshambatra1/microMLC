import jax
import jax.numpy as jnp
from jax import random

# ==============================================================================
# Unmasked Multi-Head Attention
# ==============================================================================
def multihead_attn(x: jnp.ndarray,
                   W_q: jnp.ndarray,
                   W_k: jnp.ndarray,
                   W_v: jnp.ndarray) -> jnp.ndarray:
    
    # --- Dimensions ---
    # x: (Batch, Seq, D_Model)
    batch_size, seq_len, d_model = x.shape
    
    # We choose 4 heads. 
    # If d_model is 128, head_dim will be 32.
    num_heads = 4 
    head_dim = d_model // num_heads

    # 1. Linear Projections
    # Shapes: (Batch, Seq, Dim) @ (Dim, Dim) -> (Batch, Seq, Dim)
    Q = jnp.dot(x, W_q)
    K = jnp.dot(x, W_k)
    V = jnp.dot(x, W_v)

    # 2. Split Heads
    # Reshape: (Batch, Seq, Heads, HeadDim)
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to (Batch, Heads, Seq, HeadDim)
    # This aligns memory for the dot product
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # 3. Scaled Dot-Product Attention (No Mask)
    # (Batch, Heads, Seq, HeadDim) @ (Batch, Heads, HeadDim, Seq) -> (Batch, Heads, Seq, Seq)
    # We swap the last two dims of K to transpose it
    scores = jnp.matmul(Q, K.swapaxes(-2, -1)) / jnp.sqrt(head_dim)

    # 4. Softmax
    weights = jax.nn.softmax(scores, axis=-1)
    
    # 5. Aggregate Values
    # (Batch, Heads, Seq, Seq) @ (Batch, Heads, Seq, HeadDim) -> (Batch, Heads, Seq, HeadDim)
    out = jnp.matmul(weights, V)

    # 6. Recombine Heads
    # Transpose back to (Batch, Seq, Heads, HeadDim)
    out = out.transpose(0, 2, 1, 3)
    # Reshape to (Batch, Seq, D_Model)
    out = out.reshape(batch_size, seq_len, d_model)

    return out

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    print("Generating IR for Standard (Unmasked) Multi-Head Attention...")

    # 1. Setup Inputs
    key = random.PRNGKey(0)
    batch = 1
    seq = 128
    d_model = 128
    
    x = random.normal(key, (batch, seq, d_model))
    
    k1, k2, k3 = random.split(key, 3)
    W_q = random.normal(k1, (d_model, d_model))
    W_k = random.normal(k2, (d_model, d_model))
    W_v = random.normal(k3, (d_model, d_model))
    
    # 2. Lower to StableHLO
    f_jit = jax.jit(multihead_attn)
    lowered = f_jit.lower(x, W_q, W_k, W_v)

    # === FIX START ===
    ir_module = lowered.compiler_ir(dialect="stablehlo")
    
    # Robustly get text for any JAX version
    try:
        # Newer JAX / MLIR bindings often just use str()
        stablehlo_txt = str(ir_module)
        # If it returns a simplified object representation, try specific methods
        if "module" not in stablehlo_txt and hasattr(ir_module, "operation"):
             stablehlo_txt = ir_module.operation.get_asm()
    except Exception:
        # Fallback for older versions
        stablehlo_txt = ir_module.as_text()
    # === FIX END ===

    # 3. Save to File
    filename = "attn_unmasked.mlir"
    with open(filename, "w") as f:
        f.write(stablehlo_txt)
    
    print(f"Success! Written to '{filename}'")

if __name__ == "__main__":
    main()