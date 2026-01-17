
import jax
import jax.numpy as jnp
from jax import random

from selfattention import masked_multihead_attn

def main():
    # ---- dummy, deterministic inputs ----
    key = random.PRNGKey(0)
    x    = random.normal(key, (1, 4, 8))   # (batch, seq, d_model)
    W_q  = random.normal(key, (8, 8))
    W_k  = random.normal(key, (8, 8))
    W_v  = random.normal(key, (8, 8))
    mask = jnp.ones((4, 4), dtype=bool)    # causal/non-causal mask shape (seq, seq)

    # JIT wrapper (same signature as your function)
    f = jax.jit(masked_multihead_attn)

    # --------- JAXPR (high-level JAX IR) ---------
    print("\n=== JAXPR ===")
    print(jax.make_jaxpr(masked_multihead_attn)(x, W_q, W_k, W_v, mask))

    # --------- StableHLO (compiler IR) ----------
    lowered = f.lower(x, W_q, W_k, W_v, mask)

    # Prefer StableHLO text (works on modern JAX)
    try:
        stablehlo_txt = lowered.compiler_ir(dialect="stablehlo").as_text()
    except AttributeError:
        # Older JAX uses .operation.get_asm()
        stablehlo_txt = lowered.compiler_ir(dialect="stablehlo").operation.get_asm()

    print("\n=== StableHLO ===")
    print(stablehlo_txt)

    with open("attn.mlir", "w") as f_out:
        f_out.write(stablehlo_txt)
    print("wrote stablehlo to attn_mlir")

    # (optional) also show canonical HLO text if you want
    try:
        hlo_txt = lowered.compiler_ir(dialect="hlo").as_text()
        print("\n=== HLO ===")
        print(hlo_txt)
    except Exception:
        pass  # not all versions expose HLO text

if __name__ == "__main__":
    main()
