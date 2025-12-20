import jax.numpy as jnp
import jax.nn as nn
from jax import random, jit, value_and_grad
import optax

from embeddings import make_embeddings, embed_tokens
from transformers import AutoTokenizer
from data import make_next_token_pair
from layers import init_transformer_params, transformer_apply   # PURE functions only


d_model = 512
d_ff = 2048
num_heads = 8
num_layers = 2
learning_rate = 1e-4
num_steps = 500



tok = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tok.vocab_size
key = random.PRNGKey(0)

embedding_matrix = make_embeddings(key, vocab_size, d_model)

input_ids, target_ids = make_next_token_pair(tok, "satvik is the goat", max_len=32)
input_ids = jnp.expand_dims(jnp.array(input_ids), axis=0)
target_ids = jnp.expand_dims(jnp.array(target_ids), axis=0)


key_params = random.PRNGKey(1)
params = init_transformer_params(key_params, num_layers, d_model, d_ff, num_heads)


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    log_probs = nn.log_softmax(logits, axis=-1)
    one_hot = nn.one_hot(targets, logits.shape[-1])
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


def loss_fn(params, input_ids, target_ids, embedding_matrix):
    embs = embed_tokens(embedding_matrix, input_ids)
    out = transformer_apply(params, embs, embs)
    logits = out @ embedding_matrix.T
    return cross_entropy_loss(logits, target_ids)



optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, input_ids, target_ids, embedding_matrix):
    loss, grads = value_and_grad(loss_fn)(
        params, input_ids, target_ids, embedding_matrix
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss



print("Starting training...\n")
for step in range(num_steps):
    params, opt_state, loss = train_step(
        params, opt_state, input_ids, target_ids, embedding_matrix
    )

    if step % 50 == 0:
        print(f"Step {step:04d} | Loss: {float(loss):.4f}")

print("\nTraining complete")

embs = embed_tokens(embedding_matrix, input_ids)
out = transformer_apply(params, embs, embs)
logits = out @ embedding_matrix.T
pred_ids = jnp.argmax(logits, axis=-1)
print("\nDecoded output:\n", tok.decode(pred_ids[0]))

ir = train_step.lower(params, opt_state, input_ids, target_ids, embedding_matrix)\
               .compiler_ir("stablehlo")
print(ir)
