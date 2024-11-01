import jax
import jax.nn as nn
import jax.numpy as jnp

import optax

from typing import Dict
from nanogpt import utils


class Transformer:

    def __init__(
        self,
        rng: jax.Array,
        context_size: int,
        vocab_size: int,
        num_heads: int,
        optimiser,
    ):

        self._context_size = context_size
        self._vocab_size = vocab_size
        self._num_heads = num_heads

        rng, embed_rng, pos_rng, wq_rng, wk_rng, wv_rng, mlp_rng = jax.random.split(
            rng, 7
        )

        initialiser = jax.nn.initializers.he_uniform()

        self._weights = {
            "embedding": initialiser(embed_rng, (self._vocab_size, 32)),
            "pos_embedding": initialiser(pos_rng, (self._context_size, 32)),
            "queries": initialiser(wq_rng, (32, 16)),
            "keys": initialiser(wk_rng, (32, 16)),
            "values": initialiser(wv_rng, (32, 16)),
            "mlp_weights": initialiser(mlp_rng, (16, self._vocab_size)),
        }

        self._opt_state = optimiser.init(self._weights)

        def self_attention_head(
            x: jax.Array, W_q: jax.Array, W_k: jax.Array, W_v: jax.Array
        ) -> jax.Array:
            """implement a single head of self-attention.

            args:
                x: input data (BxTxC)
                W_q: query weight matrix (CxH)
                W_k: key weight matrix (CxH)
                W_v: value weight matrix (CxH)

            returns:
                jax.numpy.ndarray: output of self-attention head
            """
            d_k = x.shape[2]
            q = jnp.dot(x, W_q)  # (BxTxH)
            k = jnp.dot(x, W_k)  # (BxTxH)
            v = jnp.dot(x, W_v)  # (BxTxH)
            tril = jnp.tri(x.shape[1])
            raw = jnp.matmul(q, jnp.transpose(k, axes=(0, 2, 1))) / jnp.sqrt(d_k)
            masked_qk = jnp.where(
                tril,
                raw,
                -jnp.inf,
            )  # (BxTxT)
            masked_qk = nn.softmax(masked_qk, axis=-1)  # (BxTxT)
            output = jnp.matmul(masked_qk, v)  # (BxTxH)

            return output

        def mlp(x: jax.Array, W: jax.Array):
            return nn.gelu(jnp.dot(x, W))

        def forward(x: jax.Array, weights: Dict) -> jax.Array:
            embedding = (
                weights["embedding"][x]
                + weights["pos_embedding"][jnp.arange(x.shape[1])]
            )  # (BxTxC)
            attended_embedding = self_attention_head(
                embedding, weights["queries"], weights["keys"], weights["values"]
            )
            logits = mlp(attended_embedding, weights["mlp_weights"])
            return logits

        def loss_fn(weights: Dict, x: jax.Array, y: jax.Array) -> float:
            logits = forward(x, weights)
            loss = utils.cross_entropy_loss(logits, y)
            return loss

        def update(opt_state, weights, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(weights, x, y)
            updates, new_opt_state = optimiser.update(grads, opt_state, weights)
            new_weights = optax.apply_updates(weights, updates)
            return new_opt_state, new_weights, loss

        self._update = update

    def learn(self, x: jax.Array, y: jax.Array):
        self._opt_state, self._weights, loss = self._update(
            self._opt_state, self._weights, x, y
        )
        return loss

    @property
    def params(self):
        return self._weights

    def generate(self, rng, context: jax.Array, max_new_tokens: int):
        """
        args:
            rng: random key
            context: (BxT)
            max_new_tokens: number of tokens to generate

        returns:
            generation: generated tokens (BxT+max_new_tokens)
        """
        for _ in range(max_new_tokens):
            logits = self.forward(context)  # (BxTxC)
            logits = logits[:, -1, :]  # take last step, (BxC)
            probs = nn.softmax(logits)  # (BxC)

            new_keys = jax.random.split(rng, probs.shape[0] + 1)
            rng, sample_keys = new_keys[0], new_keys[1:]

            # sample next token
            next_token = jax.vmap(
                lambda p, rng_key: jax.random.categorical(rng_key, p, shape=(1,))
            )(
                probs, sample_keys
            )  # (Bx1)

            generation = jnp.concat((context, next_token), axis=1)  # (BxT+1)
            context = generation
        return generation
