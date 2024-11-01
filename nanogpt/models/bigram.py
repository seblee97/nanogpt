import jax
import jax.nn as nn
import jax.numpy as jnp

import optax

from nanogpt import utils


class Bigram:

    def __init__(
        self,
        rng: jax.Array,
        context_size: int,
        vocab_size: int,
        optimiser,
    ):
        self._context_size = context_size
        self._vocab_size = vocab_size

        self._token_embedding = jax.random.normal(
            rng, (self._vocab_size, self._vocab_size)
        )

        self._opt_state = optimiser.init(self._token_embedding)

        def forward(embedding: jax.Array, x: jax.Array) -> jax.Array:
            """lookup token embeddings.

            args:
                x: input data (BxT)
                embedding: token embedding matrix (C x C)

            returns:
                logits: token embeddings (BxTxC)
            """
            logits = embedding[x]
            return logits

        def loss_fn(token_embedding: jax.Array, x: jax.Array, y: jax.Array) -> float:
            """cross entropy loss.

            args:
                token_embedding: token embedding matrix (C x C)
                x: input data (BxT)
                y: target values (BxT)

            returns:
                loss: cross entropy loss
            """
            logits = forward(token_embedding, x)
            loss = utils.cross_entropy_loss(logits, y)
            return loss

        def update(
            opt_state, token_embedding: jax.Array, x: jax.Array, y: jax.Array
        ) -> jax.Array:
            """update token embedding matrix.

            args:
                token_embedding: token embedding matrix (C x C)
                x: input data (BxT)
                y: target values (BxT)

            returns:
                updated_params: updated token embedding matrix
            """
            loss, grads = jax.value_and_grad(loss_fn)(token_embedding, x, y)
            updates, new_opt_state = optimiser.update(grads, opt_state, token_embedding)
            new_token_embedding = optax.apply_updates(token_embedding, updates)
            return new_opt_state, new_token_embedding, loss

        self._update = jax.jit(update)

    def learn(self, x: jax.Array, y: jax.Array):
        self._opt_state, self._token_embedding, loss = self._update(
            self._opt_state, self._token_embedding, x, y
        )
        return loss

    @property
    def params(self):
        return self._token_embedding

    @params.setter
    def params(self, value):
        self._token_embedding = value

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
