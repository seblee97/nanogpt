from typing import Tuple, List

import jax
import jax.numpy as jnp


def read_data(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


def train_test_split(data: str, train_fraction: float) -> Tuple[str, str]:
    split_index = int(len(data) * train_fraction)
    return data[:split_index], data[split_index:]


def chunk_data(data_stream: str, context_size: int) -> List[str]:
    return [
        data_stream[i : i + context_size]
        for i in range(len(data_stream) - context_size)
    ]


def tokenise(data_stream: str, vocabulary: List[str]) -> List[str]:
    vtoi = {v: i for i, v in enumerate(vocabulary)}
    tokenised_data = jnp.array([vtoi[v] for v in data_stream])
    return tokenised_data


def decode_tokens(tokens: jax.typing.ArrayLike, vocabulary: List[str]) -> str:
    itov = {i: v for i, v in enumerate(vocabulary)}
    return "".join([itov[t.item()] for t in tokens])


def get_batch(
    data: jax.Array,
    batch_size: int,
    context_size: int,
    rng: jax.Array,
):
    idx = jax.random.randint(
        key=rng,
        shape=(batch_size,),
        minval=0,
        maxval=len(data) - context_size,
    )
    x = jnp.stack([data[i : i + context_size] for i in idx])
    y = jnp.stack([data[i + 1 : i + context_size + 1] for i in idx])
    return x, y


def cross_entropy_loss(predictions: jax.Array, targets: jax.Array) -> float:
    """cross entropy loss.

    args:
        predictions (jax.Array): predictions from model (BxC, )
        targets (jax.Array): target values (B)

    returns:
        cross entropy loss
    """
    B, C, vocab_size = predictions.shape
    predictions = predictions.reshape(B * C, vocab_size)
    softmax_predictions = jax.nn.softmax(predictions)
    targets = targets.reshape(-1)
    return -jnp.mean(jnp.choose(targets, jnp.log(softmax_predictions).T, mode="wrap"))
