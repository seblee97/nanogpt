from nanogpt import utils
import yaml
import jax
import jax.numpy as jnp

import optax

from nanogpt.models import bigram, transformer


def prepare_data(data_path: str, train_fraction: float):
    data_stream = utils.read_data(config["data_path"])
    vocabulary = sorted(list(set(data_stream)))
    vocab_size = len(vocabulary)
    print(f"Vocabulary: {"".join(vocabulary)}")
    print(f"Vocabulary size: {vocab_size}")
    tokenised_data = utils.tokenise(data_stream, vocabulary)
    train_stream, val_stream = utils.train_test_split(
        tokenised_data, config["train_fraction"]
    )
    return train_stream, val_stream, vocab_size, vocabulary


if __name__ == "__main__":

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    rng = jax.random.PRNGKey(config["seed"])

    rng, model_rng = jax.random.split(rng)

    train_stream, val_stream, vocab_size, vocabulary = prepare_data(
        config["data_path"], config["train_fraction"]
    )

    optimiser = optax.adamw(learning_rate=config["learning_rate"])

    if config["model"] == "transformer":
        model = transformer.Transformer(
            rng=model_rng,
            context_size=config["context_size"],
            vocab_size=vocab_size,
            optimiser=optimiser,
            num_heads=1,
        )
    elif config["model"] == "bigram":
        model = bigram.Bigram(
            rng=model_rng,
            context_size=config["context_size"],
            vocab_size=vocab_size,
            optimiser=optimiser,
        )

    for _ in range(1000):
        batch_rng, rng = jax.random.split(rng)
        x, y = utils.get_batch(
            train_stream, config["batch_size"], config["context_size"], batch_rng
        )
        loss = model.learn(x, y)
        print(loss)

    import pdb

    pdb.set_trace()

    # loss = utils.cross_entropy_loss(logits, y)

    rng, generation_rng = jax.random.split(rng)
    generation = model.generate(generation_rng, jnp.zeros(shape=(3, 1), dtype=int), 10)

    # train_data = train.chunk_data(train_stream, config["context_size"])
    # val_data = train.chunk_data(val_stream, config["context_size"])

    print(utils.decode_tokens(generation[0], vocabulary))

    import pdb

    pdb.set_trace()
