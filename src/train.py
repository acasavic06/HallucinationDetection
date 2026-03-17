import numpy as np
from tqdm import tqdm

def get_negative_samples(vocab_size, k):

    return np.random.randint(0, vocab_size, size=k)


def train(model, pairs, vocab_size, epochs=5, lr_start=0.025, neg_k=5):

    total_steps= epochs * len(pairs)
    current_step = 0

    for epoch in range(epochs):

        total_loss = 0

        for target, context in tqdm(pairs):

            neg_samples = get_negative_samples(vocab_size, neg_k)

            lr = max(lr_start * (1 - current_step / total_steps), lr_start * 0.0001)

            loss = model.train_pair(target, context, neg_samples, lr)

            total_loss += loss
            current_step += 1

        print("Epoch:", epoch, "Loss:", total_loss)