import numpy as np
from tqdm import tqdm

def get_negative_samples(vocab_size, k):

    return np.random.randint(0, vocab_size, size=k)


def train(model, pairs, vocab_size, epochs=5, lr=0.025, neg_k=5):

    for epoch in range(epochs):

        total_loss = 0

        for target, context in tqdm(pairs):

            neg_samples = get_negative_samples(vocab_size, neg_k)

            loss = model.train_pair(target, context, neg_samples, lr)

            total_loss += loss

        print("Epoch:", epoch, "Loss:", total_loss)