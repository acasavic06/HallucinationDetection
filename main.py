import numpy as np

from src.preprocessing import tokenize, build_vocabulary, generate_skipgram_pairs
from src.model import Word2Vec
from src.train import train


with open("data/text.txt") as f:
    text = f.read()

tokens = tokenize(text)

word_to_idx, idx_to_word = build_vocabulary(tokens)

pairs = generate_skipgram_pairs(tokens, word_to_idx)

vocab_size = len(word_to_idx)

model = Word2Vec(vocabulary_size=vocab_size, embedding_dim=100)

train(model, pairs, vocab_size)



'''
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

stopwords = {"is", "a", "the", "and", "of", "to", "in"}

def most_similar(word, model, word_to_idx, idx_to_word, top_k=5):

    vec = model.W_in[word_to_idx[word]]

    sims = []

    for i in range(len(idx_to_word)):

        other_word = idx_to_word[i]

        if other_word in stopwords:
            continue

        if other_word==word:
            continue

        other_vec = model.W_in[i]

        sim = cosine_similarity(vec, other_vec)

        sims.append((other_word, sim))

    sims.sort(key=lambda x: x[1], reverse=True)

    return sims[:top_k]

print("Similar to king:")
print(most_similar("king", model, word_to_idx, idx_to_word))'''