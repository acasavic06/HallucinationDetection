import numpy as np
from collections import Counter

def tokenize(text):
    text=text.lower()
    return text.split()

def build_vocabulary (tokens,min_count=1):
    word_counts = Counter(tokens)

    vocabulary = [w for w,c in word_counts.items()
                  if c >=min_count]

    word_to_idx= {w: i for i,w in enumerate(vocabulary)}
    idx_to_word = {i: w for w,i in word_to_idx.items()}

    return word_to_idx, idx_to_word

def generate_skipgram_pairs(tokens,word_to_idx,window_size=2):
    pairs=[]

    for i, word in enumerate(tokens):
        if word not in word_to_idx:
            continue

        target = word_to_idx[word]

        for j in range(i-window_size, i+window_size+1):
            if j==i:
                continue
            if j<0 or j>=len(tokens):
                continue

            context_word=tokens[j]

            if context_word not in word_to_idx:
                continue

            context=word_to_idx[context_word]

            pairs.append((target,context))

    return pairs
