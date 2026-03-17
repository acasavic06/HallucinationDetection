# Word2Vec (Skip-gram with Negative Sampling) — NumPy Implementation

This project implements the core training loop of the Word2Vec model from scratch using pure NumPy (no PyTorch / TensorFlow or other ML frameworks).

---

## 📌 Project Overview

The goal of this project is to demonstrate a full understanding of how Word2Vec works internally by implementing:

- Tokenization and preprocessing  
- Vocabulary construction  
- Skip-gram pair generation  
- Negative sampling  
- Forward pass  
- Loss computation  
- Gradient calculation  
- Parameter updates (SGD)  

---

## ⚙️ Implementation Details

### 1. Preprocessing 

- Lowercasing text  
- Removing non-alphabetic characters  
- Splitting text into tokens  

```python
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()
```

---

### 2. Skip-gram Pair Generation 

For each target word, context words are extracted within a fixed window.

---

### 3. Model

The model uses two embedding matrices:

- `W_in` — input embeddings  
- `W_out` — output embeddings  

---

### 4. Training Logic

For each `(target, context)` pair:

**Positive sample:**
```
maximize log σ(u · v)
```

**Negative samples:**
```
minimize log σ(-u · v)
```

**Loss function:**
```
L = -log σ(u_pos · v) - Σ log σ(-u_neg · v)
```

**Gradients:**
```
Positive: grad = σ(u · v) - 1
Negative: grad = σ(u · v)
```

---

### 5. Negative Sampling

```python
def get_negative_samples(vocab_size, k):
    return np.random.randint(0, vocab_size, size=k)
```

---

### 6. Learning Rate

```python
lr = max(lr_start * (1 - current_step / total_steps), lr_start * 0.0001)
```

---

### 7. Similarity Evaluation

Cosine similarity is used to find similar words.

---

## 📂 Dataset

### *** Small test dataset ***

```
king is a strong man
queen is a wise woman
boy is a young man
girl is a young woman

prince is a young king
princess is a young queen

man is strong
woman is beautiful

king and queen rule the kingdom
prince and princess are royal children

man woman boy girl king queen prince princess
```

### *** Real dataset ***

- Crime and Punishment — Fyodor Dostoevsky

---

## ▶️ How to Run

```
pip install -r requirements.txt
```

Put your dataset in:

```
data/text.txt
```

Run:

```
python main.py
```

---

## 📊 Example Output

```
Epoch: 4 Loss: 860.45

Similar to king:
[('woman', 0.96), ('man', 0.96), ('queen', 0.95), ...]
```

---

## ⚠️ Limitations

- Common stopwords such as `"is"`, `"the"`, and `"a"` dominate similarity queries due to high frequency  
- These words are filtered during similarity evaluation  

- Negative sampling is uniform (not unigram^0.75)  
- Small dataset → noisy embeddings  
- No subsampling of frequent words  

---

## 🚀 Improvements

- Unigram^0.75 negative sampling  
- Subsampling frequent words  
- CBOW implementation  
- Larger datasets (Wikipedia)  
- Embedding visualization (PCA / t-SNE)  

---

## 📚 References

- Mikolov et al. — *Distributed Representations of Words and Phrases and their Compositionality*

---

## ✅ Summary

✔ Forward pass implemented manually  
✔ Loss function derived and implemented  
✔ Gradients computed manually  
✔ SGD updates applied  

All using **pure NumPy**.
