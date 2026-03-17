import numpy as np

class Word2Vec:
    def __init__(self,vocabulary_size,embedding_dim):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.randn(vocabulary_size,embedding_dim)*0.01
        self.W_out=np.random.randn(vocabulary_size,embedding_dim)*0.01

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def train_pair(self,target,context,negative_samples,lr):

        loss=0
        v=self.W_in[target].copy()
        u = self.W_out[context].copy()

        score = self.sigmoid(np.dot(u,v))
        loss -= np.log(score+1e-10)
        grad = score - 1

        grad_out= grad * v
        grad_in = grad * u

        self.W_out[context] -= lr * grad_out
        self.W_in[target] -= lr * grad_in

        for neg in negative_samples:
            u = self.W_out[neg].copy()
            score = self.sigmoid(np.dot(u, v))

            loss -= np.log(1 - score + 1e-10)

            grad = score

            grad_out = grad * v
            grad_in = grad * u

            self.W_out[neg] -= lr * grad_out
            self.W_in[target] -= lr * grad_in

        return loss

'''
    def train_pair(self,target,context,negative_samples,lr):

        loss=0
        v=self.W_in[target].copy()

        score = self.sigmoid(np.dot(self.W_out[context],v))
        loss -= np.log(score+1e-10)
        grad = (score - 1)* self.W_out[context]
        self.W_out[context] -= lr * (score-1) * v

        # grad_out= grad * v
        # grad_in = grad * self.W_out[context]
        #
        # self.W_out[context] -= lr * grad_out
        # self.W_in[target] -= lr * grad_in

        for neg in negative_samples:
            score = self.sigmoid(np.dot(self.W_out[neg], v))
            loss -= np.log(1 - score + 1e-10)

            grad = score * self.W_out[neg]

            self.W_out[neg]-=lr*score*v
            # grad_out = grad * v
            # grad_in = grad * self.W_out[neg]
            #
            # self.W_out[neg] -= lr * grad_out
            # self.W_in[target] -= lr * grad_in

        self.W_in[target] -= lr*grad
        return loss'''