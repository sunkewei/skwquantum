import random
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from trainer.DataRetriever import PinzhongData


class TsneTrainer:
    def __init__(self, x_raw_value, y_raw_value, sample_rate=0.2):
        self.X_raw = x_raw_value
        self.y_raw = y_raw_value
        self.sample_rate = sample_rate

    @staticmethod
    def getcolor(vc):
        if vc == 1:
            return 'r'
        if vc == 0:
            return 'b'
        if vc == -1:
            return 'g'

    def train_tsne(self):
        X = []
        y = []
        for idx in range(0, len(self.y_raw)):
            if random.random() < self.sample_rate and self.y_raw[idx] != 0:
                X.append(self.X_raw[idx])
                y.append(self.y_raw[idx])
        X = np.array(X)
        y = np.array(y)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(X)
        return X_tsne, y

    def display(self, X_tsne, y):
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=self.getcolor(y[i]), marker='o')
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == '__main__':
    data = PinzhongData('AP105.CZCE', recalculate=True)
    x_raw = data.X
    y_raw = data.y_class
    trainer = TsneTrainer(x_raw, y_raw, sample_rate=0.2)
    X_tsne, y_tsne = trainer.train_tsne()
    trainer.display(X_tsne, y_tsne)
