from datasets import DatasetDict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def balance_and_create_dataset(original_dataset, n_train, n_test):
    def get_balanced_split(split, n_samples):
        pos_indices = [i for i, example in enumerate(split) if example['label'] == 1][:n_samples // 2]
        neg_indices = [i for i, example in enumerate(split) if example['label'] == 0][:n_samples // 2]
        balanced_indices = pos_indices + neg_indices
        return split.select(balanced_indices)

    balanced_train_split = get_balanced_split(original_dataset['train'], n_train)
    balanced_test_split = get_balanced_split(original_dataset['test'], n_test)

    balanced_dataset = DatasetDict({
        'train': balanced_train_split,
        'test': balanced_test_split
    })

    return balanced_dataset


def plot_animation(data):
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], c=[], cmap='coolwarm', alpha=0.7)
    txt = ax.text(0.8, 0.9, '', transform=ax.transAxes)

    # find max and min values for x and y
    max_x = max([np.max(data[epoch][0][:, 0]) for epoch in data])
    min_x = min([np.min(data[epoch][0][:, 0]) for epoch in data])
    max_y = max([np.max(data[epoch][0][:, 1]) for epoch in data])
    min_y = min([np.min(data[epoch][0][:, 1]) for epoch in data])

    def init():
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)
        return sc, txt

    def update(epoch):
        embeddings, labels = data[epoch]
        sc.set_offsets(embeddings)
        sc.set_array(np.array(labels))
        txt.set_text(f'Epoch {epoch + 1}')
        return sc, txt

    ani = FuncAnimation(fig, update, frames=sorted(data.keys()), init_func=init, blit=True, interval=500)

    ani.save('assets/animation2.gif', writer='imagemagick', fps=1.5)
