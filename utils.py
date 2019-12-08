import numpy as np
import matplotlib.pyplot as plt


def plot_examples(n_samples, cover_images, encoded_images):
    n_cols = 3
    index = 0
    for j in range(0, n_samples):
        img_cover = cover_images[j:j+1, :, :, :]
        img_encoded = encoded_images[j]
        img_encoded = np.squeeze(img_encoded, -1)
        img_cover = np.squeeze(img_cover)
        img_diff = np.abs(img_cover - img_encoded)
        plt.subplot(n_samples, n_cols, index + 1)
        plt.imshow(img_encoded, cmap='gray')
        plt.axis('off')
        plt.subplot(n_samples, n_cols, index + 2)
        plt.imshow(img_cover, cmap='gray')
        plt.axis('off')
        plt.subplot(n_samples, n_cols, index + 3)
        plt.imshow(img_diff, cmap='gray')
        plt.axis('off')
        index += 3
