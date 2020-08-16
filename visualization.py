from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data_preparation as dp
import params
import os
import functools
from matplotlib import offsetbox
from sklearn import manifold, datasets
from skimage import io
from openTSNE import TSNE
AUTOTUNE = tf.data.experimental.AUTOTUNE
num_examples = 256

def aligned_ds(test_dir, brand_models, num_batches=None, seed=42):
    np.random.seed(seed)
    image_paths, num_images, ds = [], [], []
    for model in brand_models:
        images = os.listdir(os.path.join(test_dir, 'test', model))
        paths = [os.path.join(test_dir, 'test', model, im) for im in images]
        image_paths.append(paths)
        num_images.append(len(images))
    for model, images in zip(params.brand_models, image_paths):
        np.random.shuffle(images)
        ds.extend(images[0:num_examples])
    return ds

def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

ds = aligned_ds(params.patch_dir, 
                params.brand_models)
one_ds = ds[:num_examples]
two_ds = ds[num_examples:]

one_dataset = (tf.data.Dataset.from_tensor_slices(one_ds)
                .repeat()
                .map(dp.parse_image, num_parallel_calls=AUTOTUNE))
two_dataset = (tf.data.Dataset.from_tensor_slices(two_ds)
                .repeat()
                .map(dp.parse_image, num_parallel_calls=AUTOTUNE))
one_jpeg = (tf.data.Dataset.from_tensor_slices(one_ds)
                .map(functools.partial(dp.parse_image, post_processing='jpeg'),
                     num_parallel_calls=AUTOTUNE))
two_jpeg = (tf.data.Dataset.from_tensor_slices(two_ds)
                .map(functools.partial(dp.parse_image, post_processing='jpeg'),
                     num_parallel_calls=AUTOTUNE))
one_iter = iter(one_dataset)
two_iter = iter(two_dataset)
one_jpeg_iter = iter(one_jpeg)
two_jpeg_iter = iter(two_jpeg)

one_images, two_images = [], []
one_jpeg_images, two_jpeg_images = [], []
for i in range(num_examples):
    one_im = one_iter.get_next()[0].numpy()[:, :, 0].flatten()
    two_im = two_iter.get_next()[0].numpy()[:, :, 0].flatten()
    one_jpeg_im = one_jpeg_iter.get_next()[0].numpy()[:, :, 0].flatten()
    two_jpeg_im = two_jpeg_iter.get_next()[0].numpy()[:, :, 0].flatten()
    one_images.append(one_im)
    two_images.append(two_im)
    one_jpeg_images.append(one_jpeg_im)
    two_jpeg_images.append(two_jpeg_im)

X = np.concatenate((np.asarray(one_images), np.asarray(two_images),
                    np.asarray(one_jpeg_images), np.asarray(two_jpeg_images)))
y = np.int64(np.concatenate((np.zeros(num_examples), np.ones(num_examples), 
                             2 * np.ones(num_examples), 3 * np.ones(num_examples))))
# n_samples, n_features = X.shape
# n_neighbors = 30

# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# t0 = time()
# X_tsne = tsne.fit_transform(X)

tsne = TSNE(
    perplexity=30,
    metric="euclidean",
#     callbacks=ErrorLogger(),
    verbose=True,
    n_jobs=8,
    random_state=42,
)

embedding_train = tsne.fit(X)
plot(embedding_train, y, s=10)