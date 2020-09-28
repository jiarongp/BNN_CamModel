import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
plt = matplotlib.pyplot
from matplotlib.backends import backend_agg
color_palette = sns.color_palette()
fz = 20

def histogram(data, labels, title, xlabel, fname):
    plt.figure(figsize=(20, 20))
    sns.set()
    for i, (d, l) in enumerate(zip(data, labels)):
        plt.hist(d, label=l, color=color_palette[i] ,bins=20, alpha=0.3)
    plt.title(title, fontsize=fz)
    plt.xlabel(xlabel, fontsize=fz)
    plt.ylabel("Number of images", fontsize=fz)
    plt.legend(loc=0, fontsize=fz)
    plt.tick_params(axis='both', which='minor', labelsize=fz)
    # plt.xlim(left=-0.0, right=1.05)
    plt.savefig(fname, bbox_inches='tight')
    print("image is saved to {}".format(fname))

# def plot_curve(plotname,
#                 x_ls, y_ls, area_ls,
#                 xlabel, ylabel, 
#                 suptitle, fname, rows=1):
#     cols = len(plotname)
#     fig = plt.figure(figsize=(5*cols, 5*rows))
#     sns.set()
#     for i, (name, x, y, area) in enumerate(zip(plotname, x_ls, y_ls, area_ls)):
#         ax = fig.add_subplot(rows, cols, i+1)
#         ax.plot(x, y, '-',
#                 label='AUROC:{}'.format(area),
#                 lw=2)
#         ax.plot([0, 1], 'k-', lw=2)
#         ax.legend(fontsize=10)
#         ax.set_title(name, fontsize=15)
#         ax.set_xlabel(xlabel, fontsize=15)
#         if i == 0:
#             ax.set_ylabel(ylabel, fontsize=15)
#         ax.grid(True)
#     fig.suptitle(suptitle)
#     fig.savefig(fname, bbox_inches='tight')
#     print("image is saved to {}".format(fname))

def plot_curve(plotname,
                x_list, y_list, area_list,
                xlabel, ylabel, labels,
                suptitle, fname, rows=1):
    cols = len(plotname)
    fig = plt.figure(figsize=(5*cols, 5*rows))
    sns.set()
    for c, (x_ls, y_ls, area_ls, label) in enumerate(zip(x_list, y_list, area_list, labels)):
        for i, (name, x, y, area) in enumerate(zip(plotname, x_ls, y_ls, area_ls)):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.plot(x, y, '-',
                    label='{} AUROC:{}'.format(label, area),
                    lw=2, color=color_palette[c])
            ax.plot([0, 1], 'k-', lw=2)
            ax.legend(fontsize=10)
            ax.set_title(name, fontsize=15)
            ax.set_xlabel(xlabel, fontsize=15)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(True)
    fig.suptitle(suptitle)
    fig.savefig(fname, bbox_inches='tight')
    print("image is saved to {}".format(fname))