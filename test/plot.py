import numpy as np
import matplotlib

# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import json
import sys
import math as M
from scipy.interpolate import interp1d
from itertools import groupby
import seaborn as sns
import os
import glob
from pylab import rcParams

sns.set_style('ticks')

markers = ('+', 'x', 'v', 'o', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen', 'darkorange', 'oldlace', 'chocolate',
              'purple', 'lightskyblue', 'gray', 'seagreen', 'antiquewhite',
              'snow', 'darkviolet', 'brown', 'skyblue', 'mediumaquamarine', 'midnightblue', 'darkturquoise',
              'sienna', 'lightsteelblue', 'gold', 'teal', 'blueviolet', 'mistyrose', 'seashell', 'goldenrod',
              'forestgreen', 'aquamarine', 'linen', 'deeppink', 'darkslategray', 'mediumseagreen', 'dimgray',
              'mediumpurple', 'lightgray', 'khaki', 'dodgerblue', 'papayawhip', 'salmon', 'floralwhite',
              'lightpink', 'gainsboro', 'coral', 'indigo', 'darksalmon', 'royalblue', 'navy', 'orangered',
              'cadetblue', 'orchid', 'palegreen', 'magenta', 'honeydew', 'darkgray', 'palegoldenrod', 'springgreen',
              'lawngreen', 'palevioletred', 'olive', 'red', 'lime', 'yellowgreen', 'aliceblue', 'orange',
              'chartreuse', 'lavender', 'paleturquoise', 'blue', 'azure', 'yellow', 'aqua', 'mediumspringgreen',
              'cornsilk', 'lightblue', 'steelblue', 'violet', 'sandybrown', 'wheat', 'greenyellow', 'darkred',
              'mediumslateblue', 'lightseagreen', 'darkblue', 'moccasin', 'lightyellow', 'turquoise', 'tan',
              'mediumvioletred', 'mediumturquoise', 'limegreen', 'slategray', 'lightslategray', 'mintcream',
              'darkgreen', 'white', 'mediumorchid', 'firebrick', 'bisque', 'darkcyan', 'ghostwhite', 'powderblue',
              'tomato', 'lavenderblush', 'darkorchid', 'cornflowerblue', 'plum', 'ivory', 'darkgoldenrod', 'green',
              'burlywood', 'hotpink', 'cyan', 'silver', 'peru', 'thistle', 'indianred', 'olivedrab',
              'lightgoldenrodyellow', 'maroon', 'black', 'crimson', 'darkolivegreen', 'lightgreen', 'darkseagreen',
              'lightcyan', 'saddlebrown', 'deepskyblue', 'slateblue', 'whitesmoke', 'pink', 'darkmagenta',
              'darkkhaki', 'mediumblue', 'beige', 'blanchedalmond', 'lightsalmon', 'lemonchiffon', 'navajowhite',
              'darkslateblue', 'lightcoral', 'rosybrown', 'fuchsia', 'peachpuff']


def _retrieve_log(file, key, index, fn=None, x_fn=None, y_fn=None):
    if not x_fn:
        x_fn = lambda x: x[index]
    if not y_fn:
        y_fn = lambda x: x[key]
    x = []
    y = []
    with open(file=file, mode='r') as f:
        test_data = json.load(fp=f)
        if isinstance(test_data, list):
            for sample in test_data:
                if fn:
                    if fn(sample) is True:
                        y.append(y_fn(sample))
                        x.append(y_fn(sample))
                else:
                    x.append(x_fn(sample))
                    y.append(y_fn(sample))
        elif isinstance(test_data, dict):
            if fn:
                if fn(test_data) is True:
                    y.append(y_fn(test_data))
                    x.append(y_fn(test_data))
            else:
                x.append(x_fn(test_data))
                y.append(y_fn(test_data))
    return x, y


def plot_fig(fig_num, col_id, x, y, title, x_lable, y_label, label=' ', marker='*', scatter=False):
    # rcParams['figure.figsize'] = 4, 3
    rcParams['figure.figsize'] = 8, 6
    sns.set_style("darkgrid")
    plt.figure(fig_num)
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_label)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.tight_layout()

    marker_every = max(int(len(x) / 10), 1)
    if len(np.array(y).shape) > 1:
        new_shape = np.array(y).shape

        res = np.reshape(np.reshape(np.array([y]), newshape=[-1]), newshape=[new_shape[1], new_shape[0]],
                         order='F').tolist()
        res = list(res)
        for i in range(len(res)):
            res_i = res[i]
            plt.subplot(len(res), 1, i + 1)
            plt.title(title + '_' + str(i))
            if scatter:
                ax, = plt.scatter(x, res_i, color_list[col_id], label=label + '_' + str(i), marker=marker)
            else:
                ax, = plt.plot(x, res_i, color_list[col_id], label=label + '_' + str(i), marker=marker,
                               markevery=marker_every, markersize=6, linewidth=1)
            col_id += 1
    else:
        if scatter:
            ax, = plt.scatter(x, y, color_list[col_id], label=label, marker=marker)
        else:
            ax, = plt.plot(x, y, color_list[col_id], label=label, marker=marker, markevery=marker_every,
                           markersize=6,
                           linewidth=1)
    plt.legend()

    return ax


def plot_single_loss(log_dir, key, index, scatter_flag=False, fn=None, x_fn=None, y_fn=None, save_name=None):
    x, y = _retrieve_log(file=os.path.join(log_dir, 'loss.json'), key=key, fn=fn, x_fn=x_fn, y_fn=y_fn, index=index)

    plot_fig(fig_num=1,
             col_id=1,
             x=x,
             y=y,
             label='loss_1',
             title='loss',
             x_lable='step',
             y_label='loss',
             scatter=scatter_flag)
    if save_name:
        plt.savefig(os.path.join(log_dir, save_name + '.jpg'))
        plt.savefig(os.path.join(log_dir, save_name + '.pdf'))
    plt.show()


def plot_multi_loss(file_list, label_list, key, index, scatter_flag=False, fn=None, x_fn=None, y_fn=None,
                    save_name=None, loss_file='loss.json'):
    for i, path, label in zip(range(len(file_list)), file_list, label_list):
        x, y = _retrieve_log(file=os.path.join(path, loss_file), key=key, fn=fn, x_fn=x_fn, y_fn=y_fn, index=index)
        plot_fig(fig_num=1,
                 col_id=i,
                 x=x,
                 y=y,
                 label=label,
                 title='loss',
                 x_lable='step',
                 y_label='loss',
                 marker=markers[i],
                 scatter=scatter_flag)

    if save_name:
        for path in file_list:
            plt.savefig(os.path.join(os.path.dirname(path), save_name + '.jpg'))
            plt.savefig(os.path.join(os.path.dirname(path), save_name + '.pdf'))
    plt.show()


def plot_multi_validation_mIoU(file_list, key='FINAL_MIOU'):
    mIoU_list = []

    for i, path in zip(range(len(file_list)), file_list):
        _, y = _retrieve_log(file=os.path.join(path, 'eval.json'), key=key, index=key)
        assert len(y) == 1
        mIoU_list.append(y[0])
    plot_fig(fig_num=1,
             col_id=0,
             x=range(len(file_list)),
             y=mIoU_list,
             label='',
             title='mIoU',
             x_lable='learning_rate',
             y_label='mIoU',
             marker=markers[0],
             scatter=False)
    plt.show()


def _sort_lr_fn(x, key):
    with open(os.path.join(x, 'config.json'), 'r') as f:
        return json.load(f)[key]


if __name__ == '__main__':
    # plot_single_loss(
    #     log_dir='/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-07_19-37-16__v2_DEFAULT_CONFIG_LAMBDA_0.160000_0.400000_1.000000',
    #     index='EPOCH',
    #     key='LOSS_VALUE',
    #     x_fn=lambda x: x['EPOCH'] * 253 + x['STEP'],
    #     scatter_flag=False,
    #     save_name='loss_val'
    # )

    path_list = glob.glob('/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018*LAM*')
    # plot_multi_loss(file_list=path_list,
    #                 label_list=[str(i) for i in range(len(path_list))],
    #                 key='LOSS_VALUE',
    #                 index='EPOCH',
    #                 x_fn=lambda x: x['EPOCH'] * 253 + x['STEP'],
    #                 scatter_flag=False,
    #                 loss_file='eval.json',
    #                 save_name='compare_loss')
    log_list = glob.glob('/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018*LR*')
    log_list.sort(key=lambda x: _sort_lr_fn(x, key='LEARNING_RATE'))
    print("log list\n", log_list)
    plot_multi_validation_mIoU(file_list=log_list)
