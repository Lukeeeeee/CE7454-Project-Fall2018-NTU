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
import numpy
from pylab import rcParams

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


def plot_single_loss(log_folder, single=True, avg=False):
    loss_path = os.path.join(log_folder, 'loss.json')
    x, y = _retrieve_info(loss_path, single, avg=avg)
    if avg:
        save_name = 'loss_val_single_avg'
        x_label = 'epoch'
    else:
        save_name = 'loss_val_single'
        x_label = 'step'
    plot_line_graph(x, y,
                    title='loss',
                    x_label=x_label,
                    y_label='loss',
                    save_name=save_name,
                    log_dir=log_folder)


def plot_valid_loss(log_folder, single=False, avg=False):
    loss_path = os.path.join(log_folder, 'loss.json')
    x, y = _retrieve_info(loss_path, single, avg=avg)
    if avg:
        save_name = 'loss_val_twice_avg'
        x_label = 'epoch'
    else:
        save_name = 'loss_val_twice'
        x_label = 'step'

    plot_line_graph(x, y,
                    title='loss',
                    x_label=x_label,
                    y_label='loss',
                    save_name=save_name,
                    log_dir=log_folder)


def _retrieve_info(loss_path, single, avg=False):
    with open(loss_path, mode='r') as rfile:
        test_data = json.load(rfile)

    if type(test_data) == dict:
        pass
    elif type(test_data) == list:
        x, y_loss, y_valid_loss = get_info_from_loss_list(test_data, avg=avg)
    if single:
        return x, [y_loss]
    else:
        return x, [y_loss, y_valid_loss]


def plot_line_graph(x, y, title, x_label, y_label, save_name, log_dir, marker='*'):
    # rcParams['figure.figsize'] = 4, 3
    rcParams['figure.figsize'] = 8, 6
    sns.set_style("darkgrid")
    # plt.figure(fig_num)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    marker_every = max(int(len(x) / 10), 1)
    color_id = 1
    for item in y:
        ax, = plt.plot(x, item[0], color_list[color_id], label=item[1], marker=marker, markevery=marker_every,
                       markersize=6,
                       linewidth=1)
        color_id += 1

    plt.legend()
    if save_name:
        plt.savefig(os.path.join(log_dir, save_name + '.jpg'))
        plt.savefig(os.path.join(log_dir, save_name + '.pdf'))

    plt.show()


def plot_miou(log_folders, folder_type):
    params = []
    y = []
    x = []
    for folder in log_folders:
        eval_json = os.path.join(folder, 'eval.json')
        config_json = os.path.join(folder, 'config.json')
        with open(config_json) as rfile:
            config = json.load(rfile)
        if folder_type == "lambda":
            params.append("%s,%s,%s" % (config['LAMBDA1'], config['LAMBDA2'], config['LAMBDA3']))
            legend_title = 'Lambda'
        elif folder_type == "epoch":
            params.append(config['TRAINING_EPOCHS'])
            legend_title = 'Training Epoch'
        elif folder_type == "lr":
            params.append(config['LEARNING_RATE'])
            legend_title = 'Learning Rate'
        x.append(folder)
        with open(eval_json) as rfile:
            eval_ = json.load(rfile)
        y.append(eval_['FINAL_MIOU'])

    plot_bar_chart(x, y, params,
                   title='MIOU',
                   x_label=legend_title,
                   y_label='MIOU',
                   save_name='miou' + "_bar_" + folder_type,
                   log_dir='./log',
                   legend_title=legend_title)

    if legend_title == 'Lambda':
        return
    new_x, new_y = zip(*sorted(zip(params, y)))
    new_x = list(new_x)
    new_y = list(new_y)

    plot_line_graph(new_x, [(new_y, legend_title)],
                    title='MIOU',
                    x_label=legend_title,
                    y_label='loss',
                    save_name='miou' + "_line_" + folder_type,
                    log_dir='./log')


def plot_bar_chart(x, y, params, title, x_label, y_label, save_name, log_dir, marker='*', legend_title=''):
    # rcParams['figure.figsize'] = 4, 3
    rcParams['figure.figsize'] = 8, 6
    sns.set_style("darkgrid")
    # plt.figure(fig_num)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    ax1 = plt.subplot(111)
    ax1 = plt.figure().add_axes([0.1, 0.1, 0.6, 0.75])
    for j in range(len(y)):
        ax1.bar(x[j], y[j], width=0.8, bottom=0.0, align='center', color=color_list[j], alpha=0.6,
                label=params[j])
    ax1.xaxis.set_ticklabels([])

    plt.legend(loc="upper left", bbox_to_anchor=[1, 1],
               ncol=2, shadow=True, title=legend_title, fancybox=True)
    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    if save_name:
        plt.savefig(os.path.join(log_dir, save_name + '.jpg'))
        # plt.savefig(os.path.join(log_dir, save_name + '.pdf'))
    # plt.show()


def get_info_from_loss_list(test_data, avg=False):
    x = []
    x_epoch_losses = dict()
    x_epoch = []
    avg_losses = []
    avg_v_losses = []
    y_loss = []
    y_valid_loss = []
    for item in test_data:
        y_loss.append(item['LOSS_VALUE'])
        y_valid_loss.append(item['VALIDATION_LOSS_VALUE'])
        x.append(item['EPOCH'] * 253 + item['STEP'])
        if item['EPOCH'] not in x_epoch_losses:
            x_epoch_losses[item['EPOCH']] = dict()
            x_epoch_losses[item['EPOCH']]['LOSS_VALUE'] = []
            x_epoch_losses[item['EPOCH']]['VALIDATION_LOSS_VALUE'] = []
        if item['EPOCH'] not in x_epoch:
            x_epoch.append(item['EPOCH'])
        x_epoch_losses[item['EPOCH']]['LOSS_VALUE'].append(item['LOSS_VALUE'])
        x_epoch_losses[item['EPOCH']]['VALIDATION_LOSS_VALUE'].append(item['VALIDATION_LOSS_VALUE'])
    for key in x_epoch:
        avg_loss = numpy.mean(x_epoch_losses[key]['LOSS_VALUE'])
        avg_v_loss = numpy.mean(x_epoch_losses[key]['VALIDATION_LOSS_VALUE'])
        avg_losses.append(avg_loss)
        avg_v_losses.append(avg_v_loss)

    if avg:
        return x_epoch, (avg_losses, 'Average Training Loss'), (avg_v_losses, 'Average Validation Loss')
    else:
        return x, (y_loss, 'Training Loss'), (y_valid_loss, "Validation Loss"),


def main():
    # Plot three type of plot
    log_folders = get_log_folder("/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/")
    for path in log_folders:
        print("Plotting %s" % path)
        # plot_single_loss(path, avg=True)
        # plot_valid_loss(path, avg=True)
        # plot_single_loss(path, avg=False)
        # plot_valid_loss(path, avg=False)
    lambda_folders, epoch_folders, lr_folders = split_log_folders(log_folders)

    print("Plotting Miou")
    plot_miou(lambda_folders, 'lambda')
    plot_miou(epoch_folders, 'epoch')
    plot_miou(lr_folders, 'lr')


def split_log_folders(log_folders):
    lf = []
    ef = []
    lrf = []
    for item in log_folders:
        if 'DEFAULT_CONFIG_LAMBDA' in item:
            lf.append(item)
        elif 'DEFAULT_CONFIG_LR' in item:
            if '2018-11-08_12-41-35__v2_DEFAULT_CONFIG_LR_0.000500' in item:
                continue
            lrf.append(item)
        elif 'DEFAULT_CONFIG_EPOCH' in item:
            ef.append(item)
    return lf, ef, lrf


def get_log_folder(main_dir):
    folders = []
    for f in os.listdir(main_dir):
        f_path = os.path.join(main_dir, f)
        if os.path.exists(os.path.join(f_path, 'loss.json')):
            folders.append(f_path)
    return folders


if __name__ == "__main__":
    # main()
    x = [1, 2, 3, 4]
    y = [0.88124263, 0.9297373296069]
    params = ['IC-net', 'Ensemble']
    legend_title = ''
    save_name = 'ensemble'

    plot_bar_chart(x, y, params,
                   title='Ensemble Experiments',
                   x_label=legend_title,
                   y_label='MIOU',
                   save_name=save_name,
                   log_dir='/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/report_fig',
                   legend_title=legend_title)
