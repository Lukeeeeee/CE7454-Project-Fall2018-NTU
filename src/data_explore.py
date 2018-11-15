import math
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from skimage import io


def perform_analysis(type_, path):
    """
    Main Method
    :return:
    """
    type_ = type_  # "valid+test" # [Test+Valid, Train]
    print("Processing Dataset of %s" % type_)
    channel = ['r', 'g', 'b']

    img_ids, means, stds, vars, avg_pixel = examine_train(type_, path)
    return img_ids, means, stds, vars, avg_pixel

    # plot_sb(img_ids, means, title="Mean", chart_type="RGB Mean", type_=type_)
    # plot_sb(img_ids, stds, title="Standard Deviation", chart_type="RGB Standard Deviation", type_=type_)
    # plot_sb(img_ids, vars, title="Variance", chart_type="RGB Variance", type_=type_)
    # # plot_car_percentage(type_, path)
    # plot_3d_bar_chart(avg_pixel, '3D_HeatMap', type_)
    # # print("%s completed.. ############################# \n\n\n" % type_)


def plot_car_percentage(type_, path):
    """
    :return:
    """

    train_folder = os.path.join(path, 'mask')

    # print("%s, %i" % (train_folder, len(os.listdir(train_folder))))
    count = 0
    img_ids = []
    car_percents = []
    for item in os.listdir(train_folder):
        img_path = os.path.join(train_folder, item)
        im = io.imread(img_path, as_gray=False, mode='RGB')
        total_cell = im.shape[0] * im.shape[1]
        car_label = np.count_nonzero(im)
        zero = total_cell - car_label
        car_percent = car_label / float(total_cell) * 100
        img_ids.append(count)
        car_percents.append(car_percent)
        count += 1
    # print("Creating Graph for Car Percentage, Length of Labels and Y: %i, %i" % (len(img_ids), len(car_percents)))
    _plot_bar_chart(img_ids, car_percents, 'Car Percentage in the Image (%s) ' % type_, type_,
                    "Car Percentage in the Image", range=True, c='.')
    _plot_scatter_chart(img_ids, car_percents, 'Car Percentage in the Image (%s) ' % type_, type_,
                        "Car Percentage in the Image", range=True, c='.')
    _plot_histogram(img_ids, car_percents, "Car Percentage Histogram (%s)" % (type_), type_, '', range, c='.')
    # print("\n######################################################################################################\n")


def examine_train(type_, path):
    """
    :return:
    """
    img_ids = []
    means = []
    stds = []
    vars = []
    avg_pixel = []
    total_rgb = ""
    train_folder = os.path.join(path, 'data')
    len_train = len(os.listdir(train_folder))
    # print("%s, %i" % (train_folder, len_train))
    count = 0
    for item in os.listdir(train_folder):
        img_path = os.path.join(train_folder, item)
        img_id = item[:-4]

        im = io.imread(img_path, as_gray=False, mode='RGB')

        img_ids.append(img_id)
        rgb_mean = np.mean(im, axis=0)
        rgb_mean = np.mean(rgb_mean, axis=0)

        rgb_std = np.std(im, axis=0)
        rgb_std = np.std(rgb_std, axis=0)

        rgb_vars = im.transpose(2, 0, 1).reshape(3, -1)
        rgb_vars = np.var(rgb_vars, axis=1)

        means.append(rgb_mean)
        stds.append(rgb_std)
        vars.append(rgb_vars)

        im_copy = im.astype('float32')

        if total_rgb == "":
            total_rgb = im_copy
        else:
            total_rgb = np.add(total_rgb, im_copy)
        count += 1
    # # Compute average RGB for each pixels
    avg_pixel = np.divide(total_rgb, count)

    # Compute average RGB for each channel
    tmp = np.sum(total_rgb, axis=0)
    tmp = np.sum(tmp, axis=0)
    avg_channel = np.divide(tmp, 1918 * 1280)
    avg_channel = np.divide(avg_channel, count)
    print("Data Collection Completed")
    return img_ids, means, stds, vars, avg_pixel


def plot_sb(label, ys, title="", chart_type="", type_='', range=False):
    """
    Plot a bar chart based on x list and y list
    :param xs: A list of x label
    :param ys: A list of y number
    :return:
    """
    # print("Creating Graph for %s, Length of Labels and Y: %i, %i" % (title, len(label), len(ys)))
    # Handling X Axis
    index = np.arange(len(label))

    rs = []
    gs = []
    bs = []
    for item in ys:
        rs.append(item[0])
        gs.append(item[1])
        bs.append(item[2])

    _plot_bar_chart(index, rs, 'R %s Channel (%s)' % (title, type_), type_, chart_type, range, c='r')
    _plot_bar_chart(index, gs, 'G %s Channel (%s)' % (title, type_), type_, chart_type, range, c='g')
    _plot_bar_chart(index, bs, 'B %s Channel (%s)' % (title, type_), type_, chart_type, range, c='b')

    _plot_scatter_chart(index, rs, 'R %s Channel (%s)' % (title, type_), type_, chart_type, range, c='r')
    _plot_scatter_chart(index, gs, 'G %s Channel (%s)' % (title, type_), type_, chart_type, range, c='g')
    _plot_scatter_chart(index, bs, 'B %s Channel (%s)' % (title, type_), type_, chart_type, range, c='b')

    _plot_histogram(index, rs, 'R %s Channel Histogram (%s)' % (title, type_), type_, chart_type, range, c='r')
    _plot_histogram(index, gs, 'G %s Channel Histogram(%s)' % (title, type_), type_, chart_type, range, c='g')
    _plot_histogram(index, bs, 'B %s Channel Histogram(%s)' % (title, type_), type_, chart_type, range, c='b')
    # print("\n######################################################################################################\n")


def _plot_histogram(index, rgbs, title, type_, chart_type, range, c):
    bins = np.linspace(math.ceil(min(rgbs)),
                       math.floor(max(rgbs)),
                       20)  # fixed number of bins

    plt.xlim([min(rgbs) - 5, max(rgbs) + 5])

    plt.hist(rgbs, bins=bins, alpha=0.5)
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Count')
    file_out = "./charts/" + type_.lower() + "/" + c + "/histogram_" + title + ".png"
    # plt.savefig(file_out, dpi=500)
    plt.show();
    # print("Histogram for [%s] stored in %s" % (title, file_out))


def _plot_bar_chart(index, rgbs, title, type_, chart_type, range, c):
    plt.rcdefaults()
    plt.style.use('ggplot')
    plt.plot(figsize=(300, 20))
    plt.bar(index, rgbs, width=1.5)
    plt.xlabel('Image ID', fontsize=5)

    if range:
        plt.ylim(0, 100)
    plt.ylabel(chart_type + ' Values', fontsize=5)
    # plt.xticks(index, label, fontsize=5, rotation=90)
    plt.title(title)
    file_out = "./charts/" + type_.lower() + "/" + c + "/barchart_" + title + ".png"
    # plt.savefig(file_out, dpi=500)
    plt.show();
    # print("Bar Chart for [%s] stored in %s" % (title, file_out))


def _plot_scatter_chart(index, rgbs, title, type_, chart_type, range, c):
    """
    Plot a bar chart based on x list and y list
    :param xs: A list of x label
    :param ys: A list of y number
    :return:
    """
    plt.rcdefaults()
    plt.style.use('ggplot')
    plt.plot(figsize=(300, 20))
    if 'Variance' not in title:
        plt.ylim(0, 255)
    if range:
        plt.ylim(0, 100)

    plt.scatter(index, rgbs, alpha=0.5, s=1)
    plt.xlabel('Image ID', fontsize=5)
    plt.ylabel(chart_type + ' Values', fontsize=5)
    plt.title(title)
    file_out = "./charts/" + type_.lower() + "/" + c + "/scatterplot_" + title + ".png"
    # plt.savefig(file_out, dpi=500)
    # print("Scatter Plot for [%s] stored in %s" % (title, file_out))
    plt.show();


def plot_3d_bar_chart(avg_pixel, title, type_):
    """
    :param avg_pixel:
    :return:
    """
    print("Creating 3D Plot")
    style.use('ggplot')
    x3 = []
    y3 = []
    zr3 = []
    zg3 = []
    zb3 = []
    for x in range(avg_pixel.shape[0]):
        for y in range(avg_pixel.shape[1]):
            x3.append(x)
            y3.append(y)
            zr3.append(avg_pixel[x][y][0])
            zg3.append(avg_pixel[x][y][1])
            zb3.append(avg_pixel[x][y][2])
    _plot_3d(x3, y3, zr3, title, type_, 'R')
    _plot_3d(x3, y3, zg3, title, type_, 'G')
    _plot_3d(x3, y3, zb3, title, type_, 'B')


def _plot_3d(x3, y3, z, title, type_, c):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(y3, x3, z, cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_title("Average " + c + " Value for Each Pixel")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.set_zlabel("Average " + c + " Values")
    file_out = "./charts/" + type_.lower() + "/" + c.lower() + "/" + title + ".png"
    # plt.savefig(file_out, dpi=500)

    # print("3D Plot for [%s] stored in %s" % (title, file_out))
    plt.show();
