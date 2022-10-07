import os
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']=200
import numpy as np
import imageio
from dataclasses import dataclass

@dataclass
class Bound2D:
    """Class for keeping track of an item in inventory."""
    xmin: float = 95
    xmax: float = 110
    ymin: float = 97
    ymax: float = 110
    xlabel: str = "x"
    ylabel: str = "y"

def plot_4D_CS(func, z_plot, z_label='z', bound2D=Bound2D(), wbounds=None, color_num=21, path=None, func_name=None):
    # plot a 4D function w(x,y,z) at z=z_plot with a countour plot with a certain color scheme
    # if wbounds is not fixed then will be auto-fitted
    xmin, xmax = bound2D.xmin, bound2D.xmax
    ymin, ymax = bound2D.ymin, bound2D.ymax
    x = np.linspace(xmin, xmax, color_num)
    y = np.linspace(ymin, ymax, color_num)
    w = np.array([func(i,j,z_plot) for j in y for i in x])

    X, Y = np.meshgrid(x, y)
    W = w.reshape(color_num, color_num)
    
    if wbounds == None:
        wbounds = (W.min(), W.max())
    wmin, wmax = wbounds
    levels = np.linspace(wmin, wmax, color_num)
    img=plt.contourf(X, Y, W, levels=levels)
    plt.colorbar(img)
    
    plt.xlabel(bound2D.xlabel)
    plt.ylabel(bound2D.ylabel)
    if func_name == None:
        func_name = "Crosssection"
    title_str = "%s at %s=%.2f"%(func_name, z_label, z_plot)
    plt.title(title_str)
    
    if path == None:
        save_path = "Figures/GPRplots/test/"
    else:
        save_path = path
        save_path += '/' if path[-1]!= '/' else ''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = save_path + title_str + ".png"
    plt.savefig(filename)
    
    plt.show() if path==None else plt.close()
    return filename

def plot_4D(func, z_values, path=None, func_name=None, z_label='z', save_images=True, color_num = 21, wbounds=(0, 170), bound2D=Bound2D(), fps=1):
    """
    Plot a 4D function w(x,y,z) by z crosssections.

    :param z_values: values of z at cross sections we want to plot at
    :param path: path 
    :param func_name: name of the 4D function we are plotting
    :param z_label: label on the z axis
    :param save_images: if true then save the cross section png files
    :param color_num: how many colors used in counter plot
    :param w_bound: bounds on w to plot
    :param fps: frames per second used in gif
    """
    if path==None:
        path = os.getcwd() + "\test"
    filenames = []
    for z in z_values:
        fn = plot_4D_CS(func, z, func_name=func_name, z_label=z_label, wbounds=wbounds, color_num=color_num, path=path, bound2D=bound2D)
        filenames.append(fn)
    
    gif_name = "Cross Sections" if func_name==None else func_name
    gif_name += ".gif"
    with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            if not save_images:
                os.remove(filename)
    return gif_name