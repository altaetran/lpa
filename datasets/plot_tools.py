import seaborn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas
import matplotlib.cm
import matplotlib.colors as mcolors

helix_blue = [184./256, 217./256, 223./256, 1]
light_helix_blue = [171./256, 233./256, 255./256, 1]
dark_helix_blue = [48./256, 55./256, 95./256, 1]
light_helix_red = [204./256,86./256,54./256,1]
dark_helix_red = [81./256,45./256,67./256,1]
light_helix_yellow = [224./256, 213./256, 0./256, 1]
dark_helix_yellow = [44./256, 66./256, 0./256, 1]

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    
    Solution found at: http://stackoverflow.com/q/16834861/190597
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def init():
    seaborn.set(font='Helvetica')
    seaborn.set_style('white')
    seaborn.set_style("ticks")

def set_spine_style(ax):
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

def get_cmap_theme(theme='blue'):
    if theme == 'blue':
        cmap = seaborn.cubehelix_palette(50, start=0.0, rot=-0.2, light=0.6, as_cmap=True)
        darken = 1.5
    elif theme == 'purple':
        #cmap = seaborn.cubehelix_palette(50, start=0.5, rot=-0.3, light=0.5, as_cmap=True)        
        cmap = seaborn.cubehelix_palette(light=0.6,as_cmap=True)
        darken = 1.5
    elif theme == 'red':
        cmap = seaborn.cubehelix_palette(50, start=0.7, rot=0.1, light=0.6, as_cmap=True)        
        darken = 1.5
    elif theme == 'yellow':
        cmap = seaborn.cubehelix_palette(50, start=1.9, rot=0.3, light=0.6, dark=0.15, as_cmap=True)        
        darken = 1.5
    elif theme == 'green':
        cmap = seaborn.cubehelix_palette(rot=-.4, light=0.6, as_cmap=True)
        darken = 1.5
    elif theme == 'bone':
        cmap = matplotlib.cm.get_cmap('YlGn')
        darken = 1.0
    elif theme == 'bkr':
        cmap = make_colormap(
            [light_helix_blue[:3], dark_helix_blue[:3],0.35, 
             dark_helix_blue[:3], [0,0,0], 0.5, 
             [0,0,0], [0,0,0], 0.5,
             [0,0,0], dark_helix_red[:3], 0.65,
             dark_helix_red[:3], light_helix_red[:3]])
        darken = 1.0
    elif theme == 'bwr':
        cmap = make_colormap(
            [light_helix_blue[:3], [1,1,1], 0.55, 
             [1,1,1], [1,1,1], 0.55,
             [1,1,1], light_helix_red[:3]])
        darken = 1.0
    elif theme == 'bwy':
        cmap = make_colormap(
            [dark_helix_blue[:3], light_helix_blue[:3],0.35, 
             light_helix_blue[:3], [1,1,1], 0.5, 
             [1,1,1], [1,1,1], 0.5,
             [1,1,1], light_helix_yellow[:3], 0.65,
             light_helix_yellow[:3], dark_helix_yellow[:3]])
        darken = 1.0

    return cmap, darken

def plot_series(t,y_to_plot,sort_i=0,xlabel='',ylabel='',title='',linewidth=1.0,
                figsize=(6,4),labelsize=7,titlesize=8,savelocation=None,ax=None,
                sort_idx=None,
                theme='blue'):
    init()
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.0, 0.0, 1, 1]) 
        plt.sca(ax)

    if sort_idx is None:
        sort_idx = np.argsort(y_to_plot[:,sort_i], axis=0)
    y_to_plot = y_to_plot[sort_idx,:]

    color_idx = np.linspace(0, 1, y_to_plot.shape[0])

    cmap, darken = get_cmap_theme(theme)

    legend_items = []
    for i,cidx in enumerate(color_idx):
        c = cmap(cidx)

        c = (np.min([c[0]*darken,1]), np.min([c[1]*darken,1]), np.min([c[2]*darken,1]), 1)

        plt.plot(t, y_to_plot[i,:],c=c, linewidth=linewidth)

    plt.xlim([t[0],t[-1]])       
    set_spine_style(ax)

    plt.tick_params(axis='both', which='major', labelsize=6, pad=2, length=3)

    plt.ylabel(ylabel, fontsize=labelsize, labelpad=2)
    plt.xlabel(xlabel, fontsize=labelsize, labelpad=2)
    plt.title(title, fontsize=titlesize)

    if not (savelocation is None):
        plt.savefig(savelocation,dpi=300)

    return ax

def scatter_series(t,y_to_plot,sort_i=0,xlabel='',ylabel='',title='',linewidth=1.0,
                   figsize=(6,4),labelsize=7,titlesize=8,savelocation=None,ax=None,
                   marker_size=10,
                   sort_idx=None,
                   theme='blue'):
    init()
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.0, 0.0, 1, 1]) 
        plt.sca(ax)

    if sort_idx is None:
        sort_idx = np.argsort(y_to_plot[:,sort_i], axis=0)
    y_to_plot = y_to_plot[sort_idx,:]

    color_idx = np.linspace(0, 1, y_to_plot.shape[0])

    cmap, darken = get_cmap_theme(theme)

    legend_items = []
    for i,cidx in enumerate(color_idx):
        c = cmap(cidx)

        c = (np.min([c[0]*darken,1]), np.min([c[1]*darken,1]), np.min([c[2]*darken,1]), 0.7)

        plt.scatter(t, y_to_plot[i,:],c=c,s=marker_size)

    plt.xlim([t[0],t[-1]])       
    set_spine_style(ax)

    plt.tick_params(axis='both', which='major', labelsize=6, pad=2, length=3)

    plt.ylabel(ylabel, fontsize=labelsize, labelpad=2)
    plt.xlabel(xlabel, fontsize=labelsize, labelpad=2)
    plt.title(title, fontsize=titlesize)

    if not (savelocation is None):
        plt.savefig(savelocation,dpi=300)

    return ax

def show():
    plt.show()

def vert_line(ax=None,xpos=0,linewidth=1.0,linestyle='--',color='k',alpha=0.7):    
    plt.axvline(xpos,axes=ax,color=color,linestyle=linestyle,linewidth=linewidth,alpha=alpha)

def pos_heatmap(M_plot,ynames,xnames,figsize=(0.5,0.5),cbar=True,
            savelocation=None,vmin=0,vmax=None):
    init()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.0, 0.0, 1, 1])
    plt.sca(ax)

    if vmax is None:
        vmax = np.max(M_plot)
    cmap = seaborn.cubehelix_palette(50, start=0.0, rot=-0.2, light=1, as_cmap=True)
    cm_data = pandas.DataFrame(M_plot, index=ynames, columns=xnames)
    seaborn.heatmap(cm_data, linewidths=0.25, cmap=cmap, vmin=vmin,vmax=vmax, ax=ax,cbar=cbar)

    plt.yticks(rotation=0)

    plt.tick_params(axis='both', which='major', labelsize=6, pad=2, length=3)
    set_spine_style(ax)

    if not (savelocation is None):
        plt.savefig(savelocation,dpi=300)

    plt.show()

def heatmap(M_plot,ynames,xnames,figsize=(0.5,0.5),cbar=True,
            savelocation=None,vmin=None,vmax=None,theme=None):
    init()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.0, 0.0, 1, 1])
    plt.sca(ax)
    
    if vmax is None:
        vmax = np.max(M_plot)
    if vmax is None:
        vmin = np.min(M_plot)

    #cmap = seaborn.cubehelix_palette(50, start=0.0, rot=-0.2, light=1, as_cmap=True)
    if theme is None:
        cmap = seaborn.diverging_palette(255, 133, l=60, n=7, center="light", as_cmap=True)
    else:
        cmap, darken = get_cmap_theme(theme)
    cm_data = pandas.DataFrame(M_plot, index=ynames, columns=xnames)
    seaborn.heatmap(cm_data, linewidths=0.25, cmap=cmap, vmin=vmin,vmax=vmax, ax=ax,cbar=cbar)

    plt.yticks(rotation=0)

    plt.tick_params(axis='both', which='major', labelsize=6, pad=2, length=3)
    set_spine_style(ax)

    if not (savelocation is None):
        plt.savefig(savelocation,dpi=300)

    plt.show()
