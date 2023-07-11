import pandas as pd
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
import cmocean

def plot_bauble_histogram(data,x,hue=None,
                          bins='auto',binwidth=None,
                          palette=None,
                          cmap=None,
                          xlims = None,
                          adjust_xlims = True,goal_aspect = 1.2,
                          yticks=True,
                          legend=True,
                          legend_kwargs = {},
                          ax=None):
    """ Plot bauble histogram
    
    Plot a histogram that explicitly shows the number of data points in each bin,
    as a 'bauble', stacked on top of the other baubles in each bin. 
    
    NB: The inputs are vaguely modeled after `seaborn`'s inputs, so you'll find
    many of the commands to, e.g., `sns.histplot()` etc. copied here as well.
    
    Parameters
    -------------------
    data : :class:`pandas.DataFrame`
        Input data structure
        
    x : key in ``data``
        Variable that specifies positions on the x axis
        
    hue : None or key in ``data`
        Which column to use to assign colors 
        
    palette : None or list of strs, floats, :class:`matplotlib.colors.Colormap`
        Method for choosing colors when using `hue`. If None,
        then the palette is automatically created using `cmap`. 
        
    cmap : None or :class:`matplotlib.colors.Colormap`, by default cmocean's "phase"
        Which colormap to use to assign colors if no explicit palette
        is applied. If None, then `cmocean.cm.phase` is used as the colormap; 

    bins : str, number, vector, or a pair of such values; by default 'auto'
        Passed to :func:`numpy.histogram_bin_edges`, can be the name of a 
        reference rule, the number of bins, or the breaks of the bins.
        
    binwidth : number; by default `None`
        Alternate to `bins` input; used to explicitly set binwidths
        
    xlims : None or [float,float]
        Used to set custom xlimits if desired. These are used to set 
        binwidths as well, if binwidth is not None. 
        
    adjust_xlims : bool, by default True
        If True, then the aspect ratio of the plot is set to `goal_aspect`
        
    goal_aspect : float, by default 1.2
        If `adjust_xlims==True`, then the width of the plot is set to 
        `goal_aspect*[(max obs in a given bins) + 2]`. 
        
    yticks : bool, by default True
        If False, then yticks (showing the number of obs in a given 
        bin) are removed. 
        
    legend : bool, by default True
        If True, then a legend is added
        
    legend_kwargs : dict, by default {}
        Piped into `ax.legend()`
        
    
    
    Returns
    -------------------
    ax : the map axis
    
    
    Author: Kevin Schwarzwald
    Last update: 07/11/2023
    License: CC BY-SA
    """
    #----------- Input checks -----------
    if type(data) != pd.DataFrame:
        raise NotImplementedError('`data` must be a `pd.DataFrame`, alternate methods not yet supported.')
        
    if type(x) != str:
        raise NotImplementedError('`x` must be a key to `data`, alternate methods not yet supported.')
        
    if hue is not None:
        if type(hue) != str:
            raise NotImplementedError('`hue` must be a key to `data`, alternate methods not yet supported.')
        

    #----------- Setup -----------
    # Get rid of index
    data = data.reset_index()

    # Set placeholder column if no hue is set
    if hue is None:
        data['hue'] = 'all'
        hue = 'hue'
        legend = False
    # Find unique values to color by
    hue_levels = np.unique(data[hue])


    # Set palette if None
    if palette is None:
        if cmap is None:
            palette = cmocean.cm.phase(np.arange(0,len(hue_levels))/len(hue_levels))
        else:
            palette = cmap(np.arange(0,len(hue_levels))/len(hue_levels))

    # Get histogram bin edges
    if xlims is None:
        if binwidth is None:
            bin_edges = np.histogram_bin_edges(data[x],bins=bins)
        else:
            bin_edges = np.arange(data[x].min(),data[x].max(),binwidth)
    else:
        bin_edges = np.arange(*xlims,binwidth)

    # Set binwidth, if not specified
    if binwidth is None:
        binwidth = bin_edges[1]-bin_edges[0]

    # Get empty array for bin population
    bincounts = np.zeros((len(hue_levels),len(bin_edges)-1))

    #----------- Populate bins -----------
    for hue_level,level_idx in zip(hue_levels,np.arange(0,len(hue_levels))):
        bincounts[level_idx,:] = np.histogram(data.loc[data[hue]==hue_level,x],bins=bin_edges)[0]

    #----------- Plot -----------
    if ax is None:
        ax = plt.subplot()

    ## Plot bauble histogram
    # Get one handle per hue level, to use as legend handles
    hdls = [None]*len(hue_levels)
    # Iterate through histogram bins
    for bin_idx in np.arange(0,len(bin_edges)-1):
        # Iterate through hue categories
        for level_idx in np.arange(0,len(hue_levels)):
            # Iterate through number of members of hue category in 
            # a given bin 
            for marker_idx in np.arange(0,bincounts[level_idx,bin_idx]):
                # Set offset from previous hue levels in the same
                # histogram bin
                if level_idx > 0:
                    bauble_offset = np.sum(bincounts[0:level_idx,bin_idx])
                else:
                    bauble_offset = 0

                # Add patch for each point to plot
                hdls[level_idx] = ax.add_patch(mpatches.Ellipse((bin_edges[bin_idx]+binwidth/2,
                                                                    bauble_offset+marker_idx+0.5),
                                                width=0.8*binwidth,height=0.8,facecolor=palette[level_idx]))
    ## Annotate
    # Set axis limits
    if xlims is None:
        ax.set_xlim((bin_edges[0]-binwidth,bin_edges[-1]+binwidth))
        xlims = ax.get_xlim()
    else:
        ax.set_xlim(xlims)
    # Set y limit to 2 bauble-widths above the top 
    ax.set_ylim((0,np.max(np.sum(bincounts,0))+2))
    # Set aspect so the run markers look circular
    ax.set_aspect(binwidth)

    # Adjust xlims to a desired aspect ratio for readabiilty,
    # if desired
    if adjust_xlims:
        ax.set_xlim(np.mean(bin_edges)-goal_aspect*binwidth*(np.max(np.sum(bincounts,0))+2)/2,
                    np.mean(bin_edges)+goal_aspect*binwidth*(np.max(np.sum(bincounts,0))+2)/2)

    # Remove y axis ticks
    if not yticks:
        ax.set_yticks([])

    # Vertical line at 0
    if (xlims[0]<0) and (xlims[-1]>0):
        ax.axvline(0,color='k',zorder=0)

    # Grid and ticks on both sides, for ease of reading
    plt.grid(True)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False,labelsize=10)

    # Set xlabel
    ax.set_xlabel(x)

    ## Legend
    # Add legend
    if legend:
        ax.legend(hdls,hue_levels,ncol=4,**legend_kwargs)
        
    #----------- Return -----------
    return ax