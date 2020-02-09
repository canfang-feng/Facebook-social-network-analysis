#!/usr/bin/env python
# coding: utf-8

# In[70]:


# data source originally from http://toreopsahl.com/datasets/
# in the network, each node corresponds to a user of the website 
# and link weights describe the total number of messages exchanged between users.


# In[71]:


import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import binned_statistic


# In[72]:


#Let's read the network file
network = nx.read_weighted_edgelist("OClinks_w_undir.edg")
net_name = 'fb_like'
path = './ccdfs_' + net_name + '.png'
base_path = './s_per_k_vs_k_'


# In[79]:


# get the node with maximum degree in the network
nodes=network.nodes()
degree=nx.degree_centrality(network)
import operator
node=sorted(degree.items(),key=operator.itemgetter(1),reverse=True)[0][0]
sub_net=network.subgraph("node")


# In[80]:


# visulize the biggent subgraph in the newtwork
nodes=nx.node_connected_component(network,"103")
sub_net=network.subgraph(nodes)

fig= plt.figure()
ax = fig.add_axes([0, 0, 1, 0.9])
nx.draw_networkx(sub_net,with_labels=False,node_color='b', node_size=4, edge_color='y',edge_size=1,alpha=0.4)
fig.savefig("the social network sample in facebook")


# In[ ]:


def create_linbins(start, end, n_bins):
    """
    Creates a set of linear bins.

    Parameters
    -----------
    start: minimum value of data, int
    end: maximum value of data, int
    n_bins: number of bins, int

    Returns
    --------
    bins: a list of linear bin edges
    """
    bins = np.linspace(start, end, n_bins)
    return bins


# In[ ]:


def create_logbins(start, end, n_log, n_lin=0):
    """
    Creates a combination of linear and logarithmic bins: n_lin linear bins 
    of width 1 starting from start and n_log logarithmic bins further to
    max.

    Parameters
    -----------
    start: starting point for linear bins, float
    end: maximum value of data, int
    n_log: number of logarithmic bins, int
    n_lin: number of linear bins, int

    Returns
    -------
    bins: a list of bin edges
    """
    if n_lin == 0:
        bins = np.logspace(np.log10(start), np.log10(end), n_log)
    elif n_lin > 0:
        bins = np.array([start + i for i in range(n_lin)] + list(np.logspace(np.log10(start + n_lin), np.log10(end), n_log)))
    return bins


# In[ ]:


############# Complementary cumulative distribution
 """
 Before performing more sophisticated analysis, it is always good to get 
 some idea on how the network is like. To this end, plot the complementary
 cumulative distribution (1-CDF) for node degree *k*, node strength *s* and
 link weight *w*
 """


# In[17]:


def get_link_weights(net):

    """
    Returns a list of link weights in the network.

    Parameters
    -----------
    net: a networkx.Graph() object

    Returns
    --------
    weights: list of link weights in net
    """
              
    weights=[]
    dic=network.edges(data=True)

    for i,j,data in dic:
        weights.append(data["weight"])
    
    return weights


# In[18]:


def plot_ccdf(datavecs, labels, xlabel, ylabel, num, path):

    """
    Plots in a single figure the complementary cumulative distributions (1-CDFs)
    of the given data vectors.

    Parameters
    -----------
    datavecs: data vectors to plot, a list of iterables
    labels: labels for the data vectors, list of strings
    styles = styles in which plot the distributions, list of strings
    xlabel: x label for the figure, string
    ylabel: y label for the figure, string
    num: an id of the figure, int or string
    path: path where to save the figure, string
    """
    styles = ['-', '--', '-.']
    fig = plt.figure(num)
    ax = fig.add_subplot(111)
    for datavec, label, style in zip(datavecs,labels, styles):
        
        sorted_datavec =np.sort(datavec)
        
        ccdf=np.linspace(1,1./len(datavec),len(datavec))
        ax.plot(sorted_datavec,ccdf,style,label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=0)
    ax.grid()

    return fig


# In[24]:


# Get the node degrees and strengths
degrees = []
strengths = []

# get network degrees
#get network strength
degrees=network.degree()
strengths=nx.degree(network,weight="weight")

#Now, convert the degree and strength into lists.
degree_vec = []
strength_vec = []
for node in network.nodes():
    degree_vec.append(degrees[node])
    strength_vec.append(strengths[node])

# Then, compute the weights
weights = get_link_weights(network)

# Now  create 1-CDF plots
datavecs = [degree_vec, strength_vec, weights]
num = 'a)' + net_name # figure identifier

#set the correct labels
labels = ['node degree k distribution', 'node strengths s distribution ', 'link weight w distribution '] ### TIP: set the proper labels
xlabel = 'x' 
ylabel = '1-CDF(x)' 

fig=plot_ccdf(datavecs, labels, xlabel, ylabel, num, path)
fig.savefig(path)
print('1-CDF figure saved to ' + path)


# In[25]:


# average link weight per node
av_weight = []
#Tcalculate average link weight per node
av_weight=np.array(strength_vec)/np.array(degree_vec)


# In[ ]:


############# create bin-averaged versions of the plots
 """
 The large variance of the data can make the scatter plots a bit messy. 
 To make the relationship between  ⟨𝑤⟩  and  𝑘  more visible, create bin-averaged
 versions of the plots, i.e. divide nodes into bins based on their degree and 
 calculate the average  ⟨𝑤⟩  in each bin. Now, you should be able to spot a 
 trend in the data
 """


# In[28]:


n_bins = 20 # use the number of bins that find reasonable
min_deg = min(degree_vec)
max_deg = max(degree_vec)
linbins = create_linbins(min_deg, max_deg, n_bins)
logbins = create_logbins(0.5, max_deg, n_bins, n_lin=10)
print(logbins)
print(max_deg)
num = 'b) ' + net_name + "_"
alpha = 0.1 # transparency of data points in the scatter

for bins, scale in zip([linbins, logbins], ['linear', 'log']):
    fig = plt.figure(num + scale)
    ax = fig.add_subplot(111)
    # mean degree value of each degree bin
    degree_bin_means, _, _ = binned_statistic(degree_vec,degree_vec,statistic="mean",bins=bins)
    #  use binned_statistic to get mean degree of each bin 
    
    # mean strength value of each degree bin
    strength_bin_means, _, _ = binned_statistic(degree_vec,strength_vec,statistic="mean",bins=bins)
    # use binned_statistic to get mean strength of each bin)

    # number of points in each degree bin
    counts, _, _ = binned_statistic(degree_vec,strength_vec, 
                                    'count', 
                                    bins=bins)
    # use binned_statistic to get number of data points

    # b: plotting all points (scatter)
    ax.scatter(degree_vec, av_weight, marker='o', s=1.5, alpha=alpha)
    # calculating the average weight per bin
    bin_av_weight = strength_bin_means / degree_bin_means

    # c: and now, plotting the bin average
    # the marker size is scaled by number of data points in the bin
    ax.scatter(degree_bin_means,
                bin_av_weight,
                marker='o',
                color='g',
                s=np.sqrt(counts) + 1,
                label='binned data')
    ax.set_xscale(scale)
    min_max = np.array([min_deg, max_deg])
    ax.set_xlabel('degree k ') 
    ax.set_ylabel('avg.link weight s ') 
    ax.grid()

    ax.legend(loc='best')
    plt.suptitle('avg. link weight vs. strength:' + net_name)
    save_path = base_path + scale + '_' + net_name + '.png'
    fig.savefig(save_path)
    print('Average link weight scatter saved to ' + save_path)

