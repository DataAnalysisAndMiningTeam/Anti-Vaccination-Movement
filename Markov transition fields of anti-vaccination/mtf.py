# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:36:46 2021

@author: mikij
"""
import os
os.chdir('C:/Users/mikij/Desktop/实验/代码/时间序列/')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.colors as colors

from matplotlib import gridspec
from numba import njit, prange
from pyts.image import MarkovTransitionField

import tsia.plot
import tsia.markov
import tsia.network_graph

#matplotlib inline
plt.style.use('fast')

#加载数据
DATA = 'data'
tag_df = pd.read_csv(os.path.join(DATA, 'antivaccination (British English 2019).csv'))
#tag_df['timestamp'] = pd.to_datetime(tag_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
#tag_df['timestamp'] = pd.to_datetime(tag_df['timestamp'],format='%Y')
tag_df = tag_df.set_index('timestamp')

fig = plt.figure(figsize=(28,4))
plt.plot(tag_df, linewidth=2)
plt.show() #原数据图

#通过pyts获得现成的马尔科夫场
n_bins = 4 #分区
strategy = 'quantile'
X = tag_df.values.reshape(1, -1) #转置
n_samples, n_timestamps = X.shape

mtf = MarkovTransitionField(image_size=48, n_bins=n_bins, strategy=strategy) #马尔可夫转移矩阵
tag_mtf = mtf.fit_transform(X)

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
_, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf[0], ax=ax, reversed_cmap=False)
plt.colorbar(mappable_image); #马尔科夫场图

#分解过程
#信号离散
X_binned, bin_edges = tsia.markov.discretize(tag_df,4)
tsia.plot.plot_timeseries_quantiles(tag_df, bin_edges, label='signal-1')
plt.legend();

#建立马尔可夫转换矩阵
X_mtm = tsia.markov.markov_transition_matrix(X_binned)
#例：第一行 465 |86 |1 |0 |0 |0 |0 |0 |
#第一个单元格表示：bin 0中的465点（值在第一个区间）的下一个值一直位于该bin中。
#第二个单元格表示：bin 0中的86个点过渡到下一个bin（值在第二个区间）

#计算过渡概率
X_mtm = tsia.markov.markov_transition_probabilities(X_mtm)
np.round(X_mtm * 100, 1)
#例：如果给定值X [i]等于130.0，介于118.42和137.42之间，在bin 1中
#下一个值在bin 0的可能性为14.5％ 在bin1的可能性73.4%

#计算马尔可夫转换场
def _markov_transition_field(X_binned, X_mtm, n_timestamps, n_bins):
    X_mtf = np.zeros((n_timestamps, n_timestamps))
    
    # We loop through each timestamp twice to build a N x N matrix:
    for i in prange(n_timestamps):
        for j in prange(n_timestamps):
            # We align each probability along the temporal order: MTF(i,j) 
            # denotes the transition probability of the bin 'i' to the bin 'j':
            X_mtf[i, j] = X_mtm[X_binned[i], X_binned[j]]
            
    return X_mtf

X_mtf = _markov_transition_field(X_binned, X_mtm, n_timestamps, n_bins)
np.round(X_mtf * 100, 1)
#例：要查看M[1,6]，先看X_binned[1]=2和X_binned[6]=1,然后查看转移矩阵从bin2过渡到bin1的概率=10.7%，所以M[1,6]=10.7%

second_row = np.round(X_mtf * 100, 1)[1]
second_row[0:10]

fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(1,1,1)
_, mappable_image = tsia.plot.plot_markov_transition_field(mtf=X_mtf, ax=ax, reversed_cmap=False)
plt.colorbar(mappable_image);#高分辨率MTF（尺寸为数据长度）

#计算汇总的MTF
image_size = 55 #将长度m的每个子序列中的转移概率汇总在一起 在pyts包中自动生成
window_size, remainder = divmod(n_timestamps, image_size)

if remainder == 0:
    X_amtf = np.reshape(
        X_mtf, (image_size, window_size, image_size, window_size)
    ).mean(axis=(1, 3))
    
else:
    # Needs to compute piecewise aggregate approximation in this case. This
    # is fully implemented in the pyts package
    pass
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(1,1,1)
_, mappable_image = tsia.plot.plot_markov_transition_field(mtf=X_amtf, ax=ax, reversed_cmap=False)
plt.colorbar(mappable_image);

#提取有意义的指标
_ = tsia.plot.plot_mtf_metrics(X_amtf)

#将转换概率映射回初始信号
#tsia自带
mtf_map = tsia.markov.get_mtf_map(tag_df, X_amtf, reversed_cmap=True)
_ = tsia.plot.plot_colored_timeseries(tag_df, mtf_map)

#定义colormap
colorslist = ['#80E9E9','#8080FF','#DF80DF','#FF8080']
cmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=800)
cmaps='rainbow'
#自己定义
def plot_colored_timeseries(tag, image_size=96, colormap='cool'):
    # Loads the signal from disk:
    tag_df = pd.read_csv(os.path.join(DATA, f'{tag}.csv'))
    #tag_df['timestamp'] = pd.to_datetime(tag_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
    tag_df = tag_df.set_index('timestamp')

    # Build the MTF for this signal:
    X = tag_df.values.reshape(1, -1)
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins, strategy=strategy)
    tag_mtf = mtf.fit_transform(X)

    # Initializing figure:
    fig = plt.figure(figsize=(28, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,4])

    # Plotting MTF:
    ax = fig.add_subplot(gs[0])
    ax.set_title('Markov transition field')
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf[0], ax=ax, colormap=cmaps,reversed_cmap=False)
    plt.colorbar(mappable_image)
    
    # Plotting signal:
    ax = fig.add_subplot(gs[1])
    ax.set_title(f'Signal timeseries for tag {tag}')
    mtf_map = tsia.markov.get_mtf_map(tag_df, tag_mtf[0],step_size=0,colormap=cmaps,reversed_cmap=False)
    _ = tsia.plot.plot_colored_timeseries(tag_df, mtf_map, ax=ax)

    plt.plot([1998,1998],[50,0],linestyle='--',linewidth=1,color='gray')#在(1998,y0)和(1998,0)两点之间画一条直线
    #plt.text(1997.5,30,'1998: A British doctor',fontsize=8,rotation=90,horizontalalignment="right")
    plt.plot([1853,1853],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1885,1885],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1890,1890],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1907,1907],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1921,1921],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1955,1955],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1966,1966],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1990,1990],[50,0],linestyle='--',linewidth=1,color='gray')
    return tag_mtf

stats = []
mtf = plot_colored_timeseries('antivaccination (English 2019)', image_size=10)
s = tsia.markov.compute_mtf_statistics(mtf[0])
s.update({'Signal': 'signal-1'})
stats.append(s)

#2条曲线同一个ax
def plot_same_ax(tag1,tag2,image_size):
    # Loads the signal from disk:
    tag_df1 = pd.read_csv(os.path.join(DATA, f'{tag1}.csv'))
    tag_df1 = tag_df1.set_index('timestamp')
    tag_df2 = pd.read_csv(os.path.join(DATA, f'{tag2}.csv'))
    tag_df2 = tag_df2.set_index('timestamp')
    
    # Build the MTF for this signal——for曲线:
    X1 = tag_df1.values.reshape(1, -1)
    mtf1 = MarkovTransitionField(image_size=image_size, n_bins=n_bins, strategy=strategy)
    tag_mtf1 = mtf1.fit_transform(X1)
    
    X2 = tag_df2.values.reshape(1, -1)
    mtf2 = MarkovTransitionField(image_size=image_size, n_bins=n_bins, strategy=strategy)
    tag_mtf2 = mtf2.fit_transform(X2)
    
    #for左侧48
    mtf_l1=MarkovTransitionField(image_size=48, n_bins=n_bins, strategy=strategy)
    tag_mtf_l1 = mtf_l1.fit_transform(X1)
    mtf_l2=MarkovTransitionField(image_size=48, n_bins=n_bins, strategy=strategy)
    tag_mtf_l2 = mtf_l2.fit_transform(X2)
    
    #mtf单独
    fig_mtf1=plt.figure(figsize=(10, 10))
    ax_mtf1= fig_mtf1.add_subplot(111)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf_l1[0], ax=ax_mtf1, colormap=cmaps,reversed_cmap=False)
    plt.colorbar(mappable_image)
    fig_mtf2=plt.figure(figsize=(10, 10))
    ax_mtf2= fig_mtf2.add_subplot(111)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf_l2[0], ax=ax_mtf2, colormap=cmaps,reversed_cmap=False)
    plt.colorbar(mappable_image)
    
    # Initializing figure:
    fig = plt.figure(figsize=(30, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.8,1,4])
    
    # Plotting MTF:
    ax = fig.add_subplot(gs[0])
    ax.set_title('Markov transition field')
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf_l1[0], ax=ax, colormap=cmaps,reversed_cmap=False)
    
    ax = fig.add_subplot(gs[1])
    ax.set_title('Markov transition field')
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=tag_mtf_l2[0], ax=ax, colormap=cmaps,reversed_cmap=False)
    plt.colorbar(mappable_image)
    
    # Plotting signal:
    ax = fig.add_subplot(gs[2])
    ax.set_title(f'Signal timeseries for {tag1} and {tag2}')
    mtf_map1 = tsia.markov.get_mtf_map(tag_df1, tag_mtf1[0],step_size=0,colormap=cmaps,reversed_cmap=False)
    _ = tsia.plot.plot_colored_timeseries(tag_df1, mtf_map1, ax=ax)
    mtf_map2 = tsia.markov.get_mtf_map(tag_df2, tag_mtf2[0],step_size=0,colormap=cmaps,reversed_cmap=False)
    _ = tsia.plot.plot_colored_timeseries(tag_df2, mtf_map2, ax=ax)
    
    ax.text(1854,40,'anti-vaccination',size=14, color='black')
    ax.text(1870,10,'antivaccination',size=14, color='black')
    
    #单独曲线图
    fig2=plt.figure(figsize=(20, 8))
    ax2 = fig2.add_subplot(111)
    fig2.add_axes(ax2)
    #ax2.set_title(f'Signal timeseries for {tag1} and {tag2}')
    mtf_map1 = tsia.markov.get_mtf_map(tag_df1, tag_mtf1[0],step_size=0,colormap=cmaps,reversed_cmap=False)
    _ = tsia.plot.plot_colored_timeseries_jmq(tag_df1, mtf_map1, ax=ax2)
    mtf_map2 = tsia.markov.get_mtf_map(tag_df2, tag_mtf2[0],step_size=0,colormap=cmaps,reversed_cmap=False)
    _ = tsia.plot.plot_colored_timeseries_jmq(tag_df2, mtf_map2, ax=ax2)
    ax2.text(1852,73,'anti-vaccination',size=15, color='black')
    ax2.text(1881,6.5,'antivaccination',size=15, color='black')
    #plt.axhlines(0,1799,2019,linewidth=0.8,color='gray',alpha=0.6)
    plt.plot([1799,2019],[0,0],linestyle='-',linewidth=0.8,color='gray',alpha=0.6)
    
    '''
    plt.plot([1998,1998],[50,0],linestyle='--',linewidth=1,color='gray')#在(1998,y0)和(1998,0)两点之间画一条直线
    #plt.text(1997.5,30,'1998: A British doctor',fontsize=8,rotation=90,horizontalalignment="right")
    plt.plot([1853,1853],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1885,1885],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1890,1890],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1907,1907],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1921,1921],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1955,1955],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1966,1966],[50,0],linestyle='--',linewidth=1,color='gray')
    plt.plot([1990,1990],[50,0],linestyle='--',linewidth=1,color='gray')
    '''
plot_same_ax('antivaccination (English 2019)','anti-vaccination (English 2019)',8)



###网络图
tag_df = pd.read_csv(os.path.join(DATA, 'antivaccination (English 2019).csv'))
#tag_df['timestamp'] = pd.to_datetime(tag_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
tag_df = tag_df.set_index('timestamp')

image_size = 48
X = tag_df.values.reshape(1, -1)
mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins, strategy=strategy)
tag_mtf = mtf.fit_transform(X)

#建立网络图
G = tsia.network_graph.get_network_graph(tag_mtf[0])
_ = tsia.plot.plot_network_graph(G, title='Network graph')

#将分区和模块化编码到网络图表示中
encoding = tsia.network_graph.get_modularity_encoding(G)
stats = tsia.network_graph.compute_network_graph_statistics(G)

#绘制网络图
nb_partitions = stats['Partitions']
modularity = stats['Modularity']
title = rf'Partitions: $\bf{nb_partitions}$ - Modularity: $\bf{modularity:.3f}$'
_ = tsia.plot.plot_network_graph(G, title=title, encoding=encoding)

#将分区颜色映射回时间序列
ng_map = tsia.network_graph.get_network_graph_map(tag_df, encoding, reversed_cmap=True)
_ = tsia.plot.plot_colored_timeseries(tag_df, ng_map)

def plot_communities_timeseries(tag, image_size=48, colormap='jet'):
    # Loads the signal from disk:
    tag_df = pd.read_csv(os.path.join(DATA, f'{tag}.csv'))
    #tag_df['timestamp'] = pd.to_datetime(tag_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
    tag_df = tag_df.set_index('timestamp')
    
    X = tag_df.values.reshape(1, -1)
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins, strategy=strategy)
    tag_mtf = mtf.fit_transform(X)
    
    G = tsia.network_graph.get_network_graph(tag_mtf[0])
    statistics = tsia.network_graph.compute_network_graph_statistics(G)
    nb_partitions = statistics['Partitions']
    modularity = statistics['Modularity']
    encoding = tsia.network_graph.get_modularity_encoding(G, reversed_cmap=True)
    ng_map = tsia.network_graph.get_network_graph_map(tag_df, encoding, reversed_cmap=True)
    
    fig = plt.figure(figsize=(28, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,4])

    ax = fig.add_subplot(gs[0])
    title = rf'Partitions: $\bf{nb_partitions}$ - Modularity: $\bf{modularity:.3f}$'
    tsia.plot.plot_network_graph(G, ax=ax, title=title, reversed_cmap=True, encoding=encoding)
    
    ax = fig.add_subplot(gs[1])
    tsia.plot.plot_colored_timeseries(tag_df, ng_map, ax=ax)
    
    return statistics

stats = []
s = plot_communities_timeseries('antivaccination (English 2019)')
s.update({'Signal': 'signal-1'})
stats.append(s)


#循环画
signals = [f'signal-{i}' for i in range(1,7)]
stats = []
for signal in signals:
    s = plot_communities_timeseries(signal)
    s.update({'Signal': signal})
    stats.append(s)
    
stats = pd.DataFrame(stats)
stats.set_index('Signal')