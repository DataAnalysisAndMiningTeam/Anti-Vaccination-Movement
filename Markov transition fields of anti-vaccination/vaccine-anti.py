# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:01:52 2021

@author: mikij
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
os.chdir('C:/Users/mikij/Desktop/实验/代码/时间序列')

#vaccine
data=pd.read_excel("vaccine-anti.xlsx",sheet_name='vaccine')
x=data['time']
data=data.set_index('time')

xticks =np.arange(1759, 2029, 10)
y1=data['vaccine (English 2019)']
y2=data['vaccine inoculation (English 2019)']
y3=data['vaccination (English 2019)']
y4=data['variolation (English 2019)']
y5=data['vaccinate (English 2019)']

fig=plt.figure()
ax= fig.add_subplot(111)
ax=plt.gca()
#ax.spines['bottom'].set_linewidth('2.5')#设置边框线宽为2.0
#ax.spines['left'].set_linewidth('2.5')#设置边框线宽为2.0
#ax.arrow(2025.5, -5, 3, 0, head_width=1.5, head_length=3, ec='black', fc='black',color='black')#箭头
plt.plot(x, y1, color='#FF8080',linewidth=2.5)
plt.fill_between(x,y1*0.98,y1/0.98,facecolor='#FF8080',interpolate=True,alpha=0.6)
'''
plt.plot(x.loc[36:43],y1.loc[1795:1802],color='#FF8080',linewidth=6.0,alpha = 0.6)
plt.fill_between(x,y1*0.97,y1/0.97,where=((x>=1795) * (x<=1804)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y1*0.94,y1/0.94,where=((x>=1890) * (x<=1912)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y1*0.97,y1/0.97,where=((x>=1912) * (x<=1921)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y1*0.97,y1/0.97,where=((x>=1937) * (x<=1955)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y1*0.98,y1/0.98,where=((x>=1971) * (x<=2003)),facecolor='#FF8080',interpolate=True,alpha=0.6)
'''
#plt.plot(x.loc[36:45],y1.loc[1795:1804],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[131:162],y1.loc[1890:1921],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[178:196],y1.loc[1937:1955],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[212:244],y1.loc[1971:2003],color='#FF8080',linewidth=6.0,alpha = 0.6)

plt.plot(x, y2, color='#FFA280', linewidth=2.5)
plt.fill_between(x,y2*0.97,y2/0.97,facecolor='#FFA280',interpolate=True,alpha=0.6)

#plt.fill_between(x,y2*0.9,y2/0.9,where=((x>=1795) * (x<=1802)),facecolor='#FFA280',interpolate=True,alpha=0.6)
#plt.plot(x.loc[36:41],y2.loc[1795:1800],color='#FFA280',linewidth=5.0,alpha = 0.6)
#plt.plot(x.loc[36:43],y2.loc[1795:1802],color='#FFA280',linewidth=6.0,alpha = 0.6)
         
plt.plot(x, y3, color='#8080FF', linewidth=2.5)
plt.fill_between(x,y3*0.98,y3/0.98,facecolor='#8080FF',interpolate=True,alpha=0.6)
'''
plt.fill_between(x,y3*0.85,y3/0.85,where=((x>=1802) * (x<=1804)),facecolor='#8080FF',interpolate=True,alpha=0.6)
plt.fill_between(x,y3*0.95,y3/0.95,where=((x>=1804) * (x<=1808)),facecolor='#8080FF',interpolate=True,alpha=0.6)
plt.fill_between(x,y3*0.98,y3/0.98,where=((x>=1890) * (x<=1921)),facecolor='#8080FF',interpolate=True,alpha=0.6)
'''
#plt.plot(x.loc[43:49],y3.loc[1802:1808],color='#8080FF',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[131:162],y3.loc[1890:1921],color='#8080FF',linewidth=6.0,alpha = 0.6)
           
plt.plot(x, y4, color='#80C080',  linewidth=2.5)
plt.fill_between(x,y4*0.9,y4/0.9,facecolor='#80C080',interpolate=True,alpha=0.6)

#plt.fill_between(x,y4*0.85,y4/0.85,where=((x>=1799) * (x<=1815)),facecolor='#80C080',interpolate=True,alpha=0.6)
#plt.plot(x.loc[40:56],y4.loc[1799:1815],color='#80C080',linewidth=6.0,alpha = 0.6)
         
plt.plot(x, y5, color='#80E9E9',  linewidth=2.5)
plt.fill_between(x,y5*0.85,y5/0.85,facecolor='#80E9E9',interpolate=True,alpha=0.6)

#plt.fill_between(x,y5*0.93,y5/0.93,where=((x>=1890) * (x<=1921)),facecolor='#80E9E9',interpolate=True,alpha=0.6)
#plt.plot(x.loc[131:162],y5.loc[1890:1921],color='#80E9E9',linewidth=6.0,alpha = 0.6)
legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 16,  # 字号
    'weight': "normal",  # 是否加粗，不加粗
}
#去除重复的图例
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),frameon=False,loc='upper left',prop=legend_font)
#plt.plot([1799,2019],[0,0],linestyle='-',linewidth=0.8,color='gray',alpha=0.6)
'''
plt.plot([1795,1795],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1881,1881],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1890,1890],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1921,1921],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1937,1937],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1955,1955],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1971,1971],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([2004,2004],[100,0],linestyle='--',linewidth=1,color='gray')
'''
#plt.ylim(0, 100)
plt.xlabel(u'Year', fontsize=18)
plt.ylabel(u'Relative Frequency (Normalization)', fontsize=19)
plt.yticks(size=16, color='#000000')
plt.xticks(xticks, size=15, color='#000000')
#plt.savefig('vaccine.svg',dpi=1500)
plt.show()


#antivaccine
data=pd.read_excel("vaccine-anti.xlsx",sheet_name='anti')
x=data['time']
data=data.set_index('time')

xticks =np.arange(1799, 2029, 10)
y1=data['antivaccinist (English 2019)']
y2=data['anti-vaccination (English 2019)']
y3=data['antivaccination (English 2019)']
y4=data['antivaccination (British English 2019)']
y5=data['antivaccination (American English 2019)']
y6=data['antivaccination (French 2019)']
y7=data['antivacunación (Spanish 2019)']
y8=data['antivaccinazione (Italian 2019)']


fig=plt.figure()
ax= fig.add_subplot(111)
ax=plt.gca()
#ax.spines['bottom'].set_linewidth('2.5')#设置边框线宽为2.0
#ax.arrow(2024, -5, 3, 0, head_width=1.5, head_length=3, ec='black', fc='black',color='black')#箭头

plt.plot(x, y1, color='#7030A0', label=u'antivaccinist (English 2019)', linewidth=2.5)
plt.fill_between(x,y1*0.95,y1/0.95,facecolor='#7030A0',interpolate=True,alpha=0.6)
#xx=np.linspace(1802.1,1803.7,10)
#yy=np.linspace(0.25,19,10)
#plt.fill_between(x,y1*0.92,y1/0.92,where=((x>=1802) * (x<=1810)),facecolor='#7030A0',interpolate=True,alpha=0.6)
#plt.plot(xx,yy,color='#7030A0',linewidth=3,alpha = 0.6) 
#plt.plot(x.loc[7:16],y1.loc[1806:1815],color='#7030A0',linewidth=6.0,alpha = 0.6)

plt.plot(x, y2, color='#FF8080', label=u'anti-vaccination (English 2019)', linewidth=2.5)
plt.fill_between(x,y2*0.98,y2/0.98,facecolor='#FF8080',interpolate=True,alpha=0.6)
'''
xx=np.linspace(1865.05,1867.8,20)
yy=np.linspace(1.47,25.8,20)
plt.plot(xx,yy,color='#FF8080',linewidth=4,alpha = 0.6)
#plt.fill_between(x,y2*0.8,y2/0.8,where=((x>=1865) * (x<=1867)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.plot(x.loc[69:80],y2.loc[1868:1879],color='#FF8080',linewidth=6.0,alpha = 0.6)
xx=np.linspace(1878.91,1879.58,10)
yy=np.linspace(70.3,73,10)
plt.plot(xx,yy,color='#FF8080',linewidth=4,alpha = 0.6)
plt.fill_between(x,y2*0.97,y2/0.97,where=((x>=1880) * (x<=1883)),facecolor='#FF8080',interpolate=True,alpha=0.6)

plt.fill_between(x,y2*0.95,y2/0.95,where=((x>=1890) * (x<=1895)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y2*0.97,y2/0.97,where=((x>=1895) * (x<=1898)),facecolor='#FF8080',interpolate=True,alpha=0.6)

plt.fill_between(x,y2*0.93,y2/0.93,where=((x>=1924) * (x<=1955)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y2*0.93,y2/0.93,where=((x>=1998) * (x<=2007)),facecolor='#FF8080',interpolate=True,alpha=0.6)
plt.fill_between(x,y2*0.94,y2/0.94,where=((x>=2010) * (x<=2019)),facecolor='#FF8080',interpolate=True,alpha=0.6)     
#plt.plot(x.loc[69:84],y2.loc[1868:1883],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[91:99],y2.loc[1890:1898],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[122:156],y2.loc[1921:1955],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[199:208],y2.loc[1998:2007],color='#FF8080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[211:],y2.loc[2010:],color='#FF8080',linewidth=6.0,alpha = 0.6)
'''
plt.plot(x, y3, color='#8080FF', label=u'antivaccination (English 2019)', linewidth=2.5)
plt.fill_between(x,y3*0.97,y3/0.97,facecolor='#8080FF',interpolate=True,alpha=0.6)

#plt.fill_between(x,y3*0.96,y3/0.96,where=((x>=1890) * (x<=1907)),facecolor='#8080FF',interpolate=True,alpha=0.6)
#plt.plot(x.loc[91:108],y3.loc[1890:1907],color='#8080FF',linewidth=6.0,alpha = 0.6)
         
plt.plot(x, y4, color='#FFA280', label=u'antivaccination (British English 2019)', linewidth=2.5)
plt.fill_between(x,y4*0.98,y4/0.98,facecolor='#FFA280',interpolate=True,alpha=0.6)

#plt.fill_between(x,y4*0.94,y4/0.94,where=((x>=1865) * (x<=1883)),facecolor='#FFA280',interpolate=True,alpha=0.6)
#plt.plot(x.loc[66:84],y4.loc[1865:1883],color='#FFA280',linewidth=6.0,alpha = 0.6)

plt.plot(x, y5, color='#80C080', label=u'antivaccination (American English 2019)', linewidth=2.5)
plt.fill_between(x,y5*0.97,y5/0.97,facecolor='#80C080',interpolate=True,alpha=0.6)

#plt.fill_between(x,y5*0.96,y5/0.96,where=((x>=1890) * (x<=1907)),facecolor='#80C080',interpolate=True,alpha=0.6)
#plt.fill_between(x,y5*0.95,y5/0.95,where=((x>=2010) * (x<=2019)),facecolor='#80C080',interpolate=True,alpha=0.6)
#plt.plot(x.loc[91:108],y5.loc[1890:1907],color='#80C080',linewidth=6.0,alpha = 0.6)
#plt.plot(x.loc[211:],y5.loc[2010:],color='#80C080',linewidth=6.0,alpha = 0.6)
         
plt.plot(x, y6, color='#80E9E9', label=u'antivaccination (French 2019)', linewidth=2.5)
plt.fill_between(x,y6*0.92,y6/0.92,facecolor='#80E9E9',interpolate=True,alpha=0.6)

#plt.fill_between(x,y6*0.9,y6/0.9,where=((x>=1907) * (x<=1921)),facecolor='#80E9E9',interpolate=True,alpha=0.6)
#plt.plot(x.loc[108:122],y6.loc[1907:1921],color='#80E9E9',linewidth=6.0,alpha = 0.6)
         
plt.plot(x, y7, color='#DF80DF', label=u'antivacunación (Spanish 2019)', linewidth=2.5)
plt.fill_between(x,y7*0.92,y7/0.92,facecolor='#DF80DF',interpolate=True,alpha=0.6)

#plt.fill_between(x,y7*0.92,y7/0.92,where=((x>=1898) * (x<=1907)),facecolor='#DF80DF',interpolate=True,alpha=0.6)
#plt.plot(x.loc[99:108],y7.loc[1898:1907],color='#DF80DF',linewidth=6.0,alpha = 0.6)
         
plt.plot(x, y8, color='#808080', label=u'antivaccinazione (Italian 2019)', linewidth=2.5)
plt.fill_between(x,y8*0.92,y8/0.92,facecolor='#808080',interpolate=True,alpha=0.6)

#plt.fill_between(x,y8*0.92,y8/0.92,where=((x>=1890) * (x<=1907)),facecolor='#808080',interpolate=True,alpha=0.6)
#plt.plot(x.loc[91:108],y8.loc[1890:1907],color='#808080',linewidth=6.0,alpha = 0.6)

'''
plt.plot([1864,1864],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1883,1883],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1890,1890],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1898,1898],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1907,1907],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1921,1921],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1955,1955],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([1998,1998],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([2007,2007],[100,0],linestyle='--',linewidth=1,color='gray')
plt.plot([2010,2010],[100,0],linestyle='--',linewidth=1,color='gray')
'''
#plt.ylim(0, 100)

#去除重复的图例
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),frameon=False,loc='upper left',prop=legend_font)

plt.xlabel(u'Year', fontsize=18)
plt.ylabel(u'Relative Frequency (Normalization)', fontsize=19)
plt.yticks(size=16, color='#000000')
plt.xticks(xticks, size=15, color='#000000')
#plt.savefig('vaccine.svg',dpi=1500)
plt.show()

