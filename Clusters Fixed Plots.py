#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


POF11 = pd.read_csv("POF_clusters_fixed.csv")
POF11 = POF11.set_index(["Unnamed: 0"])

POF11

cols = POF11.columns
rows = POF11.index

AVG_POF = []
WOR_POF = []

for i in rows:
    SUM_POF = []
    for j in cols:
        SUM_POF.append(POF11[j][i])
    WOR_POF.append(float(max(SUM_POF)))
    AVG_POF.append(np.mean(SUM_POF))
    
f=plt.figure()
f.set_figwidth(5)
f.set_figheight(4)
plt.rc('font', size=15) 

plt.xlabel("Number of agents")
plt.ylabel("Price of Fairness")
plt.plot(rows,AVG_POF,label='Average')
plt.plot(rows,WOR_POF,"ro",label='Worst POF')
plt.legend()
plt.show()


# In[27]:


WOR_POF = []

for i in rows:
    SUM_POF = []
    for j in cols:
        cell = POF11[j][i]
        SUM_POF.append(cell)
    WOR_POF.append(max(SUM_POF))


f = plt.figure()
f.set_figwidth(10)
f.set_figheight(5)
    
plt.xlabel("Number of agents")
plt.ylabel("Price of Fairness")
plt.scatter(rows,WOR_POF,label='Worst POF')
plt.legend()
plt.show()


# In[3]:


times11 = pd.read_csv("runningtime_clusters_fixed.csv")
times11 = times11.set_index(["Unnamed: 0"])

# times11.head()

br = pd.read_csv("runningtime_clusters_fixed_br.csv")
br = br.set_index(["Unnamed: 0"])

# br.head()

gnn = pd.read_csv("runningtime_clusters_fixed_gnn.csv")
gnn = gnn.set_index(["Unnamed: 0"])

# gnn.head()

cols = times11.columns
rows = times11.index


SUM_times = []
AVG_times = []

SUM_times1 = []
AVG_times1 = []

SUM_times2 = []
AVG_times2 = []

for i in rows:
    SUM_times = []
    SUM_times1 = []
    SUM_times2 = []
    for j in cols:
        cell = times11[j][i]
        SUM_times.append(float(cell))
        SUM_times1.append(br[j][i])
        SUM_times2.append(gnn[j][i])
    AVG_times.append(np.mean(SUM_times))
    AVG_times1.append(np.mean(SUM_times1))
    AVG_times2.append(np.mean(SUM_times2))
    

f = plt.figure()
f.set_figwidth(5)
f.set_figheight(4)
plt.rc('font', size=15) 

plt.xlabel("Number of agents")
plt.ylabel("Running time")
plt.plot(rows,AVG_times,label='Our Algorithm')
plt.plot(rows,AVG_times1,marker='o',color="red",label='Brutforce')
plt.plot(rows,AVG_times2,color="green",linestyle='dashed',label='GNN')
plt.legend()
plt.show()

