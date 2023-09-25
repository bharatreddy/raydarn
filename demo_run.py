#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import datetime
import numpy
import multiprocessing

import rt
import rt_sct_utils
import plot_utils

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num, DateFormatter

import time

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cpu_count = multiprocessing.cpu_count()
# radar params
sTime = datetime.datetime(2018,2,6)#datetime.datetime(2010,10,13)
eTime = datetime.datetime(2018,2,7)#datetime.datetime(2010,10,14)
date_plot = datetime.datetime(2018,2,6,18)#datetime.datetime(2010,10,13,20)
radar = 'bks'
sel_beam = 11


# In[3]:


start_time = time.time()
rto = rt.RtRun(start_time=sTime, end_time=eTime,\
               radar_code=radar, beam=sel_beam,\
               out_dir='/tmp/', nprocs=4,\
              nhops=2,freq=12)
print("--- %s seconds ---" % (time.time() - start_time))


# In[4]:


fig = plt.figure(figsize=(16, 6))
rays_obj = plot_utils.RayPathPlot(rto, date_plot, sel_beam)
rays_obj.plot_edens()
rays_obj.add_cbar()
_ = rays_obj.cbax.set_ylabel(r"N$_{el}$ [$m^{-3}$]", fontsize=14)
rays_obj.ax1.set_title(date_plot.strftime("%Y-%m-%d %H:%M") + " ("+radar+")", fontsize=14)
rays_obj.ax1.set_ylabel(r"Alt. [km]", size=16)
rays_obj.ax1.set_xlabel(r"Ground range [km]", size=16)

# plot rays
rays_obj.plot_rays()
# plot rays

# Plot grnd & iono scat
rays_obj.plot_scatter()
# Plot grnd & iono scat


# In[5]:


plt.style.use("fivethirtyeight")

fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111)

sct_plt_obj = plot_utils.RTIPlot(rto, fig, ax)
sct_plt_obj.plot_scatter()

ax.get_xaxis().set_major_formatter(DateFormatter('%H'))

ax.set_ylabel("Range [Km]")
ax.set_xlabel("UT HOUR")
ax.set_title(date_plot.strftime("%Y-%m-%d"))

plt.xticks(rotation=45)
# cb = fig.colorbar(gnd_rti_plot)
# cb.set_label("Relative power [dB]")

ax.grid()
plt.show()


# In[ ]:




