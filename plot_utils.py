from matplotlib.transforms import Affine2D, Transform
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import polar
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num, DateFormatter

from mpl_toolkits.axes_grid1 import SubplotDivider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.pyplot as plt 

import numpy
from pylab import gcf

class RayPathPlot(object):
    """
    Plot curved axis, background elecrton densities,
    the path of the rays, 
    mark locations with good aspect conditions
    
    Inspired/Copied from the code in DaViTPy
    """
        
    def __init__(self, rto, plot_time, plot_beam,
                 rect=111, fig=None, minground=0.,
                 maxground=2000, minalt=0,maxalt=500,
                 Re=6371., nyticks=5, nxticks=4):
        """
        Create curved axes in ground-range and altitude
        """
        ang = maxground / Re
        minang = minground / Re
        angran = ang - minang
        angle_ticks = [(0, "{:.0f}".format(minground))]
        while angle_ticks[-1][0] < angran:
            tang = angle_ticks[-1][0] + 1./nxticks*angran
            angle_ticks.append((tang, "{:.0f}".format((tang-minang)*Re)))
        grid_locator1 = FixedLocator([v for v, s in angle_ticks])
        tick_formatter1 = DictFormatter(dict(angle_ticks))

        altran = float(maxalt - minalt)
        alt_ticks = [(minalt+Re, "{:.0f}".format(minalt))]

        while alt_ticks[-1][0] < Re+maxalt:
            alt_ticks.append((altran / float(nyticks) + alt_ticks[-1][0], 
                              "{:.0f}".format(altran / float(nyticks) +
                                              alt_ticks[-1][0] - Re)))
        _ = alt_ticks.pop()
        grid_locator2 = FixedLocator([v for v, s in alt_ticks])
        tick_formatter2 = DictFormatter(dict(alt_ticks))

        tr_rotate = Affine2D().rotate(numpy.pi/2-ang/2)
        tr_shift = Affine2D().translate(0, Re)
        tr = polar.PolarTransform() + tr_rotate

        grid_helper = \
            floating_axes.GridHelperCurveLinear(tr, extremes=(0, angran, Re+minalt,
                                                              Re+maxalt),
                                                grid_locator1=grid_locator1,
                                                grid_locator2=grid_locator2,
                                                tick_formatter1=tick_formatter1,
                                                tick_formatter2=tick_formatter2,)

        if not fig: 
            fig = gcf()
        self.ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)

        self.ax1.invert_xaxis()

        self.ax1.minground = minground
        self.ax1.maxground = maxground
        self.ax1.minalt = minalt
        self.ax1.maxalt = maxalt
        self.ax1.Re = Re

        fig.add_subplot(self.ax1, transform=tr)

        # create a parasite axes whose transData in RA, cz
        self.aux_ax = self.ax1.get_aux_axes(tr)

        # for aux_ax to have a clip path as in ax
        self.aux_ax.patch = self.ax1.patch

        # but this has a side effect that the patch is drawn twice, and possibly
        # over some other artists. So, we decrease the zorder a bit to prevent this.
        self.ax1.patch.zorder=0.9
        self.rto = rto
        self.plot_time = plot_time
        self.plot_beam = plot_beam

    def add_cbar(self):
        """ 
        Append colorbar to axes
        """
        mappable = self.edens_im
        fig1 = self.ax1.get_figure()

        new_geom_type = self.ax1.get_geometry()[0]*100+self.ax1.get_geometry()[1]*10+self.ax1.get_geometry()[2]
        print("geom test-->", new_geom_type, self.ax1.get_geometry())
        divider = SubplotDivider(fig1, new_geom_type,aspect=True) #*self.ax1.get_geometry(), 111

        # axes for colorbar
        self.cbax = Axes(fig1, divider.get_position())

        h = [Size.AxesX(self.ax1), # main axes
             Size.Fixed(0.1), # padding
             Size.Fixed(0.2)] # colorbar
        v = [Size.AxesY(self.ax1)]

        _ = divider.set_horizontal(h)
        _ = divider.set_vertical(v)

        _ = self.ax1.set_axes_locator(divider.new_locator(nx=0, ny=0))
        _ = self.cbax.set_axes_locator(divider.new_locator(nx=2, ny=0))

        _ = fig1.add_axes(self.cbax)

        _ = self.cbax.axis["left"].toggle(all=False)
        _ = self.cbax.axis["top"].toggle(all=False)
        _ = self.cbax.axis["bottom"].toggle(all=False)
        _ = self.cbax.axis["right"].toggle(ticklabels=True, label=True)

        _ = plt.colorbar(mappable, cax=self.cbax)

    
    def plot_edens(self, cmap="cividis"):
        """
        Plot background electron densities
        """
        self.rto.read_edens()
        print(self.rto.ionos.edens.keys())
        edenstht = self.rto.ionos.edens[self.plot_time][self.plot_beam]['th']
        edensArr = self.rto.ionos.edens[self.plot_time][self.plot_beam]['nel']
        X, Y = numpy.meshgrid(edenstht, self.ax1.Re + numpy.linspace(60,560,250))
        self.edens_im = self.aux_ax.pcolormesh(X, Y,  edensArr, cmap=cmap)

    def plot_rays(self, add_ranges=True):
        """ 
        Plot rays
        """
        self.rto.read_rays()
        rdict = self.rto.rays.paths
        for _el in rdict[self.plot_time][self.plot_beam].keys():
            rays = rdict[self.plot_time][self.plot_beam][_el]
            self.aux_ax.plot(rays['th'], numpy.array(rays['r'])*1e-3, c='#9658B1', 
                                    zorder=8, linewidth=1.)
        if add_ranges:
            range_markers = [0] + list(numpy.arange(180, 5000, 225))
            x, y = [], []
            for _el in rdict[self.plot_time][self.plot_beam].keys():
                rays = rdict[self.plot_time][self.plot_beam][_el]
                grans = numpy.array(rays['gran'])*1e-3
                th = numpy.array(rays['th'])
                r = numpy.array(rays['r'])
                for rm in range_markers:
                    inds = (grans >= rm)
                    if inds.any():
                        x.append( th[inds][0] )
                        y.append( r[inds][0]*1e-3 )
                self.aux_ax.scatter(x, y, color="darkgray",s=0.25, zorder=9, alpha=0.4)
                
    def plot_scatter(self, ground=True,iono=True):
        """ 
        Plot gnd and iono scatter
        """
        self.rto.read_scatter()
        if ground:
            for _el in self.rto.scatter.gsc[self.plot_time][self.plot_beam].keys():
                gscat = self.rto.scatter.gsc[self.plot_time][self.plot_beam][_el]
                if gscat is not None:
                    self.aux_ax.scatter(gscat['th'], self.ax1.Re*numpy.ones(gscat['th'].shape), 
                                    color='k', zorder=10)
        
        if iono:
            for _el in self.rto.scatter.isc[self.plot_time][self.plot_beam].keys():
                ionos = self.rto.scatter.isc[self.plot_time][self.plot_beam][_el]

                if ionos['nstp'] <= 0:
                    continue
                t = ionos['th']
                r = ionos['r']*1e-3
                spts = numpy.array([t, r]).T.reshape(-1, 1, 2)
                h = ionos['h']*1e-3
                rel = numpy.radians( ionos['rel'] )
                r = numpy.sqrt( r**2 + h**2 + 2*r*h*numpy.sin( rel ) )
                t = t + numpy.arcsin( h/r * numpy.cos( rel ) )
                epts = numpy.array([t, r]).T.reshape(-1, 1, 2)
                segments = numpy.concatenate([spts, epts], axis=1)
                lcol = LineCollection( segments, zorder=10,linewidths=5. )

                _ = lcol.set_color('k')
                self.aux_ax.add_collection( lcol )
        

class RTIPlot(object):
    """
    Make RTI plots of GS and IS estimated using raytracing.
    """
    def __init__(self, rto, fig, ax, start_time=None,
                 end_time=None, cmap="viridis",
                ylabel="Range [Km]",xlabel="UT HOUR"):
        """
        Create curved axes in ground-range and altitude
        """
        import rt_sct_utils
        self.sct_obj = rt_sct_utils.RT_SCT(rto)
        self.start_time = None
        self.end_time = None
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        self.ax= ax
        self.cmap=cmap
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.fig = fig
    
    def plot_scatter(self,plot_param = "lag_power", vmin=-30.,
                vmax=0., ground=True, iono=True, colorbar=True):
        """
        Plot the scatter in an RTI-like format!!
        """
        if iono:
            iono_df = self.sct_obj.get_iono_sct_df()
            if iono_df.shape[0] > 0:
                iono_plot_df = iono_df[ ["date", "range",\
                            plot_param] ].pivot( "date", "range" )
                if self.start_time is not None:
                    iono_plot_df = iono_plot_df[iono_plot_df["date"] >= self.start_time]
                if self.end_time is not None:
                    iono_plot_df = iono_plot_df[iono_plot_df["date"] <= self.end_time]

                iono_time_vals = iono_plot_df.index.values
                iono_range_vals = iono_plot_df.columns.levels[1].values
                iono_time_cntr, iono_rng_cntr  = numpy.meshgrid( iono_time_vals, iono_range_vals )
                # Mask the nan values! pcolormesh can't handle them well!
                iono_pwr_vals = numpy.ma.masked_where(\
                                numpy.isnan(iono_plot_df[plot_param].values),\
                                iono_plot_df[plot_param].values)
                iono_rti_plot = self.ax.pcolormesh(iono_time_cntr.T , iono_rng_cntr.T,\
                                        iono_pwr_vals, cmap=self.cmap, vmin=vmin,vmax=vmax)
            else:
                iono_rti_plot = None
                print("No Ionospheric scatter identified!")
        
        if ground:
            gnd_df = self.sct_obj.get_gnd_sct_df()
            if gnd_df.shape[0] > 0:
                gnd_plot_df = gnd_df[ ["date", "range",\
                            plot_param] ].pivot( "date", "range" )
                if self.start_time is not None:
                    gnd_plot_df = gnd_plot_df[gnd_plot_df["date"] >= self.start_time]
                if self.end_time is not None:
                    gnd_plot_df = gnd_plot_df[gnd_plot_df["date"] <= self.end_time]

                gnd_time_vals = gnd_plot_df.index.values
                gnd_range_vals = gnd_plot_df.columns.levels[1].values
                gnd_time_cntr, gnd_rng_cntr  = numpy.meshgrid( gnd_time_vals, gnd_range_vals )
                # Mask the nan values! pcolormesh can't handle them well!
                gnd_pwr_vals = numpy.ma.masked_where(\
                                numpy.isnan(gnd_plot_df[plot_param].values),\
                                gnd_plot_df[plot_param].values)
                gnd_rti_plot = self.ax.pcolormesh(gnd_time_cntr.T , gnd_rng_cntr.T, gnd_pwr_vals,\
                                        cmap=self.cmap, vmin=vmin,vmax=vmax)
            else:
                gnd_rti_plot = None
                print("No ground scatter identified!")
        # set the axes parameters
#         self.ax.get_xaxis().set_major_formatter(DateFormatter('%H'))
#         self.ax.set_ylabel(self.ylabel)
#         self.ax.set_xlabel(self.xlabel)
#         self.ax.set_title(date_plot.strftime("%Y-%m-%d"))
#         self.ax.tick_params(axis='x', rotation=45)
        if colorbar:
            if gnd_rti_plot:
                self.cb = self.fig.colorbar(gnd_rti_plot)
                self.cb.set_label("Relative power [dB]")
            else:
                if iono_rti_plot:
                    self.cb = self.fig.colorbar(iono_rti_plot)
                    self.cb.set_label("Relative power [dB]")