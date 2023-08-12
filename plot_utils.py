from matplotlib.transforms import Affine2D, Transform
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import polar
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter

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
        
    def __init__(self, rect=111, fig=None, minground=0., maxground=2000, minalt=0,
                        maxalt=500, Re=6371., nyticks=5, nxticks=4):
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

#         return ax1, aux_ax

    def add_cbar(self, mappable):
        """ 
        Append colorbar to axes
        """
        fig1 = self.ax1.get_figure()
        divider = SubplotDivider(fig1, *self.ax1.get_geometry(), aspect=True)

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

#         return self.cbax

    def plot_rays(self, rdict, plot_time, plot_beam, add_ranges=True):
        """ 
        Plot rays
        """
        for _el in rdict[plot_time][plot_beam].keys():
            rays = rdict[plot_time][plot_beam][_el]
            self.aux_ax.plot(rays['th'], numpy.array(rays['r'])*1e-3, c='#9658B1', 
                                    zorder=8, linewidth=1.)
        if add_ranges:
            range_markers = [0] + list(numpy.arange(180, 5000, 225))
            x, y = [], []
            for _el in rdict[plot_time][plot_beam].keys():
                rays = rdict[plot_time][plot_beam][_el]
                grans = numpy.array(rays['gran'])*1e-3
                th = numpy.array(rays['th'])
                r = numpy.array(rays['r'])
                for rm in range_markers:
                    inds = (grans >= rm)
                    if inds.any():
                        x.append( th[inds][0] )
                        y.append( r[inds][0]*1e-3 )
                self.aux_ax.scatter(x, y, color="darkgray",s=0.25, zorder=9, alpha=0.4)
                
    def plot_scatter(self, rto, plot_time, plot_beam, ground=True,iono=True):
        """ 
        Plot gnd and iono scatter
        """
        
        if ground:
            for _el in rto.scatter.gsc[date_plot][plot_beam].keys():
                gscat = rto.scatter.gsc[date_plot][plot_beam][_el]
                if gscat is not None:
                    self.aux_ax.scatter(gscat['th'], ax.Re*numpy.ones(gscat['th'].shape), 
                                    color='k', zorder=10)
        
        if iono:
            for _el in rto.scatter.isc[date_plot][plot_beam].keys():
                ionos = rto.scatter.isc[date_plot][plot_beam][_el]

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
        