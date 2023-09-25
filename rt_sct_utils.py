import numpy
import datetime
import pandas

class RT_SCT(object):
    """
    Class to convert ray-tracing results 
    into power in beam/range domain.
    An explanation for these calculations
    is given in Sebastien's dissertation.
    """
    
    def __init__(self, rt_obj, max_gates = 110):
        """
        Initialize the ray tracing params
        for a radar!
        rt_dict is obtained by running the RayTrace Class!
        """
        self.rt_obj = rt_obj
        self.rt_obj.read_scatter()
        self.max_gates = max_gates
        self.ranges = 180 + 45*numpy.arange(max_gates+1,dtype=int)
        self.pd_cut_ranges = 180 + 45*numpy.arange(max_gates+2,dtype=int)
        
    def get_gnd_sct_df(self):
        """
        Get estimated power from ground-scatter
        """
        lag_power_arr = []
        range_arr = []
        gates_arr = []
        date_arr = []
        azim_arr = []
        elev_arr = []
        df_arr = []
        for _dt in self.rt_obj.scatter.gsc.keys():
            for _az in self.rt_obj.scatter.gsc[_dt].keys():
                grp_ran = []
                weights = []
                sct_el_arr = []
                sct_alt_arr = []
                sct_hops_arr = []
                sct_nr_arr = []
                # features for geolocation
                sct_th_arr = []
                sct_lat_arr = []
                sct_lon_arr = []
                
                for _el in self.rt_obj.scatter.gsc[_dt][_az].keys():
                    gscat = self.rt_obj.scatter.gsc[_dt][_az][_el]
                    if gscat is None:
                        continue
                    
                    for _gi in range(gscat['hops'].shape[0]):
                    
                        sct_el_arr.append(_el)                        
                        sct_alt_arr.append((gscat['r'][_gi] -  6370.)/1000.) #  6370.0 is Rav
                        sct_hops_arr.append(gscat['hops'][_gi])
                        sct_nr_arr.append(gscat['nr'][_gi])
                        # geoloc
                        _wght = 1/(gscat["gran"][_gi]**3)
                        grp_ran.append(gscat["gran"][_gi]*1e-3)
                        weights.append(_wght)
                if len(grp_ran) == 0:
                    continue
                lag_power, bins = numpy.histogram(\
                            grp_ran,\
                            bins=self.ranges,\
                            weights=weights) 
                # now we need to get the "typical" hop in a bin
                # so we'll convert this into a DF
                _hop_el_df = pandas.DataFrame(list(zip(sct_el_arr, sct_hops_arr, sct_nr_arr, grp_ran, sct_alt_arr)),
                                   columns =['ele', 'hops', 'nr','grp_ran', 'alt'])
                _hop_el_df['binned'] = pandas.cut(\
                                        _hop_el_df['grp_ran'],\
                                        self.pd_cut_ranges,\
                                        labels=self.ranges
                                        )
                _hop_bin_df = _hop_el_df.groupby(['binned'], observed=False).mean().fillna(-1)
                _gates = list(numpy.arange(self.max_gates+1,dtype=int))
                _dates = [_dt for x in range(len(list(bins)))]
                _azims = [_az for x in range(len(list(bins)))]
                _elevs = [_hop_bin_df.loc[x]['ele'] for x in bins]
                _alts = [_hop_bin_df.loc[x]['alt'] for x in bins]
                _hops = [_hop_bin_df.loc[x]['hops'] for x in bins]
                _nr = [_hop_bin_df.loc[x]['nr'] for x in bins]
                
                _df = pandas.DataFrame(list(zip(lag_power, bins, _gates,\
                               _dates, _azims, _elevs, _alts,_hops, _nr)),
                            columns =['lag_power', 'range', \
                             'gate', 'date', 'azim',\
                            'median_elev', 'median_alt','hop', 'ref_ind'])
#                 print(_df.head())
                df_arr.append(_df)
        if len(df_arr) == 0:
            return None
        gnd_sct_df = pandas.concat(df_arr)
        # Now we need to normalize the power!
        # for comparing it with actual data (in some sense)
        # Refer to Sebastien's dissertation for 
        # more details!
        # IDL code: http://davit1.ece.vt.edu/doc/rt/rt_run-code.html
        if gnd_sct_df.shape[0] >= 1:
            gnd_sct_df['day'] = gnd_sct_df['date'].dt.date
            max_power_df = gnd_sct_df[['day', 'lag_power']].groupby(\
                                            ['day'], observed=False\
                                            ).max().reset_index()
            max_power_df.columns = ['day', 'max_lag_power']
            gnd_sct_df = pandas.merge(gnd_sct_df, max_power_df, on=['day'])
            gnd_sct_df['lag_power'] = 10.*numpy.log10(\
                                            gnd_sct_df['lag_power']/gnd_sct_df['max_lag_power']\
                                            )
        
        return gnd_sct_df
        
    def get_iono_sct_df(self):
        """
        Get estimated power from ionospheric-scatter
        """
        weights_arr = []
        grp_ran_arr = []
        range_arr = []
        gates_arr = []
        aspect_arr = []
        alt_arr = []
        date_arr = []
        azim_arr = []
        elev_arr = []
        df_arr = []

        for _dt in self.rt_obj.scatter.isc.keys():
            for _az in self.rt_obj.scatter.isc[_dt].keys():
                grp_ran = []
                weights = []
                hops = []
                altitude = []
                ref_ind = []
                sct_el_arr = []
                for _el in self.rt_obj.scatter.isc[_dt][_az].keys():
                    ionos = self.rt_obj.scatter.isc[_dt][_az][_el]
                    if ionos['nstp'] > 0.:
                        weights += list(ionos['w'])
                        grp_ran += list(ionos['gran']*1e-3)
                        hops += list(ionos['hops'])
                        altitude += list((ionos['r'] -  6370.)/1000.) #  6370.0 is Rav
                        ref_ind += list(ionos['nr'])
                        sct_el_arr.append(_el)
                if len(weights) == 0.:
                    continue

                
                _df = pandas.DataFrame(list(zip(weights, grp_ran,\
                               [_dt for x in range(len(grp_ran))],\
                               [_az for x in range(len(grp_ran))],\
                               [numpy.median(sct_el_arr) for x in range(len(grp_ran))],\
                               [hops[_i] for _i in range(len(grp_ran))],\
                               [altitude[_i] for _i in range(len(grp_ran))],\
                               [ref_ind[_i] for _i in range(len(grp_ran))])),
                            columns = ['weights', 'grp_ran','date', \
                             'azim', 'median_elev', 'hop','median_alt', 'ref_ind'])
                df_arr.append(_df)


        iono_sct_df = pandas.concat(df_arr)
        
        if iono_sct_df.shape[0] >= 1:
            iono_sct_df['day'] = iono_sct_df['date'].dt.date        
            # Now we need to normalize the weights by day
            # and calculate the lag_power, and then recreate 
            # the dataframe as we take a histogram 
            # (of group ranges and weights) to calc
            # lag_power and it has a different length
            # compared to the group ranges.
            max_wght_df = iono_sct_df[['day', 'weights']].groupby(\
                                    ['day'], observed=False\
                                    ).max().reset_index()
            max_wght_df.columns = ['day', 'max_weights']
            iono_sct_df = pandas.merge(iono_sct_df, max_wght_df, on=['day'])
            iono_sct_df["norm_weights"] = iono_sct_df["weights"]/iono_sct_df["max_weights"]

            lag_power_arr = []
            gates_arr = []
            range_arr = []
            date_arr = []
            azim_arr = []
            elev_arr = []
            hops_arr = []
            alt_arr = []
            ref_ind_arr = []
            
            fin_df_arr = []

            max_gates = 75
            grps = iono_sct_df.groupby(["date", "azim"], observed=False)
            ranges = 180 + 45*numpy.arange(75+1,dtype=int)
            addtnl_range = numpy.max(ranges) + 45
            for _d, _az in list(grps.indices.keys()):
                sel_df = iono_sct_df[ (iono_sct_df["date"] == _d) & (iono_sct_df["azim"] == _az) ]

                lag_power, bins = numpy.histogram( sel_df["grp_ran"], bins=self.ranges,\
                                                  weights=sel_df["norm_weights"])
                
                
#                 _hop_el_alt_df = pandas.DataFrame(list(zip(sct_el_arr, sct_hops_arr, grp_ran)),
#                                    columns =['ele', 'hops', 'grp_ran'])
                _hop_el_alt_df = sel_df[['median_elev', 'hop', 'median_alt', 'ref_ind','grp_ran']]
                # Suppress false positive warning
                pandas.options.mode.chained_assignment = None
                _hop_el_alt_df['binned'] = pandas.cut(\
                                        _hop_el_alt_df['grp_ran'],\
                                        self.pd_cut_ranges,\
                                        labels=self.ranges
                                        )
                _hop_bin_df = _hop_el_alt_df.groupby(['binned'], observed=False).mean().fillna(-1)
                
                # Note the numbers of bins is one less
                # than the lag_power, since we are getting
                # a histogram. So When converting to a DF we'll
                # need an additional element in the range!
                
                _gates = list(numpy.arange(1,self.max_gates+1,dtype=int))# 
                _dates = [_d for x in range(lag_power.shape[0])]
                _ranges = list(self.ranges[:-1])
                _azims = [_az for x in range(len(list(bins)))]
                
                _elevs = [_hop_bin_df.loc[x]['median_elev'] for x in bins]
                _hops = [_hop_bin_df.loc[x]['hop'] for x in bins]
                _alts = [_hop_bin_df.loc[x]['median_alt'] for x in bins]
                _refinds = [_hop_bin_df.loc[x]['ref_ind'] for x in bins]
                
                _df = pandas.DataFrame(list(zip(_dates, lag_power, _gates, _ranges,\
                               _azims, _elevs, _hops, _alts,_refinds)),
                            columns =['date', 'lag_power', 'gates', 'range',\
                                     'azim', 'median_elev', 'hop','median_alt', 'ref_ind'])
                fin_df_arr.append(_df)
                
            if len(fin_df_arr) == 0:
                return None    
            iono_df = pandas.concat(fin_df_arr)

            iono_df["day"] = iono_df["date"].dt.date
            max_power_df = iono_df[['day', 'lag_power']].groupby(\
                                            ['day'], observed=False\
                                            ).max().reset_index()
            max_power_df.columns = ['day', 'max_lag_power']
            iono_df = pandas.merge(iono_df, max_power_df, on=['day'])
            iono_df['lag_power'] = 10.*numpy.log10(\
                                    iono_df['lag_power']/iono_df['max_lag_power']\
                                    )

            return iono_df
        else:
            return iono_sct_df
