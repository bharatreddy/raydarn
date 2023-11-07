"""
Python wrapper for the Fortran raytracing code!
This version is adapted from a previous one in DaViTPy

"""
import numpy as np
import pandas as pd

import pydarn


#########################################################################
# Main object
#########################################################################
class RtRun(object):
    """This class runs the raytracing code and processes the output

    Parameters
    ----------
    start_time : Optional[datetime.datetime]
        start time UT
    end_time : Optional[datetime.datetime]
        end time UT (if not provided run for a single time start_time)
    radar_code : Optional[str]
        radar 3-letter code
    dTime : Optional[float]
        time step in Hours
    freq : Optional[float]
        operating frequency [MHz]
    beam : Optional[int]
        beam number (if None run all beams)
    nhops : Optional[int]
        number of hops
    elev : Optional[tuple]
        (start elevation, end elevation, step elevation) [degrees]
    azim : Optional[tuple]
        (start azimuth, end azimuth, step azimuth) [degrees East] (overrides beam specification)
    hmf2 : Optional[float]
        F2 peak alitude [km] (default: use IRI)
    nmf2 : Optional[float]
        F2 peak electron density [log10(m^-3)] (default: use IRI)
    fext : Optional[str]
        output file id, max 10 character long (mostly used for multiple users environments, like a website)
    loadFrom : Optional[str]
        file name where a pickled instance of RtRun was saved (supersedes all other args)
    nprocs : Optional[int]
        number of processes to use with MPI

    Attributes
    ----------
    radar :

    site :

    azim :

    beam :

    elev :

    time : list

    dTime : float

    freq : float

    nhops : int

    hmf2 : float

    nmf2 : float

    out_dir :

    fExt : 

    davitpy_path : str

    edens_file :

    Methods
    -------
    RtRun.read_rays
    RtRun.read_edens
    RtRun.read_scatter
    RtRun.save
    RtRun.load

    Example
    -------
        # Run a 2-hour ray trace from Blackstone on a random day
        start_time = dt.datetime(2012, 11, 18, 5)
        end_time = start_time + dt.timedelta(hours=2)
        radar = 'bks'
        # Save the results to your /tmp directory
        rto = raydarn.RtRun(start_time, end_time, radar_code=radar, out_dir='/tmp')

    """
    def __init__(self, 
        boresight = -40.0, beams = 24,
        gates = 110, geographic_lat = 37.1,
        geographic_lon = -77.95,
        beam_seperation = 3.24,
        start_time=None, end_time=None, 
        radar_code=None, 
        dTime=1., 
        freq=11, beam=None, nhops=1, 
        elev=(5, 60, .1), azim=None, 
        hmf2=None, nmf2=None, 
        out_dir=None, 
        fext=None, 
        loadFrom=None, 
        nprocs=4, use_alt_beams=False):
        
        import datetime as dt
        from os import path

        if radar_code:
            self.radar = radar_code
            # Use pydarn for getting hdw data
            site_data = pydarn.read_hdw_file(radar_code)
            self.boresite = site_data.boresight.physical#boresight
            self.nbeams = site_data.beams#beams
            self.ngates = site_data.beams
            self.geolat = site_data.geographic.lat
            self.geolon = site_data.geographic.lon
            self.beam_sep = site_data.beam_separation
            self.offset = self.nbeams/2. - 0.5
            
            self.site = Site(radar_code, self.boresite,\
                             self.nbeams, self.ngates,\
                             self.geolat, self.geolon,\
                             self.beam_sep)
            
            if (beam is not None) and not azim: 
                az = round(\
                            self.boresite + (\
                                beam - self.offset\
                                )*self.beam_sep,2\
                        )
#                 az = round(self.boresite + (beam - self.offset)*self.beam_sep,2)
                azim = (az, az, 1)
            else:
                az1 = round(\
                            self.boresite + (\
                                0 - self.offset\
                                )*self.beam_sep,2\
                        )
                az2 = round(\
                            self.boresite + (\
                                (self.nbeams-1) - self.offset\
                                )*self.beam_sep,2\
                        )
                if use_alt_beams:
                    azim = (az1, az2, self.site.beam_sep*2.)
                else:
                    azim = (az1, az2, self.site.beam_sep)
            # Set azimuth
            self.azim = azim
            self.beam = beam
            # Set elevation
            self.elev = elev
            # Set time interval
            if not start_time: 
                print('No start time. Using now.')
                start_time = dt.datetime.utcnow()
            if not end_time:
                end_time = start_time + dt.timedelta(minutes=1)
            if end_time > start_time + dt.timedelta(days=1):
                print('The time interval requested if too large. Reducing to 1 day.')
                end_time = start_time + dt.timedelta(days=1)
            self.time = [start_time, end_time]
            self.dTime = dTime
            # Set frequency
            self.freq = freq
            # Set number of hops
            self.nhops = nhops
            # Set ionosphere
            self.hmf2 = hmf2 if hmf2 else 0
            self.nmf2 = nmf2 if nmf2 else 0
            # Set output directory and file extension
            if not out_dir:
                print("out_dir not set by user. Setting it to '/tmp'")
                out_dir = "/tmp"
            self.out_dir = path.join( out_dir, '' )
            if fext is None:
                self.fExt = start_time.strftime("%Y%j") +\
                        "_" + str(freq)
            else:
                self.fExt = fext

            # Write input file
            input_file = self._gen_input()
            # Run the ray tracing
            success = self._execute(nprocs, input_file)
        
    def _gen_input(self):
        """Generate input file

        Returns
        -------
        fname

        """
        from os import path

        fname = path.join(self.out_dir, 'rtrun.{}.inp'.format(self.fExt))
        with open(fname, 'w') as f:
            f.write( "{:8.2f}  Transmitter latitude (degrees N)\n".format( self.geolat  ) )
            f.write( "{:8.2f}  Transmitter Longitude (degrees E)\n".format( self.geolon ) )
            f.write( "{:8.2f}  Azimuth (degrees E) (begin)\n".format( self.azim[0] ) )
            f.write( "{:8.2f}  Azimuth (degrees E) (end)\n".format( self.azim[1] ) )
            f.write( "{:8.2f}  Azimuth (degrees E) (step)\n".format( self.azim[2] ) )
            f.write( "{:8.2f}  Elevation angle (begin)\n".format( self.elev[0] ) )
            f.write( "{:8.2f}  Elevation angle (end)\n".format( self.elev[1] ) )
            f.write( "{:8.2f}  Elevation angle (step)\n".format( self.elev[2] ) )
            f.write( "{:8.2f}  Frequency (Mhz)\n".format( self.freq ) )
            f.write( "{:8d}  nubmer of hops (minimum 1)\n".format( self.nhops) )
            f.write( "{:8d}  Year (yyyy)\n".format( self.time[0].year ) )
            f.write( "{:8d}  Month and day (mmdd)\n".format( self.time[0].month*100 + self.time[0].day ) )
            tt = self.time[0].hour + self.time[0].minute/60.
            tt += 25.
            f.write( "{:8.2f}  hour (add 25 for UT) (begin)\n".format( tt ) )
            tt = self.time[1].hour + self.time[1].minute/60.
            tt += (self.time[1].day - self.time[0].day) * 24.
            tt += 25.
            f.write( "{:8.2f}  hour (add 25 for UT) (end)\n".format( tt ) )
            f.write( "{:8.2f}  hour (step)\n".format( self.dTime ) )
            f.write( "{:8.2f}  hmf2 (km, if 0 then ignored)\n".format( self.hmf2 ) )
            f.write( "{:8.2f}  nmf2 (log10, if 0 then ignored)\n".format( self.nmf2 ) )

            f.write( "None"+"\n" ) # DaViTpy install path

            if hasattr(self,'edens_file'):  # Path to user-defined electron profile
                f.write( self.edens_file )

        return fname
    
    def clean_up(self):
        """
        Delete the files generated by the fortran code!
        """
        import glob
        import os
        file_list = glob.glob(self.out_dir + "/*dat")
        for _fp in file_list:
            os.remove(_fp)
        file_list = glob.glob(self.out_dir + "/*inp*")
        for _fp in file_list:
            os.remove(_fp)
        
        

    def _execute(self, nprocs, input_fileName):
        """Execute raytracing command

        Parameters
        ----------
        nprocs : int
            number of processes to use with MPI
        input_filename : str

        """
        import subprocess as subp
        from os import path

        command = ['mpiexec', '-n', '{}'.format(nprocs), 
            path.join(path.abspath( __file__.split('rt.py')[0] ), 'rtFort'), 
            input_fileName, 
            self.out_dir, 
            self.fExt]
#         print(command)
        #print ' '.join(command)
        process = subp.Popen(command, shell=False, stdout=subp.PIPE, stderr=subp.STDOUT)
        output = process.communicate()[0]
        exitCode = process.returncode

        if (exitCode != 0):
            print('In:: {}'.format( command ))
            print('Exit code:: {}'.format( exitCode ))
            print('Returned:: \n' + output.decode('utf-8'))

#         print('In:: {}'.format( command ))
#         print('Exit code:: {}'.format( exitCode ))
#         print('Returned:: \n' + output.decode('utf-8'))

        
        if (exitCode != 0):
            raise Exception('Fortran execution error.')
        else:
            return True


    def read_rays(self, save_to_ascii=None):
        """Read rays.dat fortran output into dictionnary

        Parameters
        ----------
        save_to_ascii : Optional[str]
            output content to text file

        Returns
        -------
        Add a new member to class rt.RtRun *rays*, of type class rt.rays

        """
        import subprocess as subp
        from os import path

        # File name and path
        fName = path.join(self.out_dir, 'rays.{}.dat'.format(self.fExt))
        if hasattr(self, 'rays') and not path.exists(fName):
            print('The file is gone, and it seems you may already have read it into memory...?')
            return

        # Initialize rays output
        self.rays = Rays(fName, 
            site=self.site, radar=self.radar,
            save_to_ascii=save_to_ascii)
        # Remove Input file
#        subp.call(['rm',fName])


    def read_edens(self):
        """Read edens.dat fortran output

        Parameters
        ----------
        None

        Returns
        -------
        Add a new member to class rt.RtRun *rays*, of type class rt.rays

        """
        import subprocess as subp
        from os import path

        # File name and path
        fName = path.join(self.out_dir, 'edens.{}.dat'.format(self.fExt))
        if hasattr(self, 'ionos') and not path.exists(fName):
            print('The file is gone, and it seems you may already have read it into memory...?')
            return

        # Initialize rays output
        self.ionos = Edens(fName, 
            site=self.site, radar=self.radar)
        # Remove Input file
#        subp.call(['rm',fName])


    def read_scatter(self):
        """Read iscat.dat and gscat.dat fortran output

        Parameters
        ----------
        None

        Returns
        -------
        Add a new member to class rt.RtRun *rays*, of type class rt.rays

        """
        import subprocess as subp
        from os import path

        # File name and path
        isName = path.join(self.out_dir, 'iscat.{}.dat'.format(self.fExt))
        gsName = path.join(self.out_dir, 'gscat.{}.dat'.format(self.fExt))
        if hasattr(self, 'scatter') \
            and (not path.exists(isName) \
            or not path.exists(gsName)):
            print('The files are gone!!!')
            return

        # Initialize rays output
        self.scatter = Scatter(gsName, isName, 
            site=self.site, radar=self.radar)
        # Remove Input file
        # subp.call(['rm',isName])
        # subp.call(['rm',gsName])



#########################################################################
# Electron densities
#########################################################################
class Edens(object):
    """Store and process electron density profiles after ray tracing

    Parameters
    ----------
    read_from : str
        edens.dat file to read the rays from

    Attributes
    ----------
    read_from : str

    edens : dict

    name : str

    Methods
    -------
    Edens.read_edens
    Edens.plot

    """
    def __init__(self, read_from, 
        site=None, radar=None):
        self.read_from = read_from
        self.edens = {}

        self.name = ''
        if radar:
            self.name = radar#.code[0].upper()

        # Read rays
        self.read_edens(site=site)


    def read_edens(self, site=None):
        """Read edens.dat fortran output

        Parameters


        Returns
        -------
        Populate member edens class rt.Edens

        """
        from struct import unpack
        import datetime as dt
        from numpy import array

        # Read binary file
        with open(self.read_from, 'rb') as f:
#             print(self.read_from + ' header: ')
            self.header = _read_header(f)
            self.edens = {}
            while True:
                bytes = f.read(2*4)
                # Check for eof
                if not bytes: break
                # read hour and azimuth
                hour, azim = unpack('2f', bytes)
                # format time index
                hour = hour - 25.
                mm = int(self.header['mmdd']/100)
                dd = self.header['mmdd'] - mm*100
                rtime = dt.datetime(self.header['year'], mm, dd) + dt.timedelta(hours=hour)
                # format azimuth index (beam)
                raz = site.azim_to_beam(azim) if site else round(raz, 2)
                # Initialize dicts
                if rtime not in self.edens.keys(): self.edens[rtime] = {}
                self.edens[rtime][raz] = {}
                # Read edens dict
                # self.edens[rtime][raz]['pos'] = array( unpack('{}f'.format(250*2), 
                #     f.read(250*2*4)) )
                self.edens[rtime][raz]['th'] = array( unpack('{}f'.format(250), 
                    f.read(250*4)) )
                self.edens[rtime][raz]['nel'] = array( unpack('{}f'.format(250*250), 
                    f.read(250*250*4)) ).reshape((250,250), order='F')
                self.edens[rtime][raz]['dip'] = array( unpack('{}f'.format(250*2), 
                    f.read(250*2*4)) ).reshape((250,2), order='F')


#########################################################################
# Scatter
#########################################################################
class Scatter(object):
    """Stores and process ground and ionospheric scatter

    Parameters
    ----------
    read_is_from : Optional[str]
        iscat.dat file to read the ionospheric scatter from
    read_gs_from : Optional[str]
        gscat.dat file to read the ground scatter from

    Attributes
    ----------
    read_is_from : str
        iscat.dat file to read the ionospheric scatter from
    read_gs_from : str
        gscat.dat file to read the ground scatter from
    gsc :

    isc :

    Methods
    -------
    Scatter.read_gs
    Scatter.read_is
    Scatter.plot

    """
    def __init__(self, read_gs_from=None, read_is_from=None, 
        site=None, radar=None):
        self.read_is_from = read_is_from
        self.read_gs_from = read_gs_from

        # Read ground scatter
        if self.read_gs_from:
            self.gsc = {}
            self.read_gs(site=site)

        # Read ionospheric scatter
        if self.read_is_from:
            self.isc = {}
            self.read_is(site=site)


    def read_gs(self, site=None):
        """Read gscat.dat fortran output

        Parameters
        ----------

        Returns
        -------
        Populate member isc class rt.Scatter

        """
        from struct import unpack
        import datetime as dt
        import numpy as np

        with open(self.read_gs_from, 'rb') as f:
            # read header
#             print(self.read_gs_from + ' header: ')
            self.header = _read_header(f)

            scatter_list = []

            # Then read ray data, one ray at a time
            while True:
                bytes = f.read(4*4)
                # Check for eof
                if not bytes: break
                # read number of ray steps, time, azimuth and elevation
                rhr, raz, rel, ihop = unpack('4f', bytes)
#                 if ihop > 0:
#                     print(rhr, raz, rel, ihop)
                # Read reminder of the record
                rr, tht, gran, lat, lon, nr  = unpack('6f', f.read(6*4))
                # Convert azimuth to beam number
                raz = site.azim_to_beam(raz) if site else np.round(raz, 2)
                # Adjust rel to 2 decimal
                rel = np.around(rel, 2)
                # convert time to python datetime
                rhr = rhr - 25.
                mm = int(self.header['mmdd']/100)
                dd = int(self.header['mmdd'] - mm*100)
                rtime = dt.datetime(self.header['year'], mm, dd) + dt.timedelta(hours=rhr)
                # Create new entries in rays dict
                if rtime not in self.gsc.keys(): self.gsc[rtime] = {}
                if raz not in self.gsc[rtime].keys(): self.gsc[rtime][raz] = {}
                if rel not in self.gsc[rtime][raz].keys(): 
                    self.gsc[rtime][raz][rel] = {
                        'r': np.empty(0),
                        'th': np.empty(0),
                        'gran': np.empty(0),
                        'lat': np.empty(0),
                        'lon': np.empty(0),
                        'hops': np.empty(0),
                        'nr': np.empty(0)}
                self.gsc[rtime][raz][rel]['r'] = np.append( self.gsc[rtime][raz][rel]['r'], rr )
                self.gsc[rtime][raz][rel]['th'] = np.append( self.gsc[rtime][raz][rel]['th'], tht )
                self.gsc[rtime][raz][rel]['gran'] = np.append( self.gsc[rtime][raz][rel]['gran'], gran )
                self.gsc[rtime][raz][rel]['lat'] = np.append( self.gsc[rtime][raz][rel]['lat'], lat )
                self.gsc[rtime][raz][rel]['lon'] = np.append( self.gsc[rtime][raz][rel]['lon'], lon )
                self.gsc[rtime][raz][rel]['hops'] = np.append( self.gsc[rtime][raz][rel]['hops'], ihop )
                self.gsc[rtime][raz][rel]['nr'] = np.append( self.gsc[rtime][raz][rel]['nr'], nr )
                # Same thing, but let's prepare for a Pandas DataFrame...
                tmp = {}
                tmp['type']     = 'gs'
                tmp['rtime']    = rtime
                tmp['raz']      = raz
                tmp['rel']      = rel
                tmp['r']        = rr
                tmp['th']       = tht
                tmp['gran']     = gran
                tmp['lat']      = lat
                tmp['lon']      = lon
                tmp['hops']      = ihop
                scatter_list.append(tmp)

#         self.gsc_df = pd.DataFrame(scatter_list)

    def read_is(self, site=None):
        """Read iscat.dat fortran output

        Parameters
        ----------

        Returns
        -------
        Populate member isc class rt.Scatter

        """
        from struct import unpack
        import datetime as dt
        from numpy import around, array

        with open(self.read_is_from, 'rb') as f:
            # read header
#             print(self.read_is_from+' header: ')
            self.header = _read_header(f)
            # Then read ray data, one ray at a time
            while True:
                bytes = f.read(4*4)
                # Check for eof
                if not bytes: break
                # read number of ray steps, time, azimuth and elevation
                nstp, rhr, raz, rel = unpack('4f', bytes)
                nstp = int(nstp)
                # Convert azimuth to beam number
                raz = site.azim_to_beam(raz) if site else around(raz, 2)
                # Adjust rel to 2 decimal
                rel = around(rel, 2)
                # convert time to python datetime
                rhr = rhr - 25.
                mm = int(self.header['mmdd']/100)
                dd = self.header['mmdd'] - mm*100
                rtime = dt.datetime(self.header['year'], mm, dd) + dt.timedelta(hours=rhr)
                # Create new entries in rays dict
                if rtime not in self.isc.keys(): self.isc[rtime] = {}
                if raz not in self.isc[rtime].keys(): self.isc[rtime][raz] = {}
                self.isc[rtime][raz][rel] = {}
                # Read to paths dict
                self.isc[rtime][raz][rel]['nstp'] = nstp
                self.isc[rtime][raz][rel]['r'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['th'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['gran'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['rel'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['w'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['nr'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['lat'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['lon'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['h'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )
                self.isc[rtime][raz][rel]['hops'] = array( unpack('{}f'.format(nstp), 
                    f.read(nstp*4)) )

    def gate_scatter(self,beam,fov):
        """

        Parameters
        ----------
        beam :

        fov :

        Returns
        -------
        lag_power

        """
        #Add a 0 at the beginning to get the range gate numbering right.
#        beam_inx    = np.where(beam == fov.beams)[0][0]
#        ranges      = [0]+fov.slantRFull[beam_inx,:].tolist() 

        # Some useful parameters
        ngates          = fov.gates.size
        range_gate      = 180 + 45*np.arange(ngates+1,dtype=np.int)
        Re              = 6370.
        P               = np.array(range_gate,dtype=np.float)
        minpower        = 4. 

        if self.gsc_df.size > 0:
            weights         = 1/(self.gsc_df.gran**3)
            lag_power, bins = np.histogram(self.gsc_df.gran/1000.,bins=range_gate,weights=weights)
        else:
            lag_power   = np.zeros_like(fov.gates,dtype=np.float)
        
        self.pwr        = lag_power
        self.gates      = fov.gates

        return lag_power 

#########################################################################
# Rays
#########################################################################
class Rays(object):
    """Store and process individual rays after ray tracing

    Parameters
    ----------
    read_from : str
        rays.dat file to read the rays from
    
    save_to_ascii : Optional[str]
        file name where to output ray positions

    Attributes
    ----------
    read_from : str
        rays.dat file to read the rays from
    paths :

    name : str


    Methods
    -------
    Rays.read_rays
    Rays.write_to_ascii
    Rays.plot

    """
    def __init__(self, read_from, 
        site=None, radar=None, 
        save_to_ascii=None):
        self.read_from = read_from
        self.paths = {}

        self.name = ''
        if radar:
            self.name = radar#.code[0].upper()

        # Read rays
        self.read_rays(site=site)

        # If required, save to ascii
        if save_to_ascii:
            self.write_to_ascii(save_to_ascii)


    def read_rays(self, site=None):
        """Read rays.dat fortran output

        Parameters
        ----------
       

        Returns
        -------
        Populate member paths class rt.Rays

        """
        from struct import unpack
        import datetime as dt
        from numpy import round, array

        # Read binary file
        with open(self.read_from, 'rb') as f:
            # read header
#             print(self.read_from+' header: ')
            self.header = _read_header(f)
            # Then read ray data, one ray at a time
            while True:
                bytes = f.read(4*4)
                # Check for eof
                if not bytes: break
                # read number of ray steps, time, azimuth and elevation
                nrstep, rhr, raz, rel = unpack('4f', bytes)
                nrstep = int(nrstep)
                # Convert azimuth to beam number
                raz = site.azim_to_beam(raz) if site else round(raz, 2)
                # convert time to python datetime
                rhr = rhr - 25.
                mm = int(self.header['mmdd']/100)
                dd = int(self.header['mmdd'] - mm*100)
                rtime = dt.datetime(self.header['year'], mm, dd) + dt.timedelta(hours=rhr)
                # Create new entries in rays dict
                if rtime not in self.paths.keys(): self.paths[rtime] = {}
                if raz not in self.paths[rtime].keys(): self.paths[rtime][raz] = {}
                self.paths[rtime][raz][rel] = {}
                # Read to paths dict
                self.paths[rtime][raz][rel]['nrstep'] = nrstep
                self.paths[rtime][raz][rel]['r'] = array( unpack('{}f'.format(nrstep), 
                    f.read(nrstep*4)) )
                self.paths[rtime][raz][rel]['th'] = array( unpack('{}f'.format(nrstep), 
                    f.read(nrstep*4)) )
                self.paths[rtime][raz][rel]['gran'] = array( unpack('{}f'.format(nrstep), 
                    f.read(nrstep*4)) )
                # self.paths[rtime][raz][rel]['pran'] = array( unpack('{}f'.format(nrstep), 
                #     f.read(nrstep*4)) )
                self.paths[rtime][raz][rel]['nr'] = array( unpack('{}f'.format(nrstep), 
                    f.read(nrstep*4)) )


    def write_to_ascii(self, fname):
        """Save rays to ASCII file (limited use)

        Parameters
        ----------
        fname : str
            filename to save to

        """

        with open(fname, 'w') as f:
            f.write('## HEADER ##\n')
            [f.write('{:>10s}'.format(k)) for k in self.header.keys()]
            f.write('\n')
            for v in self.header.values():
                if isinstance(v, float): strFmt = '{:10.2f}'
                elif isinstance(v, int): strFmt = '{:10d}'
                elif isinstance(v, str): strFmt = '{:10s}'
                f.write(strFmt.format(v))
            f.write('\n')
            f.write('##  RAYS  ##\n')
            for kt in sorted(self.paths.keys()):
                f.write('Time: {:%Y %m %d %H %M}\n'.format(kt))
                for kb in sorted(self.paths[kt].keys()):
                    f.write('--Beam/Azimuth: {}\n'.format(kb))
                    for ke in sorted(self.paths[kt][kb].keys()):
                        f.write('----Elevation: {:4.2f}\n'.format(ke))
                        f.write('------r\n')
                        [f.write('{:10.3f}\t'.format(r*1e-3)) for r in self.paths[kt][kb][ke]['r']]
                        f.write('\n')
                        f.write('------theta\n')
                        [f.write('{:10.5f}\t'.format(th)) for th in self.paths[kt][kb][ke]['th']]
                        f.write('\n')



#########################################################################
# Misc.
#########################################################################
def _read_header(fObj):
    """Read the header part of ray-tracing *.dat files

    Parameters
    ----------
    fObj :
        file object

    Returns
    -------
    header : dict
        a dictionary of header values

    """
    from struct import unpack
    import datetime as dt
    from collections import OrderedDict
    import os

    # Declare header parameters
    params = ('nhour', 'nazim', 'nelev', 
        'tlat', 'tlon', 
        'saz', 'eaz', 'daz', 
        'sel', 'eel', 'del', 
        'freq', 'nhop', 'year', 'mmdd', 
        'shour', 'ehour', 'dhour', 
        'hmf2', 'nmf2')
    # Read header
    header = OrderedDict( zip( params, unpack('3i9f3i5f', fObj.read(3*4 + 9*4 + 3*4 + 5*4)) ) )
    header['fext'] = unpack('10s', fObj.read(10))[0].strip()
    header['outdir'] = unpack('250s', fObj.read(250))[0].strip()
    header['indir'] = unpack('250s', fObj.read(250))[0].strip()
    # Only print header if in debug mode
    for k, v in header.items(): print('{:10s} :: {}'.format(k,v))
    header.pop('fext'); header.pop('outdir')
    header.pop('indir')

    return header


def _get_title(time, beam, header, name):
    """Create a title for ground/altitude plots

    Parameters
    ----------
    time : datetime.datetime
        time shown in plot
    beam :
        beam shown in plot
    header : dict
        header of fortran output file
    name : str
        radar name

    Returns
    -------
    title : str
        a title string

    """
    from numpy import floor, round

    utdec = time.hour + time.minute/60.
    tlon = (header['tlon'] % 360.)
    ctlon = tlon if tlon <=180. else tlon - 360.
    ltdec = ( utdec + ( ctlon/360.*24.) ) % 24.
    lthr = floor(ltdec)
    ltmn = round( (ltdec - lthr)*60 )
    title = '{:%Y-%b-%d at %H:%M} UT (~{:02.0f}:{:02.0f} LT)'.format(
        time, lthr, ltmn)
    title += '\n(IRI-2016) {} beam {}; freq {:.1f}MHz'.format(name, beam, header['freq'])

    return title

class Site(object):
    """
    This class creates a radar site object
    I'm doing this since the raytracing code 
    uses davitpy, which is deprecated now. 
    Adding this for pydarn compatibility
    """
    def __init__(self, radar_code, boresite, nbeams, ngates,\
                 geolat, geolon, beam_sep):
        self.radar_code = radar_code
        self.boresite = boresite
        self.nbeams = nbeams
        self.ngates = ngates
        self.geolat = geolat
        self.geolon = geolon
        self.beam_sep = beam_sep
        self.offset = nbeams/2. - 0.5
        
    def azim_to_beam(self, raz):
        """
        convert from azim to beam number
        """
        return int( round( ((raz-self.boresite)/self.beam_sep) + self.offset ) )
