import numpy as np
import xarray as xr
import mtspec

import cftime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point as cyclic
import matplotlib.path as mpath
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

import eofs
from eofs.xarray import Eof
import copy


import warnings

# import numba

warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
warnings.filterwarnings('ignore','SerializationWarning')

###########################################################

def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

###########################################################

def set_circular_boundary(ax):
    theta = np.linspace(0, 2*np.pi, 400)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circlePath = mpath.Path(verts * radius + center)
    ax.set_boundary(circlePath, transform=ax.transAxes)
    return circlePath

###########################################################

def set_wedge_boundary(ax, minLon, maxLon, minLat, maxLat):
    wedgeLons = np.concatenate((np.linspace(minLon, maxLon, 50),
                                np.linspace(maxLon, maxLon, 50),
                                np.linspace(maxLon, minLon, 50),
                                np.linspace(minLon, minLon, 50)))
    wedgeLats = np.concatenate((np.linspace(minLat, minLat, 50),
                                np.linspace(minLat, maxLat, 50),
                                np.linspace(maxLat, maxLat, 50),
                                np.linspace(maxLat, minLat, 50)))
    wedgePath = mpath.Path(np.dstack((wedgeLons, wedgeLats))[0])
    ax.set_boundary(wedgePath, transform=ccrs.PlateCarree())
    return wedgePath

###########################################################

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
          'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
           cmap(np.linspace(minval, maxval, n)))
    return new_cmap 

###########################################################

def EOF_analysis(ds, weights=None, n=3, scale_eofs = False):
    
    """ Empirical Orthogonal Function analysis of MSLP field;  """   
    
#     assert type(ds)==xr.core.dataarray.DataArray
#     if weights.any()!=None:
#         assert type(weights)==xr.core.dataarray.DataArray
#         assert np.shape(ds[0,:,:])==np.shape(weights)
        
    # Retrieve the leading EOF, expressed as the covariance between the leading PC
    # time series and the input xa anomalies at each grid point.
    solver = Eof(ds, weights=weights, center=True)
    eofs = solver.eofsAsCovariance(neofs=n)
    if scale_eofs == True:
        for i in range(n):
            eofs[i] = eofs[i]/eofs[i].std()
    pcs  = solver.pcs(npcs=n, pcscaling=1)
#    eigs = solver.eigenvalues(neigs=n)
    varF = solver.varianceFraction(neigs=n)
    data = xr.merge([eofs, pcs, varF])
     
    return data

###########################################################
# Calculate NAM index
# slp = monthly sea level pressure data 
# lats = [lat1, lat2]; 
# norm = normalize by std deviation

def Calculate_NAM_index(slp, lats = [35, 65], norm = True):
    
    dl = 2; # +- degrees latitude
    
    if slp.dims[2] == 'longitude': slp = slp.rename({'longitude':'lon'});
    if slp.dims[1] == 'latitude': slp = slp.rename({'latitude':'lat'}); slp = slp.sortby(slp.lat);
    
    p35 = slp.mean(dim='lon').sel(lat = slice(lats[0]-dl, lats[0]+dl)).mean(dim='lat')
    p65 = slp.mean(dim='lon').sel(lat = slice(lats[1]-dl, lats[1]+dl)).mean(dim='lat')

    pdiff = p35 - p65
    NAM = pdiff - pdiff.mean()
    
    if norm == True:
        NAM = NAM/NAM.std();
        
    return NAM
    
###########################################################
# Pressure interpolation scheme 
# from hybrid sigma levels to set 'plev' levels
# either for climatology files or monthly files

def pressure_interpolation(ds, data, description = None, climatology = True):

    # define your new pressure levels (in hPa)
    Plev = np.concatenate([np.array([4,7,10,15]), np.arange(20,61,10),
                            np.arange(80,201,20), np.arange(250,401,50),
                            np.arange(500,801,100), np.array([850]),
                            np.arange(900,1001,25)])

    if climatology == True:
    
        # hybrid sigma levels. NOTE this is 4D! (record, lev, lat, lon)
        Phyb = 0.01 * (ds.hyam * ds.P0 + ds.hybm * ds.PS.isel(time=0)); # note in hPa

        # integration in height so log(p)
        Plev_log = np.log(Plev); Phyb_log = np.log(Phyb);

        y = data.values
        y_int = np.zeros((len(ds.record), len(Plev), len(ds.lat), len(ds.lon)))

        # perform integration
        for i in range(len(ds.record)):
            for j in range(len(ds.lat)):
                for k in range(len(ds.lon)):
                    y_int[i,:,j,k] = np.interp(Plev_log, Phyb_log[i,:,j,k], y[i,:,j,k])

        # collect interpolated data in an xarray DataArray
        output = xr.DataArray(data = y_int,
                             dims = ["record", "plev", "lat", "lon"],
                              coords = dict(record = ds.record, 
                                           plev = Plev,
                                           lat = ds.lat,
                                           lon = ds.lon),
                              attrs = dict(description = description,
                                          units = data.units))
    
    elif climatology == False:
        
        Plev = 100*Plev ## from hPa to Pa
        
        # hybrid sigma levels. NOTE this is 4D! (lev, time, lat, lon)
        Phyb = (ds.hyam * ds.P0 + ds.hybm * ds.PS); # in Pa

        # integration in height so log(p)
        Plev_log = np.log(Plev); Phyb_log = np.log(Phyb);

        y = data.values
        y_int = np.zeros((len(ds.time), len(Plev), len(ds.lat), len(ds.lon)))

        # perform integration
        for i in range(len(ds.time)):
            for j in range(len(ds.lat)):
                for k in range(len(ds.lon)):
                    y_int[i,:,j,k] = np.interp(Plev_log, Phyb_log[:,i,j,k], y[i,:,j,k])

        # collect interpolated data in an xarray DataArray
        output = xr.DataArray(data = y_int,
                             dims = ["time", "plev", "lat", "lon"],
                              coords = dict(time = ds.time, 
                                           plev = Plev,
                                           lat = ds.lat,
                                           lon = ds.lon),
                              attrs = dict(description = description,
                                          units = data.units))
    
    return output

    

###########################################################
## ------------------- other functions --------------------
###########################################################

def FFT_spectrum(time_series, normalise = True, scale = True, trend = None):

    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(time_series))
        p = np.polyfit(t, time_series, 1)
        l = t*p[0] + p[1]
        time_series = time_series - l

    # Normalise
    if normalise == True:
        time_series == (time_series - np.mean(time_series))/np.std(time_series)

    # Get the Fourier Spectrum of original time series
    freq_series = np.fft.fft(time_series) #Take fourier spectrum
    freq_series = ((np.real(freq_series)**2.0) + (np.imag(freq_series)**2.0)) #Determine power law (absolute value)
    freq        = np.fft.fftfreq(len(time_series))
    freq_series_original = freq_series[1:freq.argmax()] #Restrict to f = 0.5
    freq        = freq[1:freq.argmax()] #Restrict to f = 0.5
    
    # Scale power spectrum on sum (if True)
    if scale == True:
        freq_series_original = freq_series_original / np.sum(freq_series_original)
    
    return freq, freq_series_original

def MT_spectrum(time_series, normalise = True, scale = True, trend = None, nbw = 2, ntapers = 3):
    
    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(time_series))
        p = np.polyfit(t, time_series, 1)
        l = t*p[0] + p[1]
        time_series = time_series - l
    
    # Normalise
    if normalise == True:
        time_series = (time_series - np.mean(time_series))/np.std(time_series)
        
    spec, freq = mtspec.mtspec(
                data=time_series, delta=1., time_bandwidth=nbw,
                number_of_tapers=ntapers, statistics=False)
    
    spec = spec[1:freq.argmax()]
    freq = freq[1:freq.argmax()]
    
    # Scale power spectrum on sum (if True)
    if scale == True:
        spec = spec / np.sum(spec)
    
    return freq, spec

def Confidence_intervals(time_series, normalise=True, scale = True, trend = None, mode = 'FFT', N_autocorr = 250, N_surrogates = 1000, nbw = 2, ntapers = 3):
    
    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(time_series))
        p = np.polyfit(t, time_series, 1)
        l = t*p[0] + p[1]
        time_series = time_series - l

    if type(time_series) == xr.core.dataarray.DataArray:
        time_series = time_series.values
    
    # normalise
    if normalise == True:
        time_series = (time_series - np.mean(time_series))/np.std(time_series)

    #First determine the auto-correlation for each time series
    N = N_autocorr;
    auto_lag  = np.arange(N)
    auto_corr = np.zeros(len(auto_lag))

    for lag_i in range(len(auto_lag)):
        #Determine the auto-correlation
        auto_corr[lag_i] = np.corrcoef(time_series[0:len(time_series)-lag_i], time_series[lag_i:])[0][1]

    #Determine the e-folding time and keep the minimum lag
    e_1 =  np.where(auto_corr < 1.0/np.e)[0][0]

    #Determine the first coefficient for the AR(1) process
    #The auto-lag(1) is equivalent to -1.0/(a - 1)
    a   = -1.0/(e_1) + 1.0 

    #Determine the variance in the time series and the last coefficient for the AR(1) process
    var = np.var(time_series)
#     b   = np.sqrt((1.0 - a**2.0) * var)
    b   = np.sqrt(var)
    
    # generate Monte Carlo sample 
    mc_series = mc_ar1_ARMA(phi=a, std=b, n=len(time_series), N=N_surrogates)
    
    for surrogate_i in range(N_surrogates):
        #Generating surrogate spectra
        
        # select spectrum type
        if mode == 'FFT':
            freq, spec = FFT_spectrum(mc_series[surrogate_i,:], normalise=False, scale=scale)
        elif mode == 'MT':
            freq, spec = MT_spectrum(mc_series[surrogate_i,:], normalise=False, scale=scale, nbw = nbw, ntapers = ntapers)
            
        # create array for spectra    
        if surrogate_i==0: 
            surrogate_spec = np.ma.masked_all((N_surrogates, len(freq)))
        
        # allocate surrogate spectrum
        surrogate_spec[surrogate_i] = spec

    CI_90 = np.percentile(surrogate_spec, 90, axis = 0)
    CI_95 = np.percentile(surrogate_spec, 95, axis = 0)
    CI_99 = np.percentile(surrogate_spec, 99, axis = 0)
        
    return CI_90, CI_95, CI_99

def mc_ar1_ARMA(phi, std, n, N=1000):
    """ Monte-Carlo AR(1) processes
    input:
    phi .. (estimated) lag-1 autocorrelation
    std .. (estimated) standard deviation of noise
    n   .. length of original time series
    N   .. number of MC simulations 
    """
    AR_object = ArmaProcess(np.array([1, -phi]), np.array([1]), nobs=n)
    mc = AR_object.generate_sample(nsample=(N,n), scale=std, axis=1, burnin=1000)
    
    return mc

def EOF_SST_analysis(model = 'CCSM4', run = 'E280', latbound = 23, weights=None, n=1):
    
    """ Empirical Orthogonal Function analysis of SST(t,x,y) field;  """

    ## latitude and longitude bounds of tropical Pacific
    minlat = -latbound;  maxlat = latbound;
    minlon = 140; maxlon = 280;
    
    ## Open file 
    file = f'models/{model}/{model}_{run}.SST.timeseries_no_ann_cycle.nc'
    ds   = xr.open_dataset(file)
        
    ## rename tos, lat, etc
    if model == 'CCSM4' or model == 'CESM1.2' or model == 'CESM2':
        ds = ds.rename({'TS':'tos'})
    elif model == 'EC-Earth3.3' or model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'sst':'tos'})
    elif model == 'HadCM3':
        ds = ds.rename({'temp':'tos'})
        
    if model == 'CCSM4-UoT':
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'})
        
    if model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'HadGEM3':  
        ds = ds.rename({'time_counter':'time'})
    assert 'time' in ds.dims      
    
    assert type(ds.tos)==xr.core.dataarray.DataArray
    if weights!=None:
        assert type(weights)==xr.core.dataarray.DataArray
        assert np.shape(ds.tos[0,:,:])==np.shape(weights)

    ## Select SSTs in tropical pacific
    ds = ds.sortby(ds.latitude)
    sst_TP = ds.tos.sel(latitude = slice(minlat, maxlat)).sel(longitude = slice(minlon, maxlon))
        
    # Retrieve the leading EOF, expressed as the covariance between the leading PC
    # time series and the input xa anomalies at each grid point.
    solver = Eof(sst_TP, weights=weights, center=True)
    eofs = solver.eofsAsCovariance(neofs=n)
    pcs  = solver.pcs(npcs=n, pcscaling=1)
#    eigs = solver.eigenvalues(neigs=n)
    varF = solver.varianceFraction(neigs=n)
    data = xr.merge([eofs, pcs, varF])
     
    return data

###########################################################

# @numba.jit(nopython=True)
def lowess_1d(y, x, alpha=2. / 3., it=10):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(y)
    r = int(np.ceil(alpha * n))   
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(it):
        for i in range(n):
            h = np.sort(np.abs(x - x[i]))[r]
            dist = np.abs((x - x[i]) / h)
            dist[dist < 0.] = 0.
            dist[dist > 1.] = 1.
            w = (1 - dist ** 3) ** 3
            weights = delta * w
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]    
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = residuals / (6.0 * s)
        dist[dist < -1.] = -1.
        dist[dist > 1.] = 1.
        delta = (1 - delta ** 2) ** 2
    return yest

def lowess(obj, dim, alpha=0.5, it=5):
    """
    Apply a LOWESS smoothing along one dimension
    
    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset 
        The input dataarray or dataset
    dim : str
        The dimension along which the computation will be performed
    alpha : float, optional
        Span of the smoothing
    it : int, optional
        Number of iterations
    
    Returns
    -------
    res : xarray.DataArray, xarray.Dataset
        The estimated lowess smoothing
    """
    if isinstance(obj, xr.DataArray):
        if isinstance(obj[dim].data[0], cftime.DatetimeGregorian):
            res = np.apply_along_axis(lowess_1d, obj.get_axis_num(dim), obj.data, 
                                      np.arange(0,len(obj[dim])), alpha=alpha, it=it)
        else:
            res = np.apply_along_axis(lowess_1d, obj.get_axis_num(dim), obj.data, 
                                      obj[dim].astype('f4').data, alpha=alpha, it=it)
        return xr.DataArray(res, coords=obj.coords, dims=obj.dims)
    elif isinstance(obj, xr.Dataset):
        return obj.map(lowess_1d, keep_attrs=True, alpha=alpha, it=it)
#     else:
#         raise ValueError

def change_lons(ds):
    ds_ = ds
    ds_.coords['lon'] = (ds_.coords['lon'] + 180) % 360 - 180;
    ds_new = ds_.sortby(ds_.lon)
    return ds_new

def contourlines(data, ax):

    c1 = data.plot.contour(ax=ax, transform = ccrs.PlateCarree(), levels = [-2,-1,0,1,2], colors = ['w', 'w', 'k', 'w', 'w'], linestyles = ['-.', '-.', '-.', '--', '--'], linewidths = 1)
    c2 = ax.clabel(c1, c1.levels, fontsize=12, inline=True, fmt = '%1.0f')
    
    return c1, c2

def lat_ticks(ax, mode='NH'):

    if mode == 'NH':
        c1 = ax.set_yticks([30, 45, 60, 75], crs=ccrs.PlateCarree()); 
        c2 = ax.set_yticklabels([30, 45, 60, 75], fontsize=11);
        
    if mode == 'NH2':
        c1 = ax.set_yticks([0, 15, 30, 45, 60, 75], crs=ccrs.PlateCarree()); 
        c2 = ax.set_yticklabels([0, 15, 30, 45, 60, 75], fontsize=11);
    
    if mode == 'global':
        c1 = ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree()); 
        c2 = ax.set_yticklabels([-60, -30, 0, 30, 60], fontsize=11);
        
    c3 = ax.yaxis.tick_left(); 
    c4 = ax.set_ylabel(''); 
    c5 = ax.yaxis.set_major_formatter(cticker.LatitudeFormatter());
    
    return c1, c2, c3, c4, c5

def lon_ticks(ax, direction = 'top', lons = [0, 60,120,180, -120,-60]):
    
    c1 = ax.set_xticks(lons, crs=ccrs.PlateCarree()); 
    c2 = ax.set_xticklabels(lons, fontsize=11);
    if direction == 'top': c3 = ax.xaxis.tick_top(); 
    elif direction == 'bottom': c3 = ax.xaxis.tick_bottom(); 
    c4 = ax.set_xlabel(''); 
    c5 = ax.xaxis.set_major_formatter(cticker.LongitudeFormatter()); 
    
    return c1, c2, c3, c4, c5

def varfs(data):
    
    varfs = np.round(100*data.variance_fractions.values,decimals=1);

    return varfs

def compute_SLP_eofs(data, gridweights, nmodes = 3, lats = [20, 85], sector = 'NH', sim = 'PI', lowess_filter = False, rmlen = 30):
    
    NH = data.sel(lat = slice(lats[0], lats[1])); 
    gw_ = gridweights.sel(lat = slice(lats[0], lats[1]));

    nlen = len(data["time"]); 
    nplons = [120, 240]; nalons = [-90, 30];

    if lowess_filter == True: NH_ = NH - lowess(NH, dim = "time", alpha = rmlen/nlen, it = 1); 
    else: NH_ = NH - NH.mean("time")

    if sector == 'NP':
        NH_ = NH_.sel(lon = slice(nplons[0], nplons[1]));
        gw_ = gw_.sel(lon = slice(nplons[0], nplons[1]));
    elif sector == 'NA':
        NH_ = change_lons(copy.copy(NH_)).sel(lon = slice(nalons[0], nalons[1])); 
        gw_ = change_lons(copy.copy(gw_)).sel(lon = slice(nalons[0], nalons[1]));
    
    eofs = EOF_analysis(NH_, weights = gw_, n = nmodes, scale_eofs=True)

#     for i in range(nmodes):
#         if eofs.eofs.isel(mode=i).sel(lat=slice(75,85)).mean() < 0:
#             eofs.eofs[i] = -1 * eofs.eofs[i]; eofs.pcs[:,i] = -1 * eofs.pcs[:,i];
    
    if sim == 'PI':     n=[1,0];
    elif sim == 'Plio': n=[0,1];
    
    if sector == 'NH':
        if eofs.eofs.isel(mode=0).sel(lat=slice(75,85)).mean() < 0: #NHem
            eofs.eofs[0] = -1 * eofs.eofs[0]; eofs.pcs[:,0] = -1 * eofs.pcs[:,0];
    if sector == 'NA':
        if eofs.eofs.isel(mode=0).sel(lat=slice(75,85)).mean() < 0: #NAtl-z
            eofs.eofs[0] = -1 * eofs.eofs[0]; eofs.pcs[:,0] = -1 * eofs.pcs[:,0];
        if eofs.eofs.isel(mode=1).sel(lat=slice(50,60)).mean() > 0: #NAtl-a
            eofs.eofs[1] = -1 * eofs.eofs[1]; eofs.pcs[:,1] = -1 * eofs.pcs[:,1];
    if sector == 'NP':
        if eofs.eofs.isel(mode=n[0]).sel(lat=slice(75,85)).mean() < 0: #NPac-z
            eofs.eofs[n[0]] = -1 * eofs.eofs[n[0]]; eofs.pcs[:,n[0]] = -1 * eofs.pcs[:,n[0]];
        if eofs.eofs.isel(mode=n[1]).sel(lat=slice(50,60)).mean() > 0: #NPac-a
            eofs.eofs[n[1]] = -1 * eofs.eofs[n[1]]; eofs.pcs[:,n[1]] = -1 * eofs.pcs[:,n[1]];
            
    sd = NH_.std("time")
    
    return eofs, sd

def eofs_to_nc(ds, descr, simulation, sector, folder, filename):

    ds.attrs['description'] = descr
    ds.attrs['filtering'] = 'lowess 50 year'
    ds.attrs['simulation'] = simulation
    ds.attrs['sector'] = sector
    ds.to_netcdf(folder+filename+'.nc', mode = 'w')
    
    return

def varperc(eofs, mode=0):
    varperc = np.array(np.round(100*eofs.isel(mode=mode).variance_fractions.values,decimals=1))
    return varperc

def calc_siva(ds, gridweights, sector='NHem'):

    rmlen = 50; 
    nplons = [120, 240]

    if sector == 'NHem':
        icevol = ((ds.icefrac) * gridweights).sel(lat=slice(20,90)).sum("lat").sum("lon") / (gridweights.sel(lat=slice(20,90)).sum())
    elif sector == 'NPac':
        icevol = ((ds.icefrac) * gridweights).sel(lat=slice(20,90)).sel(lon=slice(120,240)).sum("lat").sum("lon") / (gridweights.sel(lat=slice(20,90)).sel(lon=slice(120,240)).sum())
    elif sector == 'NAtl':
        icevol1 = ((ds.icefrac) * gridweights).sel(lat=slice(20,90)).sel(lon=slice(270,360)).sum("lat").sum("lon") / (gridweights.sel(lat=slice(20,90)).sel(lon=slice(270,360)).sum())
        icevol2 = ((ds.icefrac) * gridweights).sel(lat=slice(20,90)).sel(lon=slice(0,30)).sum("lat").sum("lon") / (gridweights.sel(lat=slice(20,90)).sel(lon=slice(0,30)).sum())
        icevol = icevol1+icevol2
        
    icevol = icevol - lowess(icevol, dim = "time", alpha = rmlen/len(icevol), it = 1)
    niva = icevol/icevol.std()

    return niva

def scatter_hist(x, y, ax_histx, ax_histy, binsx, binsy, color):

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax_histx.hist(x, histtype='step',linewidth=2, bins=binsx, color=color)
    ax_histy.hist(y, histtype='step',linewidth=2, bins=binsy, color=color, orientation='horizontal')