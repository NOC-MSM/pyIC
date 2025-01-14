#based off IMMERSE_test_cases.ipynb

# Import packages
#import matplotlib.cm as cm
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.axes3d import Axes3D
#import seaborn as sns
#from matplotlib.colors import LightSource
#sns.set(color_codes=True)
import xarray as xr

from typing import Union

def rmax(dept):
    
    rmax_x = np.zeros_like(dept)
    rmax_y = np.zeros_like(dept)
    
    rmax_x[:,1:-1] = 0.5 * ( np.diff(dept[:,  :-1], axis=1) / (dept[:,  :-2] + dept[:, 1:-1]) +
                             np.diff(dept[:, 1:  ], axis=1) / (dept[:, 1:-1] + dept[:, 2:  ]) )
    rmax_y[1:-1,:] = 0.5 * ( np.diff(dept[ :-1, :], axis=0) / (dept[ :-2, :] + dept[1:-1, :]) +
                             np.diff(dept[1:  , :], axis=0) / (dept[1:-1, :] + dept[2:  , :]) )
    
    rmax = np.maximum(np.abs(rmax_x), np.abs(rmax_y))
    
    return rmax
    
def channel(lx = 390., ly = 294., dx = 6., dy = 6., stiff = 1, H0 = 4500.):
    ''' 
    Creates a channel idealised domain

    This function produces bathymetry of a channel with a Gaussian 
    seamount in order to simulate an idealised test case.

    Args:
        lx, ly     (float): dimensions of the domain (km)
        dx, dy     (float): length of grid cells (km)
        stiff      (float): scale factor for steepness of seamount

    Returns:
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        x, y  (np.ndarray): coordinates (km)
    '''
    
    # Work out the number of grid points in I/J
    nx, ny = np.rint(lx/dx), np.rint(ly/dy)
    nx_int, ny_int = nx.astype(np.int32), ny.astype(np.int32)
    mx, my = (nx_int+1)/2, (ny_int+1)/2
    
    # Redefine cell size to factor into lx and ly
    dx, dy = lx/nx, ly/ny
    
    # Define T-grid
    x = np.linspace(0. + dx/2., lx - dx/2., num=nx_int)
    y = np.linspace(0. + dy/2., ly - dy/2., num=ny_int)
    x, y = np.meshgrid(x, y)

    # Define Gaussian bowl and R-max
    zt = np.ma.array(H0*(1. - 0.9*np.exp(-(stiff/40.**2 * ((x-lx/2.)**2 
                                  + (y-ly/2.)**2)))), mask=False)
    rm = np.ma.array(rmax(zt), mask=False)
    
    return x, y, zt, rm

def slope(lx = 500., ly = 500., dx = 10., dy = 10., lx2 = 50., ly2 = 100., stiff = 1):
    ''' 
    Creates a idealised sloped domain

    This function produces sloped bathymetry in order to 
    simulate an idealised overflow test case.

    Args:
        lx, ly     (float): dimensions of the domain (km)
        dx, dy     (float): length of grid cells (km)
        lx2, ly2   (float): dimensions of dense water inlet (km)
        stiff      (float): scale factor for steepness of seamount

    Returns:
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        x, y  (np.ndarray): coordinates (km)
    '''
    
    # Work out the number of grid points in I/J
    nx, ny = np.rint(lx/dx), np.rint(ly/dy)
    nx_int, ny_int = nx.astype(np.int32), ny.astype(np.int32)
    mx, my = (nx_int+1)/2, (ny_int+1)/2
    
    # Redefine cell size to factor into lx and ly
    dx, dy = lx/nx, ly/ny
  
    # Add the ledge to nx
    nx2, ny2 = np.floor(lx2/dx), np.floor(ly2/dy)
    nx2_int, ny2_int = nx2.astype(np.int32), ny2.astype(np.int32)
    lx2 = nx2*dx
    
    # Define T-grid
    x = np.linspace(-lx2 + dx/2., lx - dx/2., num=nx_int + nx2_int)
    y = np.linspace(   0.+ dy/2., ly - dy/2., num=ny_int)
    x, y = np.meshgrid(x, y)

    # Define Gaussian bowl and R-max
    zt = np.ma.array(stiff * 10. * x + 600., mask=False)
    zt = np.ma.where(zt< 600.,  600., zt)
    zt = np.ma.where(zt>3600., 3600., zt)
    rm = np.ma.array(rmax(zt), mask=False)
    
    # Create source region (i.e. channel where dense water will be released)
    zt.data[:ny_int,:nx2_int] = 0.
    zt.data[ny_int+ny2_int:,:nx2_int] = 0.
    
    return x, y, zt, rm

def half_bowl(lx = 500., ly = 500., dx = 10., dy = 10., lx2 = 50., ly2 = 100., stiff = 1):
    ''' 
    Creates a half bowl idealised domain

    This function produces a half Gaussian-like bathymetry 
    in order to simulate an idealised overflow test case.

    Args:
        lx, ly     (float): dimensions of the domain (km)
        dx, dy     (float): length of grid cells (km)
        lx2, ly2   (float): dimensions of dense water inlet (km)
        stiff      (float): scale factor for steepness of slope

    Returns:
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        x, y  (np.ndarray): coordinates (km)
    '''
    
    # Work out the number of grid points in I/J
    nx, ny = np.rint(lx/dx), np.rint(ly/dy)
    nx_int, ny_int = nx.astype(np.int32), ny.astype(np.int32)
    mx, my = (nx_int+1)/2, (ny_int+1)/2
    
    # Redefine cell size to factor into lx and ly
    dx, dy = lx/nx, ly/ny
  
    # Add the ledge to nx
    nx2, ny2 = np.floor(lx2/dx), np.floor(ly2/dy)
    nx2_int, ny2_int = nx2.astype(np.int32), ny2.astype(np.int32)
    lx2 = nx2*dx
    
    # Define T-grid
    x = np.linspace(-lx2 + dx/2., lx - dx/2., num=nx_int + nx2_int)
    y = np.linspace(   0.+ dy/2., ly - dy/2., num=ny_int)
    x, y = np.meshgrid(x, y)

    # Define Gaussian bowl and R-max
    zt = np.ma.array(3000.*np.exp(-(stiff * 10./(lx/2.)**2)*(x-lx/2.)**2 
                                  -(stiff * 10./(ly/2.)**2)*(y-ly/2.)**2) + 600., mask=False)
    rm = np.ma.array(rmax(zt), mask=False)
    
    # Open bowl
    mx = int(mx-0.5)
    zt.data[:,mx+nx2_int:] = zt.data[:,mx+nx2_int-1:mx+nx2_int]
    rm.data[:,mx+nx2_int:] = rm.data[:,mx+nx2_int-1:mx+nx2_int]
    
    # Create source region (i.e. channel where dense water will be released)
    my = int(my-0.5)
    zt.data[:my,:nx2_int] = 0.
    zt.data[my+ny2_int:,:nx2_int] = 0.
    
    return x, y, zt, rm

def plot_dom(x, y, zt, rm, title):
    ''' 
    Plot idealised domain

    Plots the idealised overflow test case as a 3D surface
    plot, coloured with the associated r-max value for reference.

    Args:
        x, y  (np.ndarray): coordinates (km)
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        
    Returns:
    '''
    
    # Set figure and axes handles
    fig = plt.figure(figsize=plt.figaspect(0.33)*1.5)
    ax = fig.add_subplot(projection='3d')

    # Create light source object, shade data of r-max and rgb array
    ls = LightSource(azdeg=0, altdeg=90)
    rgb = ls.shade(rm, plt.cm.RdYlBu)

    # Plot surface
    surf = ax.plot_surface(x, y, zt, rstride=1, cstride=1, linewidth=0,
                           antialiased=False, facecolors=rgb)

    # Scrape colour map data for r-max
    m = cm.ScalarMappable(cmap=cm.RdYlBu)
    m.set_array(rm)

    # Tidy and annotate
    cb = fig.colorbar(m,ax=ax,shrink=0.8, aspect=8)
    ax.set_zlim(0,)
    ax.invert_zaxis()
    cb.set_label('R-max')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Depth [m]') 
    
    dx, dy = x[0,1]-x[0,0], y[1,0]-y[0,0]
    title = title+' (dx={}, dy={})'.format(dx, dy)
    plt.title(title,fontweight='bold')
    plt.show()

def temperature_profile(depth: Union[xr.DataArray, float]) -> Union[np.ndarray, float]:
    """    Computes the temperature (in °C) at a given depth (in metres).
    Parameters:        
    depth (DataArray or float): Depth in meters (0 to 6000).
    Returns:        numpy array or float: Temperature in °C.   
    """    
    depth = xr.DataArray(depth) if not isinstance(depth, xr.DataArray) else depth
    # Smooth temperature profile with a thermocline    
    surface_temp = 20
    deep_temp = -2
    thermocline_depth = 1000
    # Depth of the thermocline
    scale_depth = 200  
    # Sharpness of the thermocline transition    
    temperature = deep_temp + (surface_temp - deep_temp) / (1 + np.exp((depth - thermocline_depth) / scale_depth))    
    return temperature

def salinity_profile(depth: Union[xr.DataArray, float]) -> Union[np.ndarray, float]:
    """    Computes the salinity (in PSU) at a given depth (in meters).        
    Parameters:        
    depth (DataArray or float): Depth in meters (0 to 6000).    
    Returns:        
    numpy array or float: Salinity in PSU.    
    """    
    
    depth = xr.DataArray(depth) if not isinstance(depth, xr.DataArray) else depth
    # Smooth salinity profile using an exponential model    
    surface_salinity = 34.5    
    deep_salinity = 32.7    
    scale_depth = 1500  
    # Characteristic depth for salinity stabilization    
    salinity = deep_salinity - (deep_salinity - surface_salinity) * np.exp(-depth / scale_depth)    
    return salinity
'''
# Generate depth array
depths = xr.DataArray(np.linspace(0, 6000, 1000), dims=["depth"], name="depth")
# Compute profiles
temperatures = temperature_profile(depths)
salinities = salinity_profile(depths)
# Plot profiles
fig, ax1 = plt.subplots(figsize=(6, 8))
ax1.plot(temperatures, depths, label='Temperature (°C)', color='red')
ax1.set_xlabel('Temperature (°C)', color='red')
ax1.set_xlim(-2, 25)
ax1.set_ylim(6000, 0)
ax1.invert_yaxis()
ax1.tick_params(axis='x', labelcolor='red')
ax2 = ax1.twiny()
ax2.plot(salinities, depths, label='Salinity (PSU)', color='blue')
ax2.set_xlabel('Salinity (PSU)', color='blue')
ax2.set_xlim(30, 35)
ax2.tick_params(axis='x', labelcolor='blue')
# Add legends and title
fig.suptitle('Typical Ocean Temperature and Salinity Profiles')
plt.savefig('profile.png') 
'''


def make_rand(ldepths,add_rand=True):
    if add_rand:
        return np.random.random(size=(ldepths))/4
    return 0        

def make_dataset(dx=10.,dy=10.,domain='half_bowl',max_depth=3600,times=2):
    add_rand=False
    domain_types = ['half_bowl','slope','channel']
    if domain not in domain_types:
        raise Exception(f"Domain type '{domain}' not known, only acceptable domains are: {domain_types}.")
    if domain == 'half_bowl':
        x, y, zt, rm = half_bowl(lx = 450., ly = 500., dx=dx, dy= dx, stiff=2)
    if domain == 'slope':
        x, y, zt, rm = slope()
    if domain == 'channel':
        x, y, zt, rm = channel()
    max_depth = max(max_depth,np.max(zt))
    depths = np.arange(0,max_depth,100)
    ds = xr.Dataset(coords={'time_counter':np.arange(0,times,1),'depth':depths,'y':y[:,0],'x':x[0,:]})
    temps_grid = np.zeros((times,len(depths),len(x),len(y)))
    salinity_grid = np.zeros((times,len(depths),len(x),len(y)))
    nan_value = 0
    for t in range(times):
        temperatures = temperature_profile(depths)
        salinity = salinity_profile(depths)
        for i in range(len(x)):
            for j in range(len(y)):
                temps_grid[t,:,i,j] = np.where(depths<zt[i,j],temperatures+make_rand(len(depths),add_rand=add_rand),0)
        for i in range(len(x)):
            for j in range(len(y)):
                salinity_grid[t,:,i,j] = np.where(depths<zt[i,j],salinity+make_rand(len(depths),add_rand=add_rand),0)
    ds['temperature'] = (('time_counter','depth','y','x'),temps_grid)
    ds['salinity'] = (('time_counter','depth','y','x'),salinity_grid)
    ds['lon'] = (['y','x'], np.meshgrid(x[0]/10,y[:,0]/10)[0])
    ds['lat'] = (['y','x'], np.meshgrid(x[0]/10,y[:,0]/10)[1])
    ds = ds.assign_coords({'lon':ds.lon,'lat':ds.lat})
    ds = ds.drop_vars(['x','y'])
    return ds