###comparação direta entre espetro simulado e espetro observado

###imports and files###

import scipy as sp
import scipy.signal as sg
import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import os 

data_directory='/home/bolacha/University/Astronomia Computacional/Projeto_1/GES_UVESRed580_deltaAlphaFe+0.0_fits/'
database=os.listdir(data_directory)
star1_data="estrela1.fits"
star2_data="estrela2_vcor.fits"

#########

def cut_database(T_star,logg_0=3.5):
    data_list=np.array([])
    for data in database:
        T , logg=float(data[1:5]),float(data[7:11])
        if T<=T_star+400 and T>=T_star-400:
            if logg>=logg_0:
                data_list = np.append(data_list,data)
    return data_list


def observado(data):
    hdulist = pyfits.open(data)
    hdu = hdulist[0]
    header = hdu.header
    lambda0, dx = header['crval1'], header['cdelt1'] #dx -> largura de pixel
    flux_obs = hdu.data
    wave_obs = np.arange(flux_obs.size)*dx+lambda0
    return flux_obs, wave_obs, dx, lambda0

def simulado(lmb, flux):
    wave_data = pyfits.open(lmb)
    wave_sim = wave_data[0].data
    flux_data = pyfits.open(flux)
    flux_sim = flux_data[0].data
    return wave_sim, flux_sim

def cutdata(wave, flux, xmin, xmax):
    cutwave = wave[(wave>xmin)&(wave<xmax)]
    cutflux = flux[(wave>xmin)&(wave<xmax)]
    return cutwave, cutflux

def param_gauss(cutwave_sim, R):
    xmed = (cutwave_sim[1]+cutwave_sim[-1])/2
    dlmb = xmed/R
    c = dlmb/(2*np.sqrt(2*np.log(2)))
    deltax = cutwave_sim[1]-cutwave_sim[0]
    x = np.arange(-5*c,5*c+deltax,deltax)
    return deltax, c, x

def gaussian(x, b, c):
    y = np.exp(-(x-b)**2/(2*c**2))
    return y / y.sum()

def conv(cutflux_sim, gauss):
    flux_sint = sg.convolve(cutflux_sim, gauss, mode="same") #espetro observado
    return flux_sint

def interp(cutwave_obs,cutwave_sim, flux_sint):
    sinterp = np.interp(cutwave_obs, cutwave_sim, flux_sint)
    return sinterp

def espectrocenas(obs, wave_sim, flux_sim, xmin, xmax, R=60000):
    flux_obs, wave_obs, dx, lambda0 = observado(obs)
    wave_sim, flux_sim = simulado(wave_sim, flux_sim)
    cutwave_obs, cutflux_obs = cutdata(wave_obs, flux_obs, xmin, xmax)
    cutwave_sim, cutflux_sim = cutdata(wave_sim, flux_sim, xmin, xmax)
    deltax, c, x = param_gauss(cutwave_sim, R)
    gauss = gaussian(x, 0, c)
    flux_sint = conv(cutflux_sim, gauss)
    sinterp = interp(cutwave_obs, cutwave_sim, flux_sint)
    return(cutwave_obs, sinterp, cutflux_obs)

data_star=np.array(cut_database(5309))

sint_dir="GES_UVESRed580_deltaAlphaFe+0.0_fits/"
sint_file="p6000_g+5.0_m0.0_t01_z-0.25_a+0.10.GES4750.fits" #star 1
#sint_file1="p5250_g+5.0_m0.0_t01_z+0.25_a+0.00.GES4750.fits" "star2"

T_est, logg_est, feH_est, indq = sint_file[1:5], sint_file[7:11], sint_file[22:27], sint_file[29:35]

cutwave_obs,sinterp,cutflux_obs=espectrocenas(star2_data,"GES_UVESRed580_Lambda.fits",sint_dir+sint_file,5700,5730)

plt.plot(cutwave_obs, sinterp,label=r"T = {}, log(g) = {}, [FE/H] = {}, $\alpha = {}$".format(T_est, logg_est, feH_est, indq))
plt.plot(cutwave_obs, cutflux_obs-.38,label=r"Observed data")
plt.xlabel(r"$\lambda [\AA]$",fontsize=15)
plt.ylabel("Flux, normalized", fontsize=15)
plt.legend(fontsize=10)
plt.show()

