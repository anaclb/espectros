
###imports and files###

import scipy as sp
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import os 
import numpy as np
import scripts as pjt
import scipy.signal as sg

def observado(data):
    hdulist = pyfits.open(data)
    hdu = hdulist[0]
    header = hdu.header
    lambda0, dx = header['crval1'], header['cdelt1'] #dx -> largura de pixel
    flux_obs = hdu.data
    wave_obs = np.arange(flux_obs.size)*dx+lambda0
    return  wave_obs,flux_obs

def simulado(lmb, flux):
    wave_data = pyfits.open(lmb)
    wave_sim = wave_data[0].data
    flux_data = pyfits.open(flux)
    flux_sim = flux_data[0].data
    return wave_sim, flux_sim

def gaussian(x, a, b, c, d):
    y = a*np.exp(-(x-b)**2/(2*c**2))+d
    return y

def cutfit(wave,flux,xmin,xmax):
    cutwave = wave[(wave>xmin)&(wave<xmax)]
    cutflux = flux[(wave>xmin)&(wave<xmax)]
    return cutwave,cutflux

def gaussfit(cutwave,cutflux):
    p=[-1,cutwave[cutflux.argmin()],.1,cutwave[cutflux.argmax()]]
    popt,r=(0,0,0,0), 0
    while r<.92 and p[2]>0:
        try:
            popt, pcov = spo.curve_fit(gaussian, cutwave, cutflux,p)
            y=gaussian(cutwave,*popt)
            s=(1-y/cutflux)**2
            r=1-s.sum()
            p[2]-=0.010 
        except RuntimeError or RuntimeWarning:
           print("RuntimeError/Warning, ignored line")
           break
           
           p[2]-=.010
    return popt,r

def eq_width(lmb,flux,lmb_central,delta=.2):
    xmax=lmb_central+delta
    xmin=lmb_central-delta
    cutwave, cutflux = cutfit(lmb, flux, xmin, xmax)
    popt,r=gaussfit(cutwave,cutflux)
    if popt[3]==0 or r<.98:
        return 0
    else:
        eqw= np.absolute(np.sqrt(2*np.pi)*popt[0]*popt[2]/popt[3] )
        return eqw*1000

def recta(x,m,b): 
    y = m*x+b
    return y

def cut_database(T_star,logg_0=3.5):
    data_list=np.array([])
    for data in database:
        T, logg=float(data[1:5]),float(data[7:11])
        if T<=T_star+400 and T>=T_star-400:
            if logg>=logg_0:
                data_list = np.append(data_list,data)
    return data_list

def EW_list(lmb,flux,lmb_central,delta=0.2):
    lmb_cut=lmb_central[(lmb_central>np.min(lmb)-delta)&(lmb_central<np.max(lmb)+delta)]
    EQW_list=np.zeros((len(lmb_cut),2),dtype=float)
    for i, lmb_0 in enumerate(lmb_cut):
        eqw=eq_width(lmb,flux,lmb_0,delta)
        EQW_list[i,0]=lmb_0
        EQW_list[i,1]=eqw
        ind = np.where(EQW_list[:,1] != 0)
    lmb_new, EQW_new = np.array([EQW_list[:,0][ind]]), np.array([EQW_list[:,1][ind]])
    EQW_list= np.concatenate((lmb_new.T,EQW_new.T),axis=1)
    return EQW_list,ind

def diff(EW_star,EW_sint):
    params,pcov=spo.curve_fit(recta,EW_sint,EW_star)
    return np.absolute(1-params[0]),params

def sint_approx(EW_star,data_star,lmb_sint,lmb0,delta):
    difs=np.array([])
    data_star=np.array(data_star)
    for data in data_star:
        lmb_sint,flux_sint=simulado(datalmb_sint,datab_directory+data)
        EW_sint,ind=EW_list(lmb_sint,flux_sint,lmb0,delta)
        EW_star=EW_star[(EW_star[:,0]>=lmb_sint[0])&(EW_star[:,0]<=lmb_sint[-1])]
        eqw_star=EW_star[ind]
        eqw_sint=EW_sint[(EW_sint[:,0]>=lmb_sint[0])&(EW_sint[:,0]<=lmb_sint[-1])]
        dif,params=diff(EW_star[:,1],eqw_sint[:,1])
        difs=np.append(difs,dif)
    index=difs.argmin()
    print(min(difs))
    print("file index ",index, " /// T estimated =",  data_star[index][1:5]," /// logg estimated =",data_star[index][7:11],"/// [Fe/H] =",data_star[index][22:27],"/// Indice químico =", data_star[index][29:35])
    for data in [data_star[index]]:
        lmb_sint,flux_sint=simulado(datalmb_sint,datab_directory+data)
        EW_sint,ind=EW_list(lmb_sint,flux_sint,lmb0,delta)
        EW_star=EW_star[(EW_star[:,0]>=lmb_sint[0])&(EW_star[:,0]<=lmb_sint[-1])]
        eqw_star=EW_star[ind]
        dif,params=diff(eqw_star[:,1],EW_sint[:,1])
        difs=np.append(difs,dif)
        plt.plot(eqw_star[:,1],EW_sint[:,1],'.')  
        plt.plot(eqw_star[:,1],eqw_star[:,1]*params[0]-params[1],'-',label="linear fit")
        plt.xlabel("EW - synthetic", fontsize=15)
        plt.ylabel("EW - observed", fontsize=15)
        plt.legend(fontsize=13)
        plt.show()
    return data_star[index]

def limdata_star(lmb0,EP,loggf,starwave):
    lmb0_new = lmb0[(lmb0 >= starwave[0]) & (lmb0 <= starwave[-1])]
    EP_new = EP[(lmb0 >= starwave[0]) & (lmb0 <= starwave[-1])]
    loggf_new = loggf[(lmb0 >= starwave[0]) & (lmb0 <= starwave[-1])]
    return lmb0_new,EP_new,loggf_new


def plot(EW_star,data_star,datalmb_sint,lmb0,delta):
    lmb_sint,flux_sint=simulado(datalmb_sint,datab_directory+data_star)
    EW_sint,ind=EW_list(lmb_sint,flux_sint,lmb0,delta)
    EW_star=EW_star[(EW_star[:,0]>=lmb_sint[0])&(EW_star[:,0]<=lmb_sint[-1])]
    eqw_star=EW_star[ind]
    difs=np.array([])
    dif,params=diff(eqw_star[:,1],EW_sint[:,1])
    difs=np.append(difs,dif)
    plt.plot(eqw_star[:,1],EW_sint[:,1],'.')  
    plt.plot(eqw_star[:,1],eqw_star[:,1]*params[0]+1.3*params[1],'-',label="linear fit")
    plt.xlabel("EW - synthetic", fontsize=15)
    plt.ylabel("EW - observed", fontsize=15)
    plt.legend(fontsize=13)
    plt.show()
