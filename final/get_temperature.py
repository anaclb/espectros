##imports and files####

import scipy as sp
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import os 
import numpy as np
import scipy.signal as sg

###########scripts###

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
    popt,r=(0, 0, 0, 0), 0
    while r<.95 and p[2]>0:
        try:
            popt, pcov = spo.curve_fit(gaussian, cutwave, cutflux,p)
            y=gaussian(cutwave,*popt)
            s=(1-y/cutflux)**2
            r=1-s.sum()
            p[2]-=0.010
        except RuntimeError or RuntimeWarning:
           print("RuntimeError/Warning, ignored line")
           p[2]-=.010
           break
    return popt,r

def eq_width(lmb,flux,lmb_central,delta=.2):
    xmax=lmb_central+delta
    xmin=lmb_central-delta
    cutwave, cutflux = cutfit(lmb, flux, xmin, xmax)
    popt,r=gaussfit(cutwave,cutflux)
    if popt[3]==0 or r<.98:
        return 0
    else:
        eqw= np.absolute(np.sqrt(2*np.pi)*popt[0]*popt[2] / popt[3])
        return eqw*1000 

def eq_width_test(lmb,flux,lmb_central,delta=.2):
    xmax=lmb_central+delta
    xmin=lmb_central-delta
    cutwave, cutflux = cutfit(lmb, flux, xmin, xmax)
    popt,r=gaussfit(cutwave,cutflux)
    y=gaussian(cutwave,*popt)
    if popt[3]==0 or r<.98:
        return 0
    else:
        eqw= np.absolute(np.sqrt(2*np.pi)*popt[0]*popt[2] / popt[3])
        return eqw*1000,y,cutwave,cutflux


def EW_list(lmb, flux,lmb_central,delta=.2):
    EQW_list=np.zeros((len(lmb_central),2),dtype=float)
    for i, lmb_0 in enumerate(lmb_central):
        eqw0=eq_width(lmb,flux,lmb_0,delta)
        EQW_list[i,0]=lmb_0
        EQW_list[i,1]=eqw0
    ind = np.where(EQW_list[:,1] != 0)
    lmb, EQW = np.array([EQW_list[:, 0][ind]]), np.array([EQW_list[:, 1][ind]])
    EQW_list= np.concatenate((lmb.T,EQW.T),axis=1)
    return EQW_list,ind

def multipleto(lmb,loggf,EW,EP,EPint): #EPint=[EPmin,EPmax]
    lmb=lmb[(EP>=EPint[0])&(EP<=EPint[1])]
    loggf=loggf[(EP>=EPint[0])&(EP<=EPint[1])]
    print(EW.size,EP.size)
    EW=EW[(EP>=EPint[0])&(EP<=EPint[1])]
    EP=EP[(EP>=EPint[0])&(EP<=EPint[1])]
    return lmb,loggf,EW,EP

def recta(x,m,b):
    y = m*x+b
    return y

def ajuste_mtp(lmb0,loggf,EW,EP,EP_range):
    lmb,loggf,EW,EP=multipleto(lmb0,loggf,EW,EP,EP_range)
    loggfx,Y=loggf+np.log10(lmb), np.log10(EW[:,1]/lmb)
    params,pcov=spo.curve_fit(recta,loggfx,Y)
    m,b=params[0],params[1]
    return m,b,loggfx,EP,Y

def temp(lmb0,loggf,EW,EP,EP_range):
    m1,b1,loggfx1,EP1,y1=ajuste_mtp(lmb0,loggf,EW,EP,EP_range[0])
    m2,b2,loggfx2,EP2,y2=ajuste_mtp(lmb0,loggf,EW,EP,EP_range[1])
    ajuste1, ajuste2= m1*loggfx1+b1, m2*loggfx2+b2
    ymin=max(np.min(ajuste1), np.min(ajuste2))
    ymax=min(np.max(ajuste1), np.max(ajuste2))
    d=np.absolute(np.mean((np.array([ymin,ymax])*(m2-m1)+m1*b2-b1*m2) / m1 / m2))
    #d=(((ymax-ymin)/2-b2)/m2-((ymax-ymin)/2-b1)/m1)
    T=np.absolute(5040*(np.mean(EP1)-np.mean(EP2))/d)
    plt.plot(loggfx1,ajuste1,'-',label="linear fit 1")
    plt.plot(loggfx1,y1 ,'.',label="multiplet 1")
    plt.plot(loggfx2,ajuste2,'-', label="linear fit 2")
    plt.plot(loggfx2,y2,'.',label="multiplet 2")
    plt.ylabel("$log(W_\lambda / \lambda)$",fontsize=15)
    plt.xlabel("$log(gf \lambda)$",fontsize=15)
    plt.ylim(-2.5,-1.7)
    plt.legend(fontsize=13)
    plt.show()
    return T

def get_temp(stardata,lmb0,loggf,EP,EP_range,delta=.2):
    lmb, flux=observado(stardata)
    lmb0 = lmb0[(lmb0 > lmb[0]) & (lmb0 < lmb[-1])]
    EWQ_star,ind=EW_list(lmb,flux,lmb0,delta)
    EP,lmb0,loggf=EP[ind],lmb0[ind],loggf[ind]
    T=temp(lmb0,loggf,EWQ_star,EP,EP_range)
    return T
 

def limdata_star(lmb0,EP,loggf,starwave):
    lmb0_new = lmb0[(lmb0 >= starwave[0]) & (lmb0 <= starwave[-1])]
    EP_new = EP[(lmb0 >= starwave[0]) & (lmb0 <= starwave[-1])]
    loggf_new = loggf[(lmb0 >= starwave[0]) & (lmb0 <= starwave[-1])]
    return lmb0_new,EP_new,loggf_new

def get_fft_min(line, wl):
    c = 2.99e3
    min_freqs = np.zeros(3)
    sun_limbo = np.array([.66, 1.162, 1.661]) 
    j = 0
    line /= np.max(line)
    ext_line = np.concatenate((line, np.zeros(10000)))
    f_line = np.fft.fft(ext_line)
    f_freq = np.fft.fftfreq(ext_line.size)
    f_line = np.absolute(f_line)
    inds = np.where(f_freq > 1e-6)

    f_freq, f_line = f_freq[inds], f_line[inds]
    f_line /= np.max(f_line)
    for i in range(1,f_line.size-1):
        if j < min_freqs.size and f_line[i-1] > f_line[i] and f_line[i+1] > f_line[i]:
            min_freqs[j] = f_freq[i]
            j += 1
    vsini = sun_limbo * c / min_freqs / wl
    return np.mean(vsini[1:]), min_freqs


def vsinI(obs_data, wls, k = .2):
    vs = np.zeros(wls.size)
    minima = np.zeros((wls.size, 3))
    for i, wl in enumerate(wls):
        line = limit_spec(obs_data, wl-k, wl+k)
        vs[i], minima[i, :] = get_fft_min(line, wl)
    minima = np.sum(minima, axis= 0) / wls.size
    return np.mean(vs), minima

######tests######

col_names = ["lambda", "EP", "loggf", "Ei","EW"]
data=pd.read_csv("line_list_tsantaki.dat", names=col_names,sep="\t")
data=data[:120]
lmb0,loggf,EP,EW_sol=np.array(data['lambda'],dtype=float),np.array(data['loggf']),np.array(data['EP']),np.array(data['EW'])
star1_data="estrela1.fits"
star2_data="estrela2_vcor.fits"
datalmb_sint="GES_UVESRed580_Lambda.fits"

star1wave,star1flux=observado(star1_data)
star2wave,star2flux=observado(star2_data)


lmb0_s1,EP_s1,loggf_s1=limdata_star(lmb0,EP,loggf,star1wave)
lmb0_s2,EP_s2,loggf_s2=limdata_star(lmb0,EP,loggf,star2wave)

EP_range1=([[2.1,2.3],[4.2,4.3]])
EP_range2=([[2.1,2.3],[4.6,4.7]])

EQW_sol=np.concatenate((((np.array([lmb0])).T,(np.array([EW_sol])).T)),axis=1)

#T_sol=temp(lmb0,loggf,EQW_sol,EP,EP_range1)
#print("T sol =",T_sol,"K")

#T_star1=get_temp(star1_data,lmb0_s1,loggf_s1,EP_s1,EP_range2,.19)
#print("T star 1 =", T_star1, "K")
T_star2= get_temp(star2_data,lmb0_s2,loggf_s2,EP_s2,EP_range2,.18)
print("T star 2 =",T_star2, "K")
#temp(star2wave,loggf_s2,EW_star2,EP_s2,EP_range)

e1_lmb,e1_flux=observado(star1_data)
e2_lmb,e2_flux=observado(star2_data)
EW_star1,ind1=EW_list(e1_lmb,e1_flux,lmb0_s1,.18)
EW_star2,ind2=EW_list(e2_lmb,e2_flux,lmb0_s2,.18)



sint_file="p6000_g+5.0_m0.0_t01_z+0.25_a+0.00.GES4750.fits" #estrela1

sint2_file="p5750_g+5.0_m0.0_t01_z+0.25_a+0.00.GES4750.fits" #estrela2

#multipletos -- mudar oara 
#m1,b1,loggx1,EP1,y1=ajuste_mtp(lmb0_s2,loggf_s2,EW_star2,EP_s2, EP_rang#e1[0])
#m2,b2,loggx2,EP2,y2=ajuste_mtp(lmb0_s2,loggf_s2,EW_star2,EP_s2,EP_range1[1])
#ajuste1, ajuste2= m1*loggx1+b1, m2*loggx2+b2
#
#plt.plot(loggx1,ajuste1,'-',label="linear fit 1")
#plt.plot(loggx1,y1 ,'.',label="multiplet 1")
#plt.plot(loggx2,ajuste2,'-', label="linear fit 2")
#plt.plot(loggx2,y2,'.',label="multiplet 2")
#plt.ylabel("$log(W_\lambda / \lambda)$",fontsize=15)
#plt.xlabel("$log(gf \lambda)$",fontsize=15)
#plt.ylim(-2.7,-1.7)
#plt.legend(fontsize=13)
#plt.show()



#for i in range(100 ,120):
#   eqw1, y_1, cutwave1, cutflux1=eq_width_test(e1_lmb,e1_flux,lmb0_s1[i],.22)
#   dx=cutwave1[1]-cutwave1[0]
#   tflux,twave=sp.fftpack.fft(cutflux1-1,),sp.fftpack.fftfreq(cutflux1.size,d=dx)
#   plt.plot(twave,abs(tflux),'.')
#   plt.xlim(-10,10)
#plt.show()


#eqw1, y_1, cutwave1, cutflux1=eq_width_test(e1_lmb,e1_flux,lmb0_s1[100],.25)
#dx=cutwave1[1]-cutwave1[0]
#tflux,twave=sp.fftpack.fft(cutflux1-1,),sp.fftpack.fftfreq(cutflux1.size,d=dx)
#plt.plot(twave,abs(tflux))
##plt.xlim(-2,2)
##plt.plot()
#plt.show()
