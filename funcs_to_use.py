#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import smoothing_spline as sm
import matplotlib.pyplot as plt
from scipy import signal
import Paper2_alg as ppa
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import anderson
#functions


def slow_fast(x):
    return np.sin(2*np.pi*x/10) + 0.3*np.sin(2*np.pi*x/0.8)

def transient(x):
    y = 1.2*np.exp(-x/5)
    y+= 0.5*np.exp(-x/10)*np.sin(2*np.pi*x/2)
    return y/np.max(y)

def chirp(x):
    f0, f1 = 0.1, 2
    k = (f1-f0)/(x[-1]-x[0])
    phase = 2*np.pi*(f0*x+0.5*k*x**2)
    return np.sin(phase)

def smooth_step(x, width = 0.2):
    return 0.5*(1+np.tanh(x/width))
    
def periodic_log(x):
    T = 10
    y = np.log(0.5 + (x % T))
    return y/np.max(y)+ 0.3*np.sin(2*np.pi*x/0.8)
    
def event(x):
    y= 0.05*x + smooth_step(x-8, width = 0.25)+0.15*np.sin(2*np.pi*x/6)
    return y/np.max(y)

#metrics

def build_spline(x, y, degree, x_spline):
    par = {'m':int((degree+1)/2), 'p':'auto', 'con':(), 'ineq_con':(), 'plot':False}
    spf = sm.curvefit.spline_fit(x, y, par) 
    y_spline = np.array([sm.bsplines.spline_calc(a, spf) for a in x_spline]) 
    return y_spline
def build_spline_mine(x, y, degree):
    if degree == 1:
        a, b = ppa.smooth_linear(x, y)
        return ppa.spline(x, a, b, x)
    elif degree ==3:
        a, b, c, d, jsjs, shshs = ppa.smooth(x, y)
        return ppa.spline(x, a,b,c,d,x)
def residual(y_smoothed, y_orig):
   r = -y_orig + y_smoothed
   return np.asarray(r)

def infer_fs(x):
    dt = np.mean(np.diff(x))
    return 1/dt
    
def welch_psd(x, y, fs=None, npersegs = None):
    if fs==None:
        fs = infer_fs(x)
    if npersegs == None:
        npersegs =  int(len(y)/2)
    sample_freqs, power_spect_dens = signal.welch(y, fs, nperseg=npersegs)
    return sample_freqs, power_spect_dens

def spectral_flatness(freqs):
    freqs = np.maximum(freqs, 1e-30)
    gmean = np.exp(np.mean(np.log(freqs)))
    amean = np.mean(freqs)
    return gmean/amean

def acf_border(r, trsld = 1e-2):
    acfs = acf(r, nlags = min(len(r), 300))
    for i, a in enumerate(acfs):
        if a>0.2 and i>70:
            break
    return i

def ljung_box(r, lag=None):
    Q = acorr_ljungbox(r, lags = lag)
    return Q

def max_freq(psd, freqs):
    idx = np.argmax(psd)
    f_dom = freqs[idx]
    t_dom = 1/f_dom
    psd_val = psd[idx]
    return f_dom, t_dom, psd_val


def durbin_wats(r):
    return durbin_watson(r)
    
def anderson_test(r):
    return anderson(r, dist ='norm')
#plotting

def plotting_acf(r,name, clr, lags = None, alpha=0.05, conf_int=False):
    if lags == None:
       lags = np.arange(0, min(300, len(r)-1))
    plt.figure(figsize=(14.7, 11.5))
    plot_acf(r, alpha = alpha, lags = lags, use_vlines = True, bartlett_confint=conf_int,
             auto_ylims = True,color=clr ,markerfacecolor=clr,markeredgecolor=clr, vlines_kwargs={"colors": clr, "linewidth": 2})
    plt.xlabel('lag',fontsize=13)
    plt.ylabel('ACF', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f'ACF of the {name} function', fontsize=18)
    plt.legend(fontsize=13)

def plot_reg(x, y, atitle, color, xlbl, ylbl,  xlims=None, flbl=None, grid = False, separately = False, semilog = None, scatter = None):
    if semilog == 'y':
        plt.semilogy(x, y, c=color, label =flbl)
    elif semilog == 'x':
        plt.semilogx(x, y, c=color, label =flbl)
    elif scatter == True:
         plt.scatter(x, y, c=color, label =flbl)
    else:
        plt.plot(x, y, c=color, label =flbl)
    plt.xlabel(xlbl, fontsize=30)
    plt.ylabel(ylbl, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(atitle, fontsize=35)
    if flbl is not None:
        plt.legend(fontsize=30)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if grid == True:
        plt.grid()
    if separately ==True:
        plt.show()

def qqplotting(r, name, xlims = None):
    qqplot(r, line ='s')
    plt.xlabel('theoreticall',fontsize=13)
    plt.ylabel('sample', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f'QQ plot of the {name} function residual', fontsize=18)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])

def acf_manual(resids, name, cols):
    plt.figure(figsize = (14.7, 11.5))
    N = len(resids[1])  # length of original series
    conf = 1.96 / np.sqrt(N)
    fig, ax = plt.subplots()
    for degree, resid in reversed(resids.items()):
        acf_vals = acf(resid, nlags = min(300, len(resid)-1))
   
        lags = np.arange(len(acf_vals))
    
        # stem-style plot (this is what statsmodels uses conceptually)
        markerline, stemlines, baseline = ax.stem(lags, acf_vals, label=(f"d={degree}"))
        
        # styling to match statsmodels
        plt.setp(stemlines, linewidth=2, color=cols[degree])
        plt.setp(markerline, markersize=5, color=cols[degree])
        plt.setp(baseline, linewidth=2, color = cols[degree])
    # zero line
    ax.axhline(0, linewidth=1, color='black')
    
    # confidence bands
    ax.fill_between(
    lags,
    -conf,
    conf,
    color='lightblue',
    alpha=0.5
)
    ax.legend()
    # limits like statsmodels
    ax.set_ylim(auto=True)
    ax.set_xlim(auto=True)
    
    ax.set_xlabel("lag", fontsize=15)
    ax.set_ylabel("ACF", fontsize = 15)
    ax.set_title(name, fontsize = 20)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=15)