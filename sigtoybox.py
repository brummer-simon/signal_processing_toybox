"""
    Signal Analysis Toybox. This is a collection of often used
    Signal Analysis functions. 
    Author: Simon Brummer
    Mail:   brummer.simon@googlemail.com
"""
import scipy as sp
import pylab as pl
import numpy as np
import math  as ma

def read_datafile( path, seperator='\t' ):
    """ 
    Read n-dimensional Datafile. readDatafile() 
    expects the following filestructure:
    
    'dim 0 + seperator + dim 1 + ... + seperator + dim n + \n'

    Returns 
        n-dimentional List
    """
    data = []
    dataFile = open(path,'r')
    
    for line in dataFile:
        line = line.strip()                  # Strip line of whitespaces
        values = line.split(seperator)       # Split line into seperate values
        
        for i in range( len(values) ):
            try:
                if values[i] != '':
                    data[i].append(float(values[i])) # Try to add values to corresponding dimention
            except IndexError:
                data.append( [] )                    # Add Dimension if append breaks
                data[i].append( float(values[i]) )   # Append the Value again 

    dataFile.close()

    return data

def cut_samples_out( dataSet, fromValue, toValue ):
    """
       Cut out Values of a given dataset beginning from "fromValue" to "toValue". 
       This function assumes that the first dimension of the dataset is orderd ascending and 
       reflects the x-Axis values.
    """
    newData = []
    for data in dataSet:
        newData.append( [] )

    for tup in zip(*dataSet):
        if fromValue <= tup[0] and tup[0] <= toValue:
            for i, val in enumerate( tup ):
                newData[i].append( val )
    
    return newData
    
def calc_line( values, a, c):
    """
    Calculate values which draws a straight line.
    """
    line=[]
    for val in values:
        line.append( a * val + c)
    return line

def frequency_domain(values, sampleFreq, singleSided=True):
    """
    Transformation from Time to Frequency Domain. Plotready ;)
    """
    sig = np.float_( values )  # Convert Value Array to numpy type

    n = len(sig)               # Number of Samples
    k = sp.arange(n)           # List with Series from [0 : n-1]
    t = n / sampleFreq         # Samples / Samplefrequency

    freq = k/t                 # Frequency = [0 : n-1] / Sample / Samplefrequency
    y = sp.fft( sig )/n        # Fast Furier Transform of Signal / Number of Sampels

    if singleSided :                       
        freq = freq[ sp.arange( int(n/2) ) ]   # Cutoff freqlist at n/2
        y = y[ sp.arange( int(n/2) ) ]         # Cutoff manigutelist at n/2
    
    return freq , abs(y)                    # Return freq, abs(y) plotready !

def absolute_values( values ):
    """
    Calculates Absolute Values of a given List
    """
    absVal = []
    for val in values:
        absVal.append( abs(val))

    return absVal

def mean_value( values ):
    """
    Calculation of Mean Value of a list of values
    """
    return sum( values ) / len( values )

def mean_square_value( values ):
    """
    Calculation of Mean Square Value of a list of values
    """
    return sum( [ i**2 for i in values] ) / len( values )

def root_mean_square_value( values ):
    """
    Calculation of Root Mean Square Value of a list of values
    """
    return ma.sqrt(mean_square_value( values ))

def variance( values, sample=False ):
    """
    Calculation of the variance of a list of values
    """
    mean_val = mean_value( values )
    n_val = len( values ) -1 if sample else len( values )
    return sum( [ j**2 for j in [ i - mean_val for i in values ] ] ) / n_val

def standard_deviation( values, sample=False ):
    """
    Calculation of the Standard Deviation of a list of values
    """
    return ma.sqrt( variance( values, sample ) )

def auto_correlation(values, lags=100):
    """
    Calculate auto correlation a given List of values
    """
    lags, corr, line, x = pl.acorr( values, maxlags=lags, usevlines=False, marker=None)
    return lags, corr

def cross_correlation(values1, values2, lags=100):
    """
    Caluculate cross correlation of a given list of values
    """
    lags, corr, line, x = pl.xcorr( values1, values2, maxlags=lags, usevlines=False, marker=None)
    return lags, corr

def rms_smoothing( values, samples=100 ):
    """
    Simple Root Mean Square Smoothing. Returns smoothed curve as a list.
    """
    rms = []
    rng = int(samples/2)  # Sample used for Smoothing
    for i,x in enumerate( values ):        
        lo = i-rng if i-rng > 0 else 0
        hi = i+rng
        rms.append( rootMeanSquareValueOf( values[ lo : hi] ))
    return rms

def add_subplot(gridRows, gridCols, plotNo):
    """
    Add a Subplot. Configure Layout via the global variables
    """
    pl.subplot(gridRows, gridCols, plotNo)

def plot_graph(x, y, xlabel='', ylabel='', title='', xScaleLog=False, yScaleLog=False, color='blue'):
    """
    Adds a Plot to the final figure. 
    """
    if xScaleLog and yScaleLog :
        pl.loglog( x, y)
    elif xScaleLog :
        pl.semilogx( x, y)
    
    elif yScaleLog :
        pl.semilogy( x, y)    
    else:
        pl.plot(x, y, color=color)
    
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title(title)
    
def show_plot():
    """
    Shows Plot
    """
    pl.show()

