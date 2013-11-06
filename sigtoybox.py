"""
    Signal Analysis Toybox. This is a collection of often used
    Signal Analysis functions. 
    Author: Simon Brummer
    Mail:   brummer.simon@googlemail.com
"""
import scipy as sp
import pylab as pl
import numpy as np

def readDatafile( path, seperator='\t' ):
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
                data.append( [] )             # Add Dimension if append breaks
                data[i].append( float(values[i]) )   # Append the Value again 

    dataFile.close()

    return data

# Todo
def cutOutSamples( dataSet, fromValue, toValue, normalize=False ):
    """
       Cut out Values of a given dataset beginning from "fromValue" to "toValue". 
       This function assumes that the first dimension of the dataset is orderd ascending and 
       reflects the x-Axis values.
    """


    return None;
    


def freqDomainOf(signal, sampleFreq, singleSided=True):
    """
    Transformation from Time to Frequency Domain. Plotready ;)
    """
    sig = np.float_( signal )  # Convert Value Array to numpy type

    n = len(sig)               # Number of Samples
    k = sp.arange(n)           # List with Series from [0 : n-1]
    t = n / sampleFreq         # Samples / Samplefrequency

    freq = k/t                 # Frequency = [0 : n-1] / Sample / Samplefrequency
    y = sp.fft( sig )/n        # Fast Furier Transform of Signal / Number of Sampels

    if singleSided :                       
        freq = freq[ sp.arange( int(n/2) ) ]   # Cutoff freqlist at n/2
        y = y[ sp.arange( int(n/2) ) ]         # Cutoff manigutelist at n/2
    
    return freq , abs(y)                    # Return freq, abs(y) plotready !



def meanValueOf( values ):
    """
    Calculation of Mean Value of a list of values
    """
    return sum( values ) / len( values )



def meanSquareValueOf( values ):
    """
    Calculation of Mean Square Value of a list of values
    """
    return sum( [ i**2 for i in values] ) / len( values )



def varianceOf( values, sample=False ):
    """
    Calculation of the variance of a list of values
    """
    mean_val = meanValueOf( values )
    n_val = len( values ) -1 if sample else len( values )
    return sum( [ j**2 for j in [ i - mean_val for i in values ] ] ) / n_val



def standardDeviationOf( values, sample=False ):
    """
    Calculation of the Standard Deviation of a list of values
    """
    return sqrt( varianceOf( values, sample ) )


# Todo
def plotHistogramOf(values, nBins=100 ):
    """
    Calculates histogram for nBins bins( Default 100).
    """
    return None	

#Todo
def psdFunction():
    """
    Add DocString faule sau!
    """
    return None

def autoCorrelationOf(values, lags=100):
    """
    Calculate auto correlation a given List of values
    """
    lags, corr, line, x = pl.acorr( values, maxlags=lags, usevlines=False, marker=None)
    return lags, corr



def crossCorrelationOf(values1, values2, lags=100):
    """
    Caluculate cross correlation of a given list of values
    """
    lags, corr, line, x = pl.xcorr( values1, values2, maxlags=lags, usevlines=False, marker=None)
    return lags, corr



def addSubplot(plotNo):
    """
    Add a Subplot. Configure Layout via the global variables
    """
    pl.subplot(plot_row, plot_col, plotNo)



def plotgraph(x, y, xlabel='', ylabel='', title='', xScaleLog=False, yScaleLog=False, color='blue'):
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
    


def showPlot():
    pl.show()



# Configure Rows and Columns is Subplot is used
plot_col = 1
plot_row = 3

if __name__ == '__main__':   

    dataFile = 'Simon_Lead_1.dat'
    sampleFreq = 100 # Samplefrequency in Hz

    # Read Signal
    sig = readDatafile( dataFile )
    t, val = sig[0], sig[1]

    # Plot Signal
    addSubplot(1) 
    plotgraph(t,val, xlabel='Time in s', ylabel = 'Magnitude', title='Dataset "' + dataFile + '"')

    # Plot Frequency Spectrum
    addSubplot(2)
    f,m = freqDomainOf( val, sampleFreq)
    plotgraph(f,m, 'Frequency in Hz', 'Magnitude', 'Frequency Spectrum of Dataset "' + dataFile + '"')

    # Plot Autocorrelation
    addSubplot(3)
    lag,corr = autoCorrelationOf( val )
    plotgraph( lag, corr, xlabel='Lags', ylabel='Correlation', title='Autocorrelation of "' + dataFile + '"' )

    # Show final Plot
    showPlot()
