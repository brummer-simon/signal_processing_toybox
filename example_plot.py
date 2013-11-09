from sigtoybox import * # Import SignalToybox
"""
   Simple Dataplot, Frequency analysis and Autocorrelation.
   Author: Simon Brummer
   Mail:   brummer.simon@googlemail.com
"""

# Plot Grid Layout
nCols = 1 # Number of Columns 
nRows = 3 # Number of Rows

# Read Data
dataFile = 'Simon_HeartBeat.dat'         # Datasource
signal = readDatafile( dataFile )        # Read Data from source
sample = cutOutSamples( signal, 10, 20 ) # Get Datasample between 10 and 20 Secounds 

time   = sample[0]
values = sample[1]

# Plot in Time Domain
xl = 'Time(s)'
yl = 'Magnitude'
tt = 'Dataset: "' + dataFile + '"'

addSubplot( nRows, nCols, 1 )
plotgraph( time, values, xl, yl, tt )

# Plot in Frequency Domain
xl = 'Frequency(Hz)'
tt = 'Frequency Analysis, Dataset: "' + dataFile + '"'

freq, mag = freqDomainOf( values, 100) # FFT Transform with 100Hz Sample Frequency 

addSubplot( nRows, nCols, 2)
plotgraph( freq, mag, xl, yl, tt) 

#Plot Autocorrelation
xl = 'Lags'
yl = 'Correlation'
tt = 'Autocorrelation, Dataset: "' + dataFile + '"'
numberOfLags = 100

lags, corr = autoCorrelationOf( values, numberOfLags )
addSubplot( nRows, nCols, 3 )
plotgraph( lags, corr, xl, yl, tt )

# Finally show Plot
showPlot()
