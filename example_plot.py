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
dataFile = 'Simon_HeartBeat.dat'        
signal = read_datafile( path=dataFile, seperator='\t' )             # Read Data from source
sample = cut_samples_out( dataSet=signal, fromValue=10, toValue=20 ) # Cut out Datasamples 

time   = sample[0]
values = sample[1]

# Plot in Time Domain
xl = 'Time(s)'
yl = 'Magnitude'
tt = 'Dataset: "' + dataFile + '"'

add_subplot( gridRows=nRows, gridCols=nCols, plotNo=1 )
plot_graph( x=time, y=values, xlabel=xl, ylabel=yl, title=tt )

# Plot in Frequency Domain
xl = 'Frequency(Hz)'
tt = 'Frequency Analysis, Dataset: "' + dataFile + '"'

freq, mag = frequency_domain( values=values, sampleFreq=100) # FFT Transform with 100Hz Sample Frequency 

add_subplot( gridRows=nRows, gridCols=nCols, plotNo=2)
plot_graph( x=freq, y=mag, xlabel=xl, ylabel=yl, title=tt) 

#Plot Autocorrelation
xl = 'Lags'
yl = 'Correlation'
tt = 'Autocorrelation, Dataset: "' + dataFile + '"'
numberOfLags = 100

lags, corr = auto_correlation( values=values, lags=numberOfLags )
add_subplot( gridRows=nRows, gridCols=nCols, plotNo=3 )
plot_graph( x=lags, y=corr, xlabel=xl, ylabel=yl, title=tt )

# Finally show Plot
show_plot()
