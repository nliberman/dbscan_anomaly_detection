# dbscan_anomaly_detection

## Dependency Requirements
    numpy
    sklearn

## Usage
    from dbscan_anomaly_detection import detect_anomalies
    
    ## returns a list of index positions deemed to be anomalies
    anomalies = detect_anomalies(data, window, tolerance_multiple, tolerance_threshold)
    
## Documentation
    data: a one dimensional list of numeric values that must be passed by user
        
    window: (default=4) int greater than 1 representing the window size for grouping sequential elements in the data
        
    tolerance_multipe: (default=3) a multiple greater than zero representing the number of standard deviations away 
                         from the (outlier adjusted) mean to use as the tolerance parameter (epsilon)
                            
    tolerance_threshold: (default=False) boolean, representing whether or not to use 5% of the outlier adjusted mean 
                         as a minimum value of the tolerance parameter (epsilon)
