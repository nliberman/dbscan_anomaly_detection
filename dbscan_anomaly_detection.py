import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def detect_anomalies(data, window=4, tolerance_multiple=3, tolerance_threshold=False):
    anomaly_list = [0]
    try:
        data = list(data)
    except:
        pass
        
    if type(data) != list:
        raise Exception("data must be a one dimensional list or array")
    if (window<2) | (type(window)!=int):
        raise Exception("window must be an integer greater than 1")
    if (tolerance_multiple <= 0) | (type(tolerance_multiple) not in [int, float]):
        raise Exception("tolerance_multiple must be an integer or float greater than 0")
    if type(tolerance_threshold) != bool:
        raise Exception("tolerance_threshold requires a boolean")
        
    if len(data) > 10:
        trim = int(np.ceil(len(data)*.01))
        sd = np.std(sorted(data)[trim:-trim])
    else:
        sd = np.std(data)
        
    eps = sd*tolerance_multiple
    if tolerance_threshold:
        five_pct = np.mean(data) * 0.05
        if five_pct > eps:
            eps = five_pct
    print("tolerance_multiple has yielded a tolerance of " + str(eps))
        
    current_window = 1
    max_window = window - 1
    counter = 0
    list_len = len(data)

    while True:
        ## if last observation was an anomaly
        if anomaly_list[-1] == 1:
            current_window = 1
            ## check the clustering of the t and t+1 val, 
            subset = [[data[counter]], [data[counter+1]]]
            anom_value = run_dbscan(subset, eps)
            ## check the clustering of the t-1 and t+1 val
            subset = [[data[counter-1]], [data[counter+1]]]
            anom_value2 = run_dbscan(subset, eps)
            if (anom_value == 0) | (anom_value2 == 0):
                anomaly_list.append(0)
                if anom_value == 0:
                    current_window += 1
                elif anom_value2 == 0:
                    counter += 1
                    current_window = 1
            else:
                anomaly_list.append(1)
                counter += 1
                current_window = 1

        ## last observation was not an anomaly
        else:
            subset = data[counter:counter+current_window+1]
            subset = [[element] for element in subset]
            anom_value = run_dbscan(subset, eps)
            anomaly_list.append(anom_value)
            if anom_value == 1:
                counter = counter + current_window
                current_window = 1
            else:
                if current_window < max_window:
                    current_window += 1
                else:
                    counter+=1

        if counter+current_window >= list_len:
            break
            
    indices = []
    for enum, i in enumerate(anomaly_list):
        if i == 1:
            indices.append(enum)
        
    print("anomalous indices: " + str(indices))
    return indices

def run_dbscan(subset, eps):
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='euclidean')
    y_pred = dbscan.fit_predict(subset)
    if len(set(y_pred)) > 1:
        return 1
    else:
        return 0
