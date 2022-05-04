import os
import re
import sys
from numpy import NaN
import pandas as pd

def read_steady_data(file_name):
    """
    Read Ansys CSV export file.
    """

    path = os.path.join(sys.path[0], ".." , ".." , "Daten", "steady", file_name)

    result = {}
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            m = re.match('^\[(.*)\]', line)
            if m:
                tag = m.group(1)
                continue
            if not tag in result.keys():
                result[tag] = line.split(', ')
                continue
            if tag == 'Data':
                for i, val in enumerate(line.split(', ')):
                    key = result[tag][i]
                    if not key in result.keys():
                        result[key] = []
                    result[key].append(round(float(val), 6))

    return result

def get_case_nums(case_dir):
    path = os.path.join(sys.path[0], "../..", "Daten", "transient", case_dir)
    
    files = os.listdir(path)

    return len(files)
    

def read_transient_data(case_dir, times):
    """
    Read Ansys transient export data

    ansys_filename: C:/Users/TPG247/Documents/Daniel Fischer/thesis-diplom-daniel/Daten/transient/tmp/FFF.1-Setup-Output
    """

    path = os.path.join(sys.path[0], "../..", "Daten", "transient", case_dir)
    
    files = os.listdir(path)

    data = {}

    for file in files:
        
        m = re.findall(r'\d+', file)

        if m:
            timestamp =int(m[0])

        if timestamp in times:
            
            drops = ['nodenumber']
            tmp_data = pd.read_csv(os.path.join(path,file), delimiter=",")
            tmp_data.columns = [x.strip() for x in tmp_data.columns]
            
            for id, col in enumerate(tmp_data.columns):
                m = re.findall(r'\d+', col)

                if m != []:
                    drops.append(col)

            tmp_data.drop(drops, axis=1, inplace=True)

            tmp_data.dropna(subset=['x-coordinate','y-coordinate'], inplace=True)
            
            if tmp_data.empty == False:
                # round data
                tmp_data = tmp_data.round(6)
                data[timestamp] = tmp_data

    return data
