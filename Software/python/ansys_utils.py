import os
import re
import sys
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

def read_transient_data(case_dir):
    """
    Read Ansys transient export data
    """

    path = os.path.join(sys.path[0], "../..", "Daten", "transient", case_dir)
    
    files = os.listdir(path)

    data = {}

    for file in files:
        
        m = re.findall(r'^.*\-(.*)\.', file)

        if m:
            timestamp =m[0]

        tmp_data = pd.read_csv(os.path.join(path,file))

        data[timestamp] = tmp_data

    return data
