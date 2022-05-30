import os
import re
import sys
import logging
from numpy import NaN
import pandas as pd
import json

def read_steady_data(case, file_name):
    """
    Read Ansys CSV export file.
    """

    path = os.path.join(sys.path[0], ".." , ".." , "Daten", "steady", case, file_name)

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

def get_case_info(cases_dir_path, case):
    """
    Function that reads information for one case from cases.json
    """
    path = os.path.join(sys.path[0], *cases_dir_path, "cases.json")
    with open(path) as f:
        cfg = json.load(f)

    if case in list(cfg.keys()):

        return cfg[case]
    else:
        logging.error("No config for case {} found in config.json. Please add the case to the config file".format(case))
        exit()

def get_colsest_plots(plots, timestep, cases_dir_path, case_dir):
    """
    Funtion that returns the closest dataset to a given timestep
    """
    cases = get_cases(cases_dir_path, case_dir)
    cases.sort()
    logging.debug("Cases = {}".format(cases))
    for id, ele in enumerate(plots):

        if ele < 0:
            case_id = int(ele * timestep)
            case = cases[case_id]
        else:
            case = min(cases, key=lambda x:abs(x-ele))
        plots[id] = case
    return plots

def get_cases(cases_dir_path, case_dir, auto_add=False):
    """
    Function that returns all cases from a case_dir
    """
    path = os.path.join(sys.path[0], *cases_dir_path, case_dir)
    
    # default indices

    if os.path.exists(path) == False:
        if auto_add == False:
            logging.warning("Case {} doesn't exsist. Please add it by running add_data.py".format(case_dir))
            exit()
        else:
            logging.warning("Case {} doesn't exsist. Adding case ...".format(case_dir))
            os.mkdir(path)
            logging.info(f"Case {case_dir} added")


    files = os.listdir(path)
    cases = []

    for id, file in enumerate(files):
        m = re.findall(r'\d+', file)
        if m:
            cases.append(float(m[0]))
    return cases

def get_default_cases(cases_dir_path, case_dir):
    """
    Function that creates default cases to plot if no plots have been set
    """

    cases = get_cases(cases_dir_path, case_dir)

    if cases == []:
        logging.error("No data to process for case {}".format(case_dir))
        exit()
    
    middle = round((max(cases)- min(cases))/2,0)
    if not middle in cases:
        middle = min(cases, key=lambda x:abs(x-middle))

    default_cases = [min(cases), middle , max(cases)]
        

    return default_cases

def get_case_vars(cases_dir_path, case_dir):
    """
    Function that gets all resulting variables for a case
    """
    cases = get_cases(cases_dir_path, case_dir)

    data = read_transient_data(cases_dir_path, case_dir, [cases[0]])

    vars = list(data[cases[0]].columns)

    return vars


def build_journal(cases_dir_path, exit=False):
    """
    Function that builds a journal file to run multiple cases in series
    """
    
    path = os.path.join(sys.path[0], *cases_dir_path, "cases.json")
    with open(path) as f:
        cfg = json.load(f)

    cfg.pop("tmp")

    template_path = os.path.join(sys.path[0], "..", "ansys", "journals", "gui_template.jou")
    logging.debug(f"Template path: {template_path}")
    with open(template_path) as f:
        template = f.readlines()
    logging.debug(f"Template = {template[0:3]}")

    tmp_file = []

    for key, val in cfg.items():

        case_geo_path = os.path.join(sys.path[0], "..", "ansys", "cases", val["case"])
        val["case_path"] = case_geo_path

        export_path = os.path.join(sys.path[0], *cases_dir_path, key, "FFF.1-Setup-Output")
        val["export_path"] = export_path

        cases = get_cases(cases_dir_path, key, True)

        if cases == []:

            case_str = ";--------------Next Case = {}--------------------".format(key)
            tmp_file = tmp_file + ['\n', '\n', case_str,'\n', '\n']

            for line in template:
                m = re.findall(r'\%(.*)\%', line)
                if m != []:
                    ele = m[0]
                    if ele in val:
                        line = line.replace(f"%{ele}%", str(val[ele]))
                        logging.debug(f"line = {line}")
                    else:
                        logging.error(f"Element {ele} not defined in cases.json for element {key}.")
                        exit()
                
                tmp_file.append(line)
            

        else:
            logging.info(f"Already calculated data for case {key}")
            continue
    
    if exit:
        tmp_file = tmp_file + [";Exiting Fluent \n", "/exit ok \n"]

    journal_path = os.path.join(sys.path[0], "..", "ansys", "journals", "journal.jou")
    with open(journal_path, "w") as f:
        f.writelines(tmp_file)
    logging.info(f"Wrote journal to file {journal_path}")

    


def read_transient_data(cases_dir_path ,case_dir, times):
    """
    Read Ansys transient export data

    ansys_filename: C:/Users/TPG247/Documents/Daniel Fischer/thesis-diplom-daniel/Daten/transient/tmp/FFF.1-Setup-Output
    """

    path = os.path.join(sys.path[0], *cases_dir_path, case_dir)
    
    files = os.listdir(path)

    if files == []:
        logging.error("No data to process for case {}".format(case_dir))
        exit()
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

def check_data_format():

        """
        Function that checks that all needed config parameters are set
        """

        base_path = sys.path[0]
        path = os.path.join(base_path, "../..", "Daten")
        modes = os.listdir(path)
        logging.debug("Modi = {}".format(modes))

        for mode in modes:
            cases = os.listdir(os.path.join(path, mode))
            if "cases.json" in cases:
                cases.remove("cases.json")
            logging.debug("Found {} for mode {}".format(cases, mode))

            for case in cases:

                if re.findall(r".csv", case) != []:
                    continue

                timestamps = os.listdir(os.path.join(path, mode, case))
                logging.debug("Found {} files for case {} in mode {}".format(len(timestamps), case, mode))
                renamed = 0
                for file in timestamps:
                    if re.findall(r'.csv', file) == []:
                        old_path = os.path.join(path, mode, case, file)
                        new_path = os.path.join(path, mode, case, file + ".csv")
                        logging.debug(f"Rename {file} to {file + '.csv'}")
                        renamed += 1
                        os.rename(old_path, new_path)
                if renamed != 0:
                    logging.info("Renamed {} files in case {} in mode {}".format(renamed, case, mode))