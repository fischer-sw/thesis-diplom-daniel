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

def get_case_info(case):
    """
    Function that reads information for one case from cases.json
    """
    path = os.path.join(sys.path[0], "..", "ansys", "cases.json")
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
    path = os.path.join(*cases_dir_path, case_dir)

    cases = os.listdir(os.path.join(*cases_dir_path))
    
    # default indices

    if os.path.exists(path) == False:
        if auto_add == False:
            logging.warning("Case {} doesn't exsist. Please add it by running add_data.py".format(case_dir))
            logging.warning(f"Following cases exsist: {cases}")
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


def build_journal(cases_dir_path, split_cases, end_exit=False):
    """
    Function that builds a journal file to run multiple cases in series
    """
    
    path = os.path.join(sys.path[0], "..", "ansys", "cases.json")
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

        if split_cases:
            tmp_file = []

        case_geo_path = os.path.join(sys.path[0], "..", "ansys", "cases", val["case"])
        val["case_path"] = case_geo_path

        export_path = os.path.join(*cases_dir_path, key, "FFF.1-Setup-Output")
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
        
        if split_cases:

            if end_exit:
                tmp_file = tmp_file + [";Exiting Fluent \n", "/exit ok \n"]

            journal_cases = os.path.join(sys.path[0], "..", "ansys", "journals", "cases")
            journal_path = os.path.join(journal_cases, key + ".jou")
            if os.path.exists(journal_cases) == False:
                os.mkdir(journal_cases)
                logging.info(f"Created cases directory for journals {journal_cases}")

            with open(journal_path, "w") as f:
                f.writelines(tmp_file)
            logging.info(f"Wrote journal to file {journal_path}")

    if end_exit:
        tmp_file = tmp_file + [";Exiting Fluent \n", "/exit ok \n"]

    if split_cases == False:
        journal_path = os.path.join(sys.path[0], "..", "ansys", "journals", "all_cases.jou")
        with open(journal_path, "w") as f:
            f.writelines(tmp_file)

    logging.info(f"Wrote journal to file {journal_path}")

def parse_logs(path2logfile):
    """
    Function that extracts residuals from terminal log
    """
    tmp = {}

    filterd = []
    raw_data = {}

    with open(path2logfile) as f:
        lines = f.readlines()
    
    case = ""
    case_lines = []

    # find cases and remember lines

    for idx, line in enumerate(lines):
        line = line.strip()
        new_case = re.findall(r'cx-set-text-entry', line)
        if new_case != []:
            case = re.findall(r'transient\\(.*)\\FFF', line)[0]
            tmp[case] = {}
            case_lines.append(idx)

    logging.info(f"cases = {list(tmp.keys())}")

    cases = list(tmp.keys())

    # split raw data for each case

    for idx, ele in enumerate(case_lines):
        if idx != len(case_lines)-1:
            raw_data[cases[idx]] = lines[ele:case_lines[idx+1]]
        else:
            raw_data[cases[idx]] = lines[ele:-1]

    for key, val in raw_data.items():
        for line in val:
            line = line.strip()
            find_resid_header = re.findall(r'iter', line)
            if find_resid_header != []:
                resid_header = re.sub(r"\s+", " ", line)
                resid_header = resid_header.split(" ")

            residuals = re.findall(r'\d+\.\d*', line)
            if residuals != [] and len(residuals) > 4:
                raw_nums = re.sub(r'\s+', " ", line)
                nums = raw_nums.split(" ")
                nums.pop()
                if len(nums) != len(resid_header):
                    logging.warning(f"Number of Header vars {len(resid_header)} and number of datapoints {len(nums)} doesn't match for line {line}")
                else:
                    for id, ele in enumerate(nums):

                        if (resid_header[id] in list(tmp[key].keys())) == False:
                            tmp[key][resid_header[id]] = [] 
                            logging.debug(f"Adding var {resid_header[id]} to case {key}")   

                        tmp[key][resid_header[id]].append(ele)
    return tmp


def read_transient_data(cases_dir_path ,case_dir, times):
    """
    Read Ansys transient export data

    ansys_filename: C:/Users/TPG247/Documents/Daniel Fischer/thesis-diplom-daniel/Daten/transient/tmp/FFF.1-Setup-Output
    """

    path = os.path.join(*cases_dir_path, case_dir)
    
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

def check_data_format(cases_dir_path):

        """
        Function that checks that all needed config parameters are set
        """

        path = os.path.join(*cases_dir_path, "..")
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


if __name__ == "__main__":
    parse_logs("./bla.log")