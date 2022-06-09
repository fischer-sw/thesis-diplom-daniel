import os
import re
import sys
import glob
import time
import logging
from numpy import NaN
import pandas as pd
import json
import psutil

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
    case_path = os.path.join(*cases_dir_path, case_dir)
    
    # default indices

    if os.path.exists(case_path) == False:
        if auto_add == False:
            logging.warning("Case {} doesn't exsist. Please add it by running add_data.py".format(case_dir))
            logging.warning(f"Following cases exsist: {cases}")
            exit()
        else:
            logging.warning("Case {} doesn't exsist. Adding case ...".format(case_dir))
            os.mkdir(case_path)
            logging.info(f"Case {case_dir} added")


    files = glob.glob(r'*Output.csv', root_dir=case_path)
    cases = []

    for id, file in enumerate(files):
        m = re.findall(r'\d+', file)
        if m:
            cases.append(float(m[0]))
    return cases

def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;

def run_fluent(dims='2ddp', p_num=6, jou_name=""):
    
    logs = []

    if jou_name == "":
        logging.error("No journal name provided")
        exit()

    if jou_name == "all_cases":
        jou_path = os.path.join(sys.path[0], "..", "ansys", "journals", jou_name + ".jou")
    else:
        jou_path = os.path.join(sys.path[0], "..", "ansys", "journals", "cases", jou_name + ".jou")


    if os.path.exists(jou_path) == False:
        logging.warning(f"journal {jou_name} doesn't exsist. Continuing with next one.")
        return

    # cmd = " ".join(['fluent', dims, f'-t{p_num}', f'-i{jou_path}', '> ' + os.path.join(sys.path[0], 'test.log'), '2>&1'])
    cmd = " ".join(['fluent', dims, f'-t{p_num}', f'-i{jou_path}'])
    logging.info(f"Running case {jou_name}")
    os.system(cmd)


    # os.remove("test.log")
    time.sleep(20)
    files = glob.glob('*.trn', root_dir=os.path.join(sys.path[0], "..", ".."), recursive=False)
    if files == []:
        files = glob.glob('**/*.trn', root_dir=os.path.join(sys.path[0], "..", ".."), recursive=True)
        
    if files == []:
        logging.warning(f"No log files created for journal {jou_name}")
        exit()
        # return logs

    log_path = os.path.join(sys.path[0], "..", "..", files[0])

    processing = True
    while processing:
        time.sleep(10)
        processing = checkIfProcessRunning("fluent")

    logs = parse_logs(log_path)
    logging.info(f"Removing .trn file {log_path}")
    os.remove(log_path)

def get_default_cases(cases_dir_path, case_dir):
    """
    Function that creates default cases to plot if no plots have been set
    """

    cases = get_cases(cases_dir_path, case_dir)

    if cases == []:
        logging.error("No data to process for case {}".format(case_dir))
        exit()
    
    middle = round((max(cases)- min(cases))/2 + min(cases),0)
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

    if "tmp" in cfg.keys():
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
        journal_cases = os.path.join(sys.path[0], "..", "ansys", "journals", "cases")
        journal_path = os.path.join(journal_cases, key + ".jou")

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
                tmp_file = tmp_file + ["\n", "\n", ";Exiting Fluent \n", "/exit ok \n"]

            
            if os.path.exists(journal_cases) == False:
                os.mkdir(journal_cases)
                logging.info(f"Created cases directory for journals {journal_cases}")

            with open(journal_path, "w") as f:
                f.writelines(tmp_file)
            logging.info(f"Wrote journal to file {journal_path}")

    if end_exit:
        tmp_file = tmp_file + ["\n", "\n", ";Exiting Fluent \n", "/exit ok \n"]

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

    with open(path2logfile, "r") as f:
        lines = f.readlines()
    
    case = ""
    case_lines = []

    # find cases and remember lines

    for idx, line in enumerate(lines):
        line = line.strip()
        new_case = re.findall(r'cx-set-text-entry', line)
        if new_case != []:
            case = re.findall(r".*\\(.*)\\FFF", line)[0]
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

                        tmp[key][resid_header[id]].append(float(ele.split(":")[-1]))
    save_residuals(tmp)
    return tmp

def save_residuals(resis):

    path = os.path.join(sys.path[0], "conf.json")
    with open(path) as f:
        cfg = json.load(f)

    cases = list(resis.keys())

    for ele in cases:
        resi_path = os.path.join(*cfg["cases_dir_path"], ele, ele + "_residuals.csv")
        logging.info(f"Saving residuals for case {ele} to {resi_path}")
        tmp = pd.DataFrame(resis[ele])
        tmp.drop_duplicates(keep='first', inplace=True, subset=['iter'])
        tmp.to_csv(resi_path, index=False)
        


def read_transient_data(cases_dir_path ,case_dir, times):
    """
    Read Ansys transient export data

    ansys_filename: C:/Users/TPG247/Documents/Daniel Fischer/thesis-diplom-daniel/Daten/transient/tmp/FFF.1-Setup-Output
    """

    path = os.path.join(*cases_dir_path, case_dir)
    
    files = glob.glob(r'*Output.csv', root_dir=path)

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

        path = os.path.join(*cases_dir_path)
    
        cases = os.listdir(os.path.join(path))

        for case in cases:

            if re.findall(r".csv", case) != []:
                continue

            timestamps = os.listdir(os.path.join(path, case))
            logging.debug("Found {} files for case {}".format(len(timestamps), case))
            renamed = 0
            for file in timestamps:
                if re.findall(r'.csv', file) == []:
                    old_path = os.path.join(path, case, file)
                    new_path = os.path.join(path, case, file + ".csv")
                    logging.debug(f"Rename {file} to {file + '.csv'}")
                    renamed += 1
                    os.rename(old_path, new_path)
            if renamed != 0:
                logging.info("Renamed {} files in case {}".format(renamed, case))

if __name__ == "__main__":

    path = os.path.join(sys.path[0], "fluent-20220608-110207-10944.trn")
    parse_logs(path)