import os
import re
import sys
import glob
import time
import logging
import pandas as pd
import json
import psutil
import shutil

import numpy as np

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

def get_closest_plots(config, cases_cfg, case, export_times="flow_time"):
    """
    Funtion that returns the closest dataset to a given timestep
    """

    plots = np.array(config["plots"])
    cases = get_cases(config, case)
    cases.sort()
    logging.debug("Cases = {}".format(cases))

    if export_times == "timestep":
        plots = plots / cases_cfg[case]["timestep"]

    for id, ele in enumerate(plots):

        if ele < 0:
            case = cases[-1 * int(-ele)]
        else:
            case = min(cases, key=lambda x:abs(x-ele))
        plots[id] = case
    return list(plots)

def get_cases(config, case_dir, auto_add=False, export_time="flow_time"):
    """
    Function that returns all cases from a case_dir
    """
    data_path = config["data_path"]

    if config["hpc_calculation"]:
        cases_dir_path = config["cases_dir_path"][1:]
        cases_dir_path[0] = "/" + cases_dir_path[0]
    else:
        cases_dir_path = config["cases_dir_path"]


    case_path = os.path.join(*cases_dir_path, case_dir, *data_path)
    
    # default indices
    if os.path.exists(case_path) == False:
        if auto_add == False:
            logging.warning(f"Case {case_dir} doesn't exsist.")
            cases = [x.split(os.sep)[0] for x in glob.glob("*" + os.sep, root_dir=os.path.join(*cases_dir_path))]
            logging.warning(f"Following cases exsist: {cases}")
            exit()
        else:
            logging.warning("Case {} doesn't exsist. Adding case ...".format(case_dir))
            os.makedirs(case_path)
            logging.info(f"Case {case_dir} added")


    files = glob.glob(r'*-Output*', root_dir=case_path)
    cases = []

    for id, file in enumerate(files):
        m = re.findall(r'\d+', file)
        if m:
            if export_time == "flow_time":
                cases.append(float(m[-2] + "." + m[-1]))
            else:
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
    return False

def run_fluent(dims='2ddp', p_num=4, jou_name="", run_gui_case=True):
    """
    Function to run fluent process with journal file
    """

    remove_ansys_logs()
    
    logs = []
    log_path = ""

    if jou_name == "":
        logging.error("No journal name provided")
        exit()

    if jou_name == "all_cases":
        jou_path = os.path.join(sys.path[0], "..", "ansys", "journals", jou_name + ".jou")
    else:
        jou_path = os.path.join(sys.path[0], "..", "ansys", "journals", "cases", jou_name)

    cmd = " ".join(['fluent', dims, f'-t{p_num}', f'-i{jou_path}'])
    logging.info(f"Running case {jou_name}")
    os.system(cmd)
   
    time.sleep(20)
    files = glob.glob('*.trn', root_dir=os.path.join(sys.path[0], "..", ".."), recursive=False)
    if files == []:
        files = glob.glob('**/*.trn', root_dir=os.path.join(sys.path[0], "..", ".."), recursive=True)
        
    if files == []:
        logging.warning(f"No log files created for journal {jou_name}")
    else:
        log_path = os.path.join(sys.path[0], "..", "..", files[0])

    processing = True
    while processing:
        time.sleep(10)
        processing = checkIfProcessRunning("fluent")

    if run_gui_case:
        logs = parse_logs(log_path)

    if log_path != "":
        logging.info(f"Removing .trn file {log_path}")
        os.remove(log_path)

def get_default_cases(config, case_dir):
    """
    Function that creates default cases to plot if no plots have been set
    """

    cases = get_cases(config, case_dir)

    if cases == []:
        logging.error("No data to process for case {}".format(case_dir))
        exit()
    
    middle = round((max(cases)- min(cases))/2 + min(cases),0)
    if not middle in cases:
        middle = min(cases, key=lambda x:abs(x-middle))

    default_cases = [min(cases), middle , max(cases)]
        

    return default_cases

def remove_ansys_logs():

    """
    Method to remove .trn files still remaining from manual testing
    """

    files = glob.glob('*.trn', root_dir=os.path.join(sys.path[0], "..", ".."), recursive=False)
    if files == []:
        files = glob.glob('**/*.trn', root_dir=os.path.join(sys.path[0], "..", ".."), recursive=True)
        
    if files == []:
        logging.warning(f"No log files deleted.")

    for file in files:
        log_path = os.path.join(sys.path[0], "..", "..", file)
        os.remove(log_path)
        logging.info(f"Removed logfile {log_path}")


def get_case_vars(config, case_dir):
    """
    Function that gets all resulting variables for a case
    """
    
    data_path = config["data_path"]
    cases_dir = config["cases_dir_path"]
    hpc_cases_dir = config["cases_dir_path"][1:]
    hpc_cases_dir[0] = "/" + hpc_cases_dir[0]

    if config["hpc_calculation"]:
        case_path = os.path.join(*hpc_cases_dir, case_dir, *data_path)
    else:

        case_path = os.path.join(*cases_dir, case_dir, *data_path)

    files = glob.glob(r'*-Output*', root_dir=case_path)
    if files == []:
        logging.error("No data to process for case {}".format(case_dir))
        exit()
    
    file_path = os.path.join(case_path, files[0])

    with open(file_path) as f:
        header = f.readline()

    elements = header.replace("\n", "").replace(" ", "").split(",")
    elements.remove('nodenumber')

    return elements


def build_journal(config, cases_cfg, end_exit=False, mode="cmd", update_exsisting=False):
    """
    Function that builds a journal file to run multiple cases in series
    """
    cfg = cases_cfg
    match mode:
        case "cmd":
            gui_template_path = os.path.join(sys.path[0], "..", "ansys", "journals", "cmd_creation_template.jou")
            cmd_template_path = os.path.join(sys.path[0], "..", "ansys", "journals", "cmd_template.jou")
            logging.debug(f"CMD Template path: {gui_template_path}")
            with open(cmd_template_path) as f:
                cmd_template = f.readlines()
            
            logging.debug(f"CMD Template = {cmd_template[0:3]}")
            case_template_path = os.path.join(sys.path[0], "..", "hpc", "fluent_template.sh")
            with open(case_template_path) as f:
                case_template = f.readlines()

            logging.debug(f"Case Template = {case_template[0:3]}")
            post_template_path = os.path.join(sys.path[0], "..", "hpc", "post_template.sh")
            with open(post_template_path) as f:
                post_template = f.readlines()
            logging.debug(f"Post Template = {post_template[0:3]}")
        case "gui":
            gui_template_path = os.path.join(sys.path[0], "..", "ansys", "journals", "gui_template.jou")



    logging.debug(f"GUI Template path: {gui_template_path}")
    with open(gui_template_path) as f:
        cmd_creation_template = f.readlines()
    logging.debug(f"GUI Template = {cmd_creation_template[0:3]}")

    for key, val in cfg.items():

        if "." in key or len(key) > 20:
            logging.warning(f"Case name {key} too long or contains character '.'")
            continue

        gui_tmp_file = []
        cmd_tmp_file = []
        case_tmp_file = []
        post_tmp_file = []

        cases = get_cases(config, key, auto_add=True)
        match mode:

            case "cmd":
                case_export_base_path = os.path.join(*config["cases_dir_path"], key)
                cases_dir_path = "/".join([".", *config["data_path"], "FFF.1-Setup-Output.csv"])   
                export_path = cases_dir_path

            case "gui":
                case_export_base_path = os.path.join(sys.path[0], "..", "ansys", "cmd_cases", key)
                cases_dir_path = config["cases_dir_path"]
                export_path = os.path.join(*cases_dir_path, key, *config["data_path"], "FFF.1-Setup-Output.csv")

        cases_dir_windows_path = config["cases_dir_path"]
        
        case_export_path = os.path.join(case_export_base_path, val['case'])

        case_data = glob.glob("*.gz", root_dir=os.path.join(*cases_dir_windows_path, key))

        if case_data != [] and cases == []:
            for file in case_data:
                os.remove(os.path.join(*cases_dir_windows_path, key, file))
            logging.info(f"Removed exsisting .gz files for case {key}")
        
        case_geo_path = os.path.join(sys.path[0], "..", "ansys", "cases", val["case"])
        data_dir = os.path.join(*cases_dir_windows_path, key, *config["data_path"])
        res_dir = os.path.join(*cases_dir_windows_path, key, *config["hpc_results_path"])
        journal_cases = os.path.join(sys.path[0], "..", "ansys", "journals", "cases")

        dirs = [case_export_base_path, case_geo_path, journal_cases, data_dir, res_dir]

        for dir in dirs:
            if os.path.exists(dir) == False:
                os.makedirs(dir)

        val["case_path"] = case_geo_path
        val["export_path"] = export_path
        val["case_export_path"] = case_export_path
        tmp_keys = list(cases_cfg[key].keys())
        if "post_walltime" in tmp_keys:
            val["post_walltime"] = cases_cfg[key]["post_walltime"]
        else:
            val["post_walltime"] = "12:00:00"

        if not "job_name" in val.keys():
            val["job_name"] = key
            val["post_name"] = key + "_post"

        
        
        gui_journal_path = os.path.join(journal_cases, key + ".jou")
        cmd_journal_path = os.path.join(case_export_base_path, "case.jou")
        
        case_str = ";--------------Next Case = {}--------------------".format(key)
        gui_tmp_file = gui_tmp_file + ['\n', '\n', case_str,'\n', '\n']

        # GUI Case
        for line in cmd_creation_template:
            m = re.findall(r'\%(.*?)\%', line)
            if m != []:
                for ele in m:
                    if ele in val:
                        line = line.replace(f"%{ele}%", str(val[ele]))
                        logging.debug(f"line = {line}")
                    else:
                        logging.error(f"Element {ele} not defined in cases.json for element {key}.")
                        exit()
            
            gui_tmp_file.append(line)

        if mode == "cmd":

            # CMD Case
            for line in cmd_template:
                m = re.findall(r'\%(.*?)\%', line)
                if m != []:
                    for ele in m:
                    
                        if ele in val:
                            line = line.replace(f"%{ele}%", str(val[ele]))
                            logging.debug(f"line = {line}")
                        else:
                            logging.error(f"Element {ele} not defined in cases.json for element {key}.")
                            exit()
                
                cmd_tmp_file.append(line)  

            for line in case_template:
                m = re.findall(r'\%(.*?)\%', line)
                if m != []:
                    for ele in m:
                    
                        if ele in val:
                            line = line.replace(f"%{ele}%", str(val[ele]))
                            logging.debug(f"line = {line}")
                        else:
                            logging.error(f"Element {ele} not defined in cases.json for element {key}.")
                            exit()
                
                case_tmp_file.append(line) 

            for line in post_template:
                m = re.findall(r'\%(.*?)\%', line)
                if m != []:
                    for ele in m:
                    
                        if ele in val:
                            line = line.replace(f"%{ele}%", str(val[ele]))
                            logging.debug(f"line = {line}")
                        else:
                            logging.error(f"Element {ele} not defined in cases.json for element {key}.")
                            exit()
                
                post_tmp_file.append(line)

        if end_exit:
            gui_tmp_file = gui_tmp_file + ["\n", "\n", ";Exiting Fluent \n", "/exit ok \n"]
            cmd_tmp_file = cmd_tmp_file + ["\n", "\n", ";Exiting Fluent \n", "/exit ok \n"]

        if cases != []:
            logging.info(f"Already calculated data for case {key}")
            cfg_path = os.path.join(*cases_dir_windows_path, key, "conf.json")
            if update_exsisting:
                add_post_proc(config, cases_cfg, key, post_tmp_file, case_tmp_file)
            continue

        with open(gui_journal_path, "w") as f:
            f.writelines(gui_tmp_file)
        logging.info(f"Wrote journal to file {gui_journal_path}")

        with open(cmd_journal_path, "w") as f:
            f.writelines(cmd_tmp_file)
        logging.info(f"Wrote journal to file {cmd_journal_path}")
        add_post_proc(config, cases_cfg, key, post_tmp_file, case_tmp_file)
        
        

def add_post_proc(config, cases_cfg, cas, post_file, case_file):

    cases_dir_windows_path = config["cases_dir_path"]
    case_path = os.path.join(*cases_dir_windows_path, cas, "case.sh")
    post_path = os.path.join(*cases_dir_windows_path, cas, "post.sh")

    cfg_path = os.path.join(*cases_dir_windows_path, cas, "conf.json")
    tmp_cfg = {}
    for ele in config.keys():
        tmp_cfg[ele] = config[ele]
    tmp_cfg["cases"] = [cas]
    tmp_cfg["hpc_calculation"] = True
    tmp_cfg["create_widths"] = True
    tmp_cfg["create_resi_plot"] = True
    tmp_cfg["create_image"] = True
    tmp_cfg["ignore_exsisting"] = True
    tmp_cfg["c_bar"] = "fluid_c"
    tmp_cfg["image_file_type"] = "png"
    tmp_cfg["plot_file_type"] = "png"
    tmp_cfg["create_plot"] = True
    tmp_cfg["create_gif"] = True
    tmp_cfg["create_front"] = True
    tmp_cfg["field_var"] = ["molef-fluid_c"]
    tmp_cfg["image_conf"] = {'set_custom_range': True, 'min': 0.0, 'max': 1.0}
    tmp_cfg["gif_conf"]["cases"] = {'start': 0, 'end': cases_cfg[cas]["total_time"], 'step': 1}
    tmp_cfg["gif_conf"]["keep_images"] = True
    tmp_cfg["gif_conf"]["videos"] = True
    tmp_cfg["gif_conf"]["gif_plot"] = True
    tmp_cfg["gif_conf"]["gif_image"] = True
    tmp_cfg["gif_conf"]["loop"] = 0
    tmp_cfg["gif_conf"]["frame_duration"] = 300
    tmp_cfg["plot_vars"] = ["molef-fluid_a", "molef-fluid_b", "molef-fluid_c"]

    copy_files = ["ansys_utils.py", "hpc_requirements.txt", "transient_field.py"]

    del_files = copy_files + ["conf.json", "cases.json"]

    logging.info(f"Deleting old config files for case {cas}")
    for file in del_files:
        path_tmp = os.path.join(*cases_dir_windows_path, cas, file)
        if os.path.exists(path_tmp):
            os.remove(path_tmp)

    logging.info(f"Creating conf.json at hpc destination")
    with open(cfg_path, "w") as f:
        json.dump(tmp_cfg, f, ensure_ascii=False, indent=4)

    logging.info(f"Creating cases.json at hpc destination")
    cases_path = os.path.join(*cases_dir_windows_path, cas, "cases.json")
    with open(cases_path, "w") as f:
        json.dump(cases_cfg, f, ensure_ascii=False, indent=4)

    for file in copy_files:
        dest_path = os.path.join(*cases_dir_windows_path, cas, file)
        src_path = os.path.join(sys.path[0], file)         
        shutil.copy(src_path, dest_path)
        logging.info(f"Copied {file} to destination for case {cas}")

    with open(case_path, "w") as f:
        f.writelines(case_file)
    logging.info(f"Wrote case.sh to file {case_path}")

    with open(post_path, "w") as f:
        f.writelines(post_file)
    logging.info(f"Wrote post.sh to file {post_path}")


def parse_logs(path2logfile, journal=True, case=""):
    """
    Function that extracts residuals from terminal log
    """
    tmp = {}

    filterd = []
    raw_data = {}

    with open(path2logfile, "r") as f:
        lines = f.readlines()
    
    if journal == True:

        
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

    else:
        # for idx, ele in enumerate(lines):
        #     if idx != len(lines)-1:
        #         raw_data[cases[idx]] = lines[ele:lines[idx+1]]
        #     else:
        #         raw_data[cases[idx]] = lines[ele:-1]


        tmp[case] = {}
        for tmp_idx, line in enumerate(lines):
            line = line.strip()
            find_resid_header = re.findall(r'iter', line)
            if len(find_resid_header) == 2:
                first_line= tmp_idx
                break

        lines = lines[first_line:]
        for tmp_idx, line in enumerate(lines):
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
                        keys = list(tmp[case].keys())
                        if (resid_header[id] in keys) == False:
                            tmp[case][resid_header[id]] = [] 
                            logging.debug(f"Adding var {resid_header[id]} to case {case}")   

                        tmp[case][resid_header[id]].append(float(ele.split(":")[-1]))
        save_residuals(tmp)
    return tmp

def save_residuals(resis):
    path = os.path.join(sys.path[0], "conf.json")
    with open(path) as f:
        cfg = json.load(f)

    hpc_cases_dir = cfg["cases_dir_path"][1:]
    hpc_cases_dir[0] = "/" + hpc_cases_dir[0]

    cases = list(resis.keys())

    for ele in cases:
        if cfg["hpc_calculation"]:
            resi_path = os.path.join(*hpc_cases_dir, ele, ele + "_residuals.csv")
        else:
            resi_path = os.path.join(*cfg["cases_dir_path"], ele, ele + "_residuals.csv")
        logging.info(f"Saving residuals for case {ele} to {resi_path}")
        tmp = pd.DataFrame(resis[ele])
        tmp.drop_duplicates(keep='first', inplace=True, subset=['iter'])
        tmp.to_csv(resi_path, index=False)
        
def read_transient_data(config, case, export_times="flow_time"):
    """
    Read Ansys transient export data

    """
    if config["hpc_calculation"]:
        cases_dir_path = config["cases_dir_path"][1:]
        cases_dir_path[0] = "/" + cases_dir_path[0]
    else:
        cases_dir_path = config["cases_dir_path"]
    data_path = config["data_path"]
    times = config["plots"]

    path = os.path.join(*cases_dir_path, case, *data_path)
    files = glob.glob(r'*-Output*', root_dir=path)

    if files == []:
        logging.error("No data to process for case {}".format(case))
        exit()
    data = {}

    for file in files:
        
        m = re.findall(r'\d+', file)

        if m:
            if export_times != "flow_time":
                timestamp =int(m[0])
            else:
                timestamp =float(m[-2] + "." + m[-1])


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
            
            tmp_data_old = tmp_data

            if tmp_data.empty == False:
                tmp_data['x-coordinate'] = tmp_data['x-coordinate'].round(6)
                tmp_data['y-coordinate'] = tmp_data['y-coordinate'].round(6) 
                # tmp_data = tmp_data.round(6)
                data[timestamp] = tmp_data

    return data

def check_data_format(cases_dir_path):

        """
        Function that checks that all needed config parameters are set
        """

        path = os.path.join(*cases_dir_path)


        cases = os.listdir(path)
        

        for case in cases:
            csv_times = glob.glob(r'*-Output*.csv', root_dir=os.path.join(path, case))

            if csv_times != []:
                continue

            timepoints = glob.glob(r'*-Output*', root_dir=os.path.join(path, case))
            logging.debug("Found {} files for case {}".format(len(timepoints), case))
            renamed = 0
            for file in timepoints:
                if re.findall(r'.csv', file) == []:
                    old_path = os.path.join(path, case, file)
                    new_path = os.path.join(path, case, file + ".csv")
                    logging.debug(f"Rename {file} to {file + '.csv'}")
                    renamed += 1
                    os.rename(old_path, new_path)
            if renamed != 0:
                logging.info("Renamed {} files in case {}".format(renamed, case))