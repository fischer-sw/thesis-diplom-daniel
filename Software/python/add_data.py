#!/usr/bin/python3

import os
import shutil
import sys
import re
import logging
import json
import glob
import datetime

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")

base_path = sys.path[0]
path = os.path.join(base_path, "../..","Daten")
zip_path = os.path.join(base_path, "../.." ,"Daten.zip")

def zip_data():
    
    logging.debug("Pfad = {}".format(path))

    if os.path.exists(path):
        
        logging.debug("Folder Daten exsists")
        
        # if os.path.exists(zip_path):
        #     logging.info("Removing Daten.zip")    
        #     os.remove("Daten.zip")
        # else:
        #     logging.info("Daten.zip already removed")

        name = "Daten_{}.zip".format(datetime.date.today().strftime("%d-%m-%Y"))
        logging.info("Adding Daten to Daten.zip")
        os.system("zip -r {} ./Daten".format(name))

    else:
        logging.warning("Folder Daten doesn't exsist. Please run setup.py")
    

def check_data_format():
    
    modes = os.listdir(path)
    logging.debug("Modi = {}".format(modes))

    for mode in modes:
        cases = os.listdir(os.path.join(path, mode))
        logging.debug("Found {} for mode {}".format(cases, mode))
        if "cases.json" in cases:
            cases.remove("cases.json")

        for case in cases:
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

def add_case(original, new_case, move_csv, move_all=False):
    """
    Function that moves new results from tmp to new case directory
    """
    
    old_path = os.path.join(path, "transient", original)
    new_path = os.path.join(path, "transient", new_case)
    # files = os.listdir(old_path)    
    # check for new directory

    deleted = 0
    moved = 0
    csv = glob.glob('*.csv', root_dir=old_path)
    non_csv = glob.glob('*', root_dir=old_path)
    [non_csv.remove(x) for x in csv]

    if move_all:
        files = glob.glob('*', root_dir=old_path)
    else:
        if move_csv == True:
            files = csv
        else:
            files = non_csv


    if os.path.exists(new_path) == False and files != []:
        os.mkdir(new_path)
        logging.info("Created directory for case {}".format(new_case))
    else:
        if files != []:
            old_files = os.listdir(new_path)
            for ele in old_files:
                os.remove(os.path.join(new_path, ele))
                deleted += 1
            logging.info("Deleted {} files in case {}".format(deleted, new_case))
        else:
            logging.info("No files to add from dir {} to case {}".format(original, new_case))
            exit()

        
    
    for file in files:
        if not ".csv" in file:
            old = os.path.join(old_path, file)
            new = os.path.join(new_path, file + ".csv")
            shutil.move(old, new)
            moved += 1
        else:
            old = os.path.join(old_path, file)
            new = os.path.join(new_path, file)
            shutil.move(old, new)
            moved += 1
    if moved != 0:
        logging.info("Moved {} files from {} to {}".format(moved, original, new_case))
        
    with open(os.path.join(path, "transient", "cases.json")) as f:
        case_cfg = json.load(f)

    if new_case in list(case_cfg.keys()):
        logging.info("Case {} already part of config.".format(new_case))
    else:
        case_cfg[new_case] = {}
        logging.info("Adding case {} to case.json. Please input all relevant parameters there before further processing".format(new_case))
        with open(os.path.join(path, "transient", "cases.json"), "w") as f:
            f.write(json.dumps(case_cfg, indent=4))


if __name__ == "__main__":

    add_case_opt = input("Do you want to add a case? (y/n)")

    if add_case_opt == "y":
        
        case_name = input("Please enter case_name: ")

        old_path = os.path.join(path, "transient", "tmp")
        csv_dat = glob.glob('*.csv', root_dir=old_path)
        non_csv = glob.glob('*', root_dir=old_path)
        [non_csv.remove(x) for x in csv_dat]

        # csv = "n_csv"

        if csv_dat != [] and non_csv != []:
            csv = input("Do you want to move csv or non_csv files?(csv/n_csv/all)")
        
        else:
            if csv_dat != []:
                csv = "csv"
            else:
                csv = "n_csv"
            
        match csv:

            case "csv":
                move_csv = True
                move_all = False

            case "n_csv":
                move_csv = False
                move_all = False

            case "all":
                move_all = True
                move_csv = True

            
        if case_name != "":
            add_case("tmp", case_name, move_csv, move_all)

    do_zip = input("Do you want to zip the Data? (y/n) (default n)")

    if do_zip == "y":
        check_data_format()
        zip_data()
    