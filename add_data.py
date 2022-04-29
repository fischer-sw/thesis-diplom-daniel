#!/usr/bin/python3

import os
import sys
import re
import logging

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")

base_path = sys.path[0]
path = os.path.join(base_path, "Daten")
zip_path = os.path.join(base_path, "Daten.zip")

def zip_data():
    
    logging.debug("Pfad = {}".format(path))

    if os.path.exists(path):
        
        logging.debug("Folder Daten exsists")
        
        if os.path.exists(zip_path):
            logging.info("Removing Daten.zip")    
            os.remove("Daten.zip")
        else:
            logging.info("Daten.zip already removed")

        logging.info("Adding Daten to Daten.zip")
        os.system("zip -r Daten.zip ./Daten")

    else:
        logging.warning("Folder Daten doesn't exsist. Please run setup.py")
    

def check_data_format():
    
    modes = os.listdir(path)
    logging.debug("Modi = {}".format(modes))

    for mode in modes:
        cases = os.listdir(os.path.join(path, mode))
        logging.debug("Found {} for mode {}".format(cases, mode))

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
            logging.info("Renamed {} files in case {} in mode {}".format(renamed, case, mode))


if __name__ == "__main__":
    check_data_format()
    zip_data()
    