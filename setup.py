import os
import sys
import shutil
import logging
import zipfile

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")

def setup():
    logging.info("Starting Latex clean")
    # path = os.path.join(sys.path[0], "Latex", "clean.py")
    data_path = os.path.join("\\\\gssnas", "bigdata", "fwdt", "DFischer")
    
    if os.path.exists(data_path):
        logging.info(os.listdir(data_path))
    else:
        logging.info(f"Path {data_path} doesn't exsist.")
        exit()
    # os.system("python3 " + path)
    if os.path.exists(data_path):
        logging.info("Removing old Daten folder")
        shutil.rmtree("./Daten")
    logging.info("Adding new Daten folder")
    with zipfile.ZipFile("./Daten.zip", 'r') as zip_ref:
        zip_ref.extractall("./")
    

if __name__ == "__main__":
    setup()