#!/usr/bin/python3

import os
import sys
import logging

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")

def clean():
    """
    Method to clean latex intermediate files
    """
    path = sys.path[0]
    data = os.listdir(path)
    delete(data, path)

    path = os.path.join(sys.path[0], "tex")
    data = os.listdir(path)
    delete(data, path)

def delete(data, path):
    deleted = 0
    for line in data:
        ending = line.split(".")[-1]
        if ending in ["aux", "bbl", "log", "out", "gz", "blg", "toc", "pdf", "lof", "lot"]:
            file = os.path.join(path, line)
            os.remove(file)
            logging.debug("Deleted {}".format(file))
            deleted += 1
    logging.info("Deleted {} files in folder {}".format(deleted, path))

if __name__ == "__main__":
    clean()