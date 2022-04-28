#!/usr/bin/python3

import os
import sys

def setup():
    path = os.path.join(sys.path[0], "Latex", "clean.py")
    data_path = os.path.join(sys.path[0], "Daten")
    os.system("python3 " + path)
    if os.path.exists(data_path):
        os.system("rm -rf ./Daten")
    os.system("unzip ./Daten")

setup()