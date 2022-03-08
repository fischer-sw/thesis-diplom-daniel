#!/usr/bin/python3

import os
import sys


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
    for line in data:
        ending = line.split(".")[-1]
        if ending in ["aux", "bbl", "log", "out", "gz", "blg", "toc", "pdf", "lof", "lot"]:
            os.remove(os.path.join(path, line))

clean()