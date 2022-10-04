import os
import sys
import glob

def clear_tmp():
    
    """
    Function that clears all exported data from tmp directory
    """

    path = os.path.join(sys.path[0], "..", "..", "..", "..")
    trn_files = glob.glob('**/*.trn', root_dir=path, recursive=True)
    bat_files = glob.glob('**/*.bat', root_dir=path, recursive=True)
    for file in trn_files:
        os.remove(os.path.join(path, file))
    for file in bat_files:
        os.remove(os.path.join(path, file))

if __name__ == "__main__":
    clear_tmp()