import os
import sys
import glob

def clear_tmp():
    
    """
    Function that clears all exported data from tmp directory
    """

    path = os.path.join(sys.path[0])
    files = glob.glob('*.trn', root_dir=path)
    for file in files:
        os.remove(os.path.join(path, file))

if __name__ == "__main__":
    clear_tmp()