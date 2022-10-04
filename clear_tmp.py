import os
import sys
import glob

def clear_tmp():
    
    """
    Function that clears all exported data from tmp directory
    """

    clear = input("What do you want to clear? (csv|n_csv|all)")

    path = os.path.join(sys.path[0], "Daten", "transient", "tmp")

    match clear:

        case "csv":
            files = glob.glob('*.csv', root_dir=path)
        
        case "n_csv":
            csv = glob.glob('*.csv', root_dir=path)
            files = glob.glob('*', root_dir=path)
            [files.remove(x) for x in csv]
        
        case "all":
            files = glob.glob('*', root_dir=path)

    
    
    for file in files:
        os.remove(os.path.join(path, file))

if __name__ == "__main__":
    clear_tmp()