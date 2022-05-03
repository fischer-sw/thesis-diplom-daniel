import math
import numpy as np
import os
import sys
import logging

import matplotlib.pyplot as plt

from ansys_utils import *

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")


class flowfield:

    def __init__(self, case, var, plots=[]):
        self.case = case
        self.var = var
        
        if plots == []:
            logging.info("Plots are not set. Creating default ones ...")
            case_ns = get_case_nums(case)
            self.plots = [1, round(case_ns/2), case_ns]
        else:
            self.plots = plots
        logging.info("Plots are set to {}".format(self.plots))

        self.data = read_transient_data(case, self.plots)

        cols = list(self.data[list(self.data.keys())[0]].columns)
        if not var in cols:
            logging.info("Declared var {} not within data. Please use one of {}".format(var, cols))
            exit()
        

    def convert2field(self, data):

        X = sorted(set(data['x-coordinate']))
        Y = sorted(set(data['y-coordinate']))
        Vals = np.zeros((len(Y), len(X)))

        for i in range(len(data[self.var])):
            x = data['x-coordinate'][i]
            y = data['y-coordinate'][i]
            val = data[self.var][i]
            ix = X.index(x)
            iy = Y.index(y)
            Vals[iy,ix] = val

        return X, Y, Vals
    
    def multi_field(self, conf={}):
        """
        Function that shows calculated and ansys flowfield within one image
        """
        
        
        fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6,2*len(self.plots)+5))
        

        for idx, ele in enumerate(self.plots):

            X, Y, Vals = self.convert2field(self.data[ele])

            logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))

            # mirror across diagonal
            x_tmp = X
            y_tmp = Y
            # Vals = np.rot90(np.fliplr(Vals))
            
            # plot n
            cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
            # add axis description
            if idx == len(self.plots)-1 :
                axs[idx].set_xlabel("radius r [m]")
            axs[idx].set_ylabel("height z [m]")
            axs[idx].set_title("t = {}".format(ele))
            # add colorbar
            cbar = fig.colorbar(cax, ax=axs[idx])
            cbar.set_label(conf["c_bar"], rotation=90, labelpad=7)

        path = sys.path[0]
        path = os.path.join(path, "Images")
        sub_path = os.path.join(path, "transient")
        if os.path.exists(path) == False:
            os.mkdir(path)
        if os.path.exists(sub_path) == False:
            os.mkdir(sub_path)

        image_name = conf["name"] + ".png"
        image_path = os.path.join(sub_path, image_name)

        plt.savefig(image_path)

if __name__ == "__main__":

    test = flowfield("test", "velocity-magnitude", plots=list(range(1,2)))

    config = {
        "name" : "test_fields",
        "c_bar" : "Velocity [m/s]"
    }

    test.multi_field(conf=config) 