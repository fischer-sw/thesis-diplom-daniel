import json
from matplotlib import style
import numpy as np
import os
import sys
import logging

import matplotlib.pyplot as plt

from ansys_utils import *

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")


class flowfield:

    def __init__(self, config):

        self.config = config
        self.case = config["case"]
        self.field_var = config["field_var"]
        self.plots = config["plots"]
        self.one_plot = config["one_plot"]
        self.plot_vars = config["plot_vars"]
        self.do_plots = config["create_plot"]
        self.do_image = config["create_image"]

        self.case_conf = get_case_info(self.config["cases_dir_path"], self.case)


        check_data_format()
        
        if self.plots == []:
            logging.info("Plots are not set. Creating default ones ...")
            self.plots = get_default_cases(self.config["cases_dir_path"], self.case)
        logging.info("Plots are set to {}".format(self.plots))

        self.data = read_transient_data(self.config["cases_dir_path"], self.case, self.plots)

        cols = list(self.data[list(self.data.keys())[0]].columns)
        if self.do_plots == True:
            for var in self.config["plot_vars"]:
                if not var in cols:
                    logging.info("Declared var {} not within data. Please use one of {}".format(var, cols))
                    exit()
                
        if self.do_image == True:
            if not self.field_var[0] in cols:
                logging.info("Declared var {} not within data. Please use one of {}".format(self.field_var, cols))
                exit()
        
    def convert2field(self, data, vars):

        X = sorted(set(data['x-coordinate']))
        Y = sorted(set(data['y-coordinate']))
        res = {}
        for var in vars:
            Vals = np.zeros((len(Y), len(X)))
            for i in range(len(data[self.field_var])):
                x = data['x-coordinate'][i]
                y = data['y-coordinate'][i]
                val = data[var][i]
                ix = X.index(x)
                iy = Y.index(y)
                Vals[iy,ix] = val
            res[var] = Vals
        return X, Y, res

    def check_plot_cfg(self, conf):
        if len(self.config["plot_vars"]) != len(conf["linestyles"]):
            logging.info("Plot vars and colordefinitions don't match. Please check if all plotted variables have a respective color defined")
            exit()

        if len(self.plots) != len(conf["linestyles"]):
            logging.info("Please define a linestyle for all times that are plotted under linestyles")
            exit()

    def multi_plot(self):
        """
        Function that plots variables over the radius for multiple timesteps
        """

        plot_cfg = self.config["plot_conf"]

        self.check_plot_cfg(plot_cfg)

        if self.one_plot == False:

            fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))
            

            for idx, ele in enumerate(self.plots):

                X, Y, res = self.convert2field(self.data[ele], self.plot_vars)
                data_tmp = {}

                for var in self.plot_vars:
                    
                    # mirror across diagonal
                    x_tmp = Y
                    y_tmp = X
                    Vals = res[var]
                    logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))
                    Vals = np.rot90(np.fliplr(Vals))

                    tmp = np.array([])
                    for i in range(Vals.shape[1]):
                        tmp = np.append(tmp, np.mean(Vals[:,i]))
                    data_tmp[var] = tmp 

                # plot n

                if len(self.plots) != 1:

                    for tmp_var in data_tmp.keys():
                        col = plot_cfg["colors"][tmp_var]
                        cax = axs[idx].plot(x_tmp, data_tmp[tmp_var], color=col)
                        # add axis description
                    if idx == len(self.plots)-1 :
                        axs[idx].set_xlabel("radius r [m]")
                    axs[idx].set_ylabel("height z [m]")
                    axs[idx].set_title("t = {}s".format(ele * self.case_conf["timestep"]))
                    

                else:
                    for tmp_var in data_tmp.keys():
                        cax = axs.plot(x_tmp, data_tmp[tmp_var], color=col)
                        # add axis description

                    axs.set_xlabel("radius r [m]")
                    axs.set_ylabel("height z [m]")
                    axs.set_title("t = {}s".format(ele* self.case_conf["timestep"]))

        else:
            
            fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))
            
            l_conf = []
            for idx, ele in enumerate(self.plots):

                X, Y, res = self.convert2field(self.data[ele], self.plot_vars)
                data_tmp = {}
                
                for var in self.plot_vars:
                    
                    # mirror across diagonal
                    x_tmp = Y
                    y_tmp = X
                    Vals = res[var]
                    logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))
                    Vals = np.rot90(np.fliplr(Vals))

                    tmp = np.array([])
                    for i in range(Vals.shape[1]):
                        tmp = np.append(tmp, np.mean(Vals[:,i]))
                    data_tmp[var] = tmp 

                # plot n
                
                if len(self.plots) != 1:
                    
                    for tmp_var in data_tmp.keys():
                        l_style = plot_cfg["linestyles"]["t" + str(idx + 1)]
                        col = plot_cfg["colors"][tmp_var]
                        cax = axs.plot(x_tmp, data_tmp[tmp_var], linestyle=l_style, color=col)
                        l_conf.append(plot_cfg["legend"][tmp_var] + ", t={}s".format(ele * self.case_conf["timestep"]))
                        
                        # add axis description
                    if idx == len(self.plots)-1 :
                        axs.set_xlabel("radius r [m]")
                    axs.set_ylabel("height z [m]")
                    # axs.set_title("t = {}".format(ele))
                    axs.legend(l_conf)

                else:
                    for tmp_var in data_tmp.keys():
                        col = plot_cfg["colors"][tmp_var]
                        cax = axs.plot(x_tmp, data_tmp[tmp_var], color=col)
                        l_conf.append(plot_cfg["legend"][tmp_var])
                        # add axis description

                    axs.set_xlabel("radius r [m]")
                    axs.set_ylabel("height z [m]")
                    axs.legend(l_conf)
                    # axs.set_title("t = {}".format(ele))
            
        path = sys.path[0]
        path = os.path.join(path, "Images")
        sub_path = os.path.join(path, "transient")
        if os.path.exists(path) == False:
            os.mkdir(path)
        if os.path.exists(sub_path) == False:
            os.mkdir(sub_path)

        image_name = self.config["plot_file_name"] + "." + self.config["plot_file_type"]
        image_path = os.path.join(sub_path, image_name)

        plt.savefig(image_path)
    
    def multi_field(self):
        """
        Function that shows ansys fields for multiple timesteps within one image
        """
        
        
        fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))
        

        for idx, ele in enumerate(self.plots):

            X, Y, res = self.convert2field(self.data[ele], self.field_var)
            # mirror across diagonal
            x_tmp = Y
            y_tmp = X
            Vals = res[self.field_var[0]]
            logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))
            Vals = np.rot90(np.fliplr(Vals))            
            # plot n

            if len(self.plots) != 1:
                cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
                # add axis description
                if idx == len(self.plots)-1 :
                    axs[idx].set_xlabel("radius r [m]")
                axs[idx].set_ylabel("height z [m]")
                axs[idx].set_title("t = {}s".format(ele * self.case_conf["timestep"]))
                # add colorbar
                cbar = fig.colorbar(cax, ax=axs[idx])
                cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)

            else:
                cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))

                axs.set_xlabel("radius r [m]")
                axs.set_ylabel("height z [m]")
                axs.set_title("t = {}".format(ele))
                # add colorbar
                cbar = fig.colorbar(cax, ax=axs)
                cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)
            
        path = sys.path[0]
        path = os.path.join(path, "Images")
        sub_path = os.path.join(path, "transient")
        if os.path.exists(path) == False:
            os.mkdir(path)
        if os.path.exists(sub_path) == False:
            os.mkdir(sub_path)

        image_name = self.config["image_file_name"] + "." + self.config["image_file_type"]
        image_path = os.path.join(sub_path, image_name)

        plt.savefig(image_path)

if __name__ == "__main__":

    cfg_path = os.path.join(sys.path[0], "conf.json")

    with open(cfg_path) as f:
        config = json.load(f)

    field = flowfield(config)
    
    if config["create_image"]:
        field.multi_field()
    
    if config["create_plot"]:
        field.multi_plot()
    