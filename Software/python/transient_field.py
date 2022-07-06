import json
from matplotlib import style
import numpy as np
import os
import sys
import logging
import glob
import cv2

from PIL import Image
import matplotlib.pyplot as plt

from ansys_utils import *

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


class flowfield:

    def __init__(self, config):

        self.config = config
        check_data_format(self.config["cases_dir_path"])

    def convert2field(self, data, vars, vel_field=False):
        """
        Function that converts list of x,y,<var> to matrix of dimensions x,y with <var> as values inside
        """

        if vel_field == False:
            X = sorted(set(data['x-coordinate']))
            Y = sorted(set(data['y-coordinate']))
            res = {}
            for var in vars:
                Vals = np.zeros((len(Y), len(X)))
                # X_vals = np.zeros((len(Y), len(X)))
                # Y_vals = np.zeros((len(Y), len(X)))
                
                for i in range(len(data[vars[0]])):
                    x = data['x-coordinate'][i]
                    y = data['y-coordinate'][i]
                    val = data[var][i]
                    ix = X.index(x)
                    iy = Y.index(y)
                    Vals[iy,ix] = val
                    # X_vals[iy,ix] = x
                    # Y_vals[iy,ix] = y
                res[var] = Vals
                # res['x-field'] = X_vals
                # res['y-field'] = Y_vals

            return X, Y, res
        else:

            x_diffs = 16
            y_diffs = 16

            X = sorted(set(data['x-coordinate']))
            step_x = int(round(len(X)/x_diffs,0))
            X_tmp = [X[i] for i in range(0, len(X), step_x)]
            Y = sorted(set(data['y-coordinate']))
            step_y = int(round(len(Y)/y_diffs,0))
            Y_tmp = [Y[i] for i in range(0, len(Y), step_y)]
            
            res = {}
            for var in vars:
                Vals = np.zeros((len(Y), len(X)))
                X_vals = np.zeros((len(Y), len(X)))
                Y_vals = np.zeros((len(Y), len(X)))
                
                for i in range(len(data[vars[0]])):
                    x = data['x-coordinate'][i]
                    y = data['y-coordinate'][i]
                    val = data[var][i]
                    ix = X.index(x)
                    iy = Y.index(y)
                    Vals[iy,ix] = val
                    X_vals[iy,ix] = x
                    Y_vals[iy,ix] = y
                res[var] = Vals
                res['x-field'] = X_vals
                res['y-field'] = Y_vals

                small_field = np.zeros([int(len(Y_tmp)), int(len(X_tmp))])
                small_x = np.zeros([int(len(Y_tmp)), int(len(X_tmp))])
                small_y = np.zeros([int(len(Y_tmp)), int(len(X_tmp))])

                for i in range(len(Y_tmp)):
                    big_y_idx = Y.index(Y_tmp[i])
                    for j in range(len(X_tmp)):
                        big_x_idx = X.index(X_tmp[j])
                        small_field[i,j] = res[var][big_y_idx][big_x_idx]
                        small_x[i,j] = X_tmp[j]
                        small_y[i,j] = Y_tmp[i]

                res[var] = small_field
                res['x-field'] = small_x
                res['y-field'] = small_y
                    

            return X_tmp, Y_tmp, res

    def check_plot_cfg(self, conf):

        """
        Function that checks plot configuration for defined vars
        """

        if len(self.config["plot_vars"]) != len(conf["linestyles"]):
            logging.info("Plot vars and colordefinitions don't match. Please check if all plotted variables have a respective color defined")
            exit()

        if len(self.plots) != len(conf["linestyles"]):
            logging.info("Please define a linestyle for all times that are plotted under linestyles")
            exit()

    def update_plot_cfg(self, case=None):

        """
        Function that updates plot config and returns plot configuration (return value is used for checking cfg in one_plot case)
        """

        plot_cfg = self.config["plot_conf"]
        self.plots = self.config["plots"]
        self.cases = self.config["cases"]
        self.resid = self.config["create_resi_plot"]
        self.cases_dir = self.config["cases_dir_path"]
        self.case = case
        self.field_var = self.config["field_var"]
        self.plots = self.config["plots"]
        self.one_plot = self.config["one_plot"]
        self.plot_vars = self.config["plot_vars"]
        self.do_plots = self.config["create_plot"]
        self.do_image = self.config["create_image"]
        self.image_conf = self.config["image_conf"]
        self.gif_conf = self.config["gif_conf"]
        

        if case != None:
            self.case_conf = get_case_info(self.case)
            self.export_times = self.case_conf["export_times"]
            
            if self.plots == []:
                logging.info("Plots are not set. Creating default ones ...")
                self.plots = get_default_cases(self.config["cases_dir_path"], self.case, self.export_times)

            else:
                self.plots = get_closest_plots(np.array(self.plots)/self.case_conf["timestep"], self.case_conf["timestep"] ,self.config["cases_dir_path"], self.case, self.export_times)
            
            self.data = read_transient_data(self.config["cases_dir_path"], self.case, self.plots, self.export_times)

            cols = list(self.data[list(self.data.keys())[0]].columns)
            cols.append('velocity-field')
            if self.do_plots == True:
                for var in self.config["plot_vars"]:
                    if not var in cols:
                        logging.warning("Declared var {} not within data. Please use one of {}".format(var, cols))
                        exit()
                    
            if self.do_image == True:
                if not self.field_var[0] in cols:
                    logging.warning("Declared var {} not within data. Please use one of {}".format(self.field_var, cols))
                    exit()

        return plot_cfg

    def vel_field(self):
        """
        Function that plots velocity filed with directional info
        """
        
        for cas in self.cases:

            self.case = cas

            self.update_plot_cfg(cas)
            var = "velocity-field"

            path = sys.path[0]
            path = os.path.join(path, "assets")
            sub_path = os.path.join(path, var)
            if os.path.exists(path) == False:
                os.mkdir(path)
            if os.path.exists(sub_path) == False:
                os.mkdir(sub_path)

            image_name = "field_" + self.case + "_" + var + "." + self.config["image_file_type"]
            image_path = os.path.join(sub_path, image_name)

            if os.path.exists(image_path) and self.config["ignore_exsisting"] == True:
                logging.info(f"{var} field for case {self.case} already exsists.")
                continue

            logging.info(f"Creating {var} field {self.plots} for case {self.case}")

            title = self.case

            fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))
            fig.suptitle(title, size=12)
            # axs = fig.add_subplot(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))

            for idx, ele in enumerate(self.plots):
                elements = ['x-coordinate', 'y-coordinate', 'axial-velocity', 'radial-velocity']
                data_header = list(self.data[ele].columns)
                if all(elem in data_header for elem in elements) == False:
                    logging.error(f"Not all needed elements {elements} for plot {ele} in data for case {cas}")
                    continue

                X, Y, res = self.convert2field(self.data[ele], ['axial-velocity', 'radial-velocity'], vel_field=True)
                # mirror across diagonal
                x_tmp = Y
                y_tmp = X
                v_z_vals = res['axial-velocity']
                v_z_vals = np.rot90(np.fliplr(v_z_vals))
                v_r_vals = res['radial-velocity']
                v_r_vals = np.rot90(np.fliplr(v_r_vals))

                x_final = res['x-field']
                x_final = np.rot90(np.fliplr(x_final))
                y_final = res['y-field']
                y_final = np.rot90(np.fliplr(y_final))
                
                logging.debug("Size X = {}, Size Y = {}, Size v_z = {}, Size v_r = {}".format(len(X), len(Y), len(v_z_vals), len(v_r_vals)))
                            
                # plot n
                scl_val= 0.02

                if len(self.plots) != 1:
                    if self.image_conf["set_custom_range"]:
                        # cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=self.image_conf["min"], vmax=self.image_conf["max"])
                        cax = axs[idx].quiver(y_final, x_final, v_r_vals, v_z_vals, scale=scl_val)
                    else:
                        cax = axs[idx].quiver(y_final, x_final, v_r_vals, v_z_vals, scale=scl_val)
                        # cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
                    # add axis description
                    if idx == len(self.plots)-1 :
                        axs[idx].set_xlabel("radius r [m]")
                    axs[idx].set_ylabel("height z [m]")
                    axs[idx].set_title("t = {}s".format(round(ele * self.case_conf["timestep"], 1)))
                    # add colorbar
                    cbar = fig.colorbar(cax, ax=axs[idx])
                    cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)

                else:
                    if self.image_conf["set_custom_range"]:
                        # cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=self.image_conf["min"], vmax=self.image_conf["max"])
                        cax = axs.quiver(y_final, x_final, v_r_vals, v_z_vals, scale=scl_val)
                    else:
                        # cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
                        cax = axs.quiver(y_final, x_final, v_r_vals, v_z_vals, scale=scl_val)
                    axs.set_xlabel("radius r [m]")
                    axs.set_ylabel("height z [m]")
                    axs.set_title("t = {}s".format(round(ele * self.case_conf["timestep"], 1)))
                    # add colorbar
                    cbar = fig.colorbar(cax, ax=axs)
                    cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)
            
            plt.savefig(image_path)
            plt.close(fig)
            logging.info(f"saved image {image_name}.")



    def resi_plot(self):
        """
        Function that generates residuals plot for a case
        """
        self.update_plot_cfg()

        for cas in self.cases:

                path = sys.path[0]
                path = os.path.join(path, "assets")
                sub_path = os.path.join(path, "residuals")
                if os.path.exists(path) == False:
                    os.mkdir(path)
                if os.path.exists(sub_path) == False:
                    os.mkdir(sub_path)

                image_name = "residuals_" + cas + "." + self.config["plot_file_type"]
                image_path = os.path.join(sub_path, image_name)

                if os.path.exists(image_path):
                    logging.info(f"Residuals for case {cas} already created")
                    continue
            
                logging.info(f"Creating residuals plot for case {cas}")

                resid_file = glob.glob(r'*residuals.csv', root_dir=os.path.join(*self.cases_dir, cas))
                data_path = os.path.join(*self.cases_dir, cas, resid_file[0])
                data = pd.read_csv(data_path)

                plot_vars = list(data.columns)
                plot_vars.remove("time/iter")
                plot_vars.remove("iter")

                fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,4))

                l_style = "solid"

                for ele in plot_vars:

                    axs.plot(data['iter'], data[ele], linestyle=l_style)
                
                # axs.set_ylim([1e-10, 1e-2])
                axs.set_yscale('log')
                axs.set_xlabel("Iterations")
                axs.set_ylabel("value")
                axs.legend(plot_vars)
                axs.set_title(f"Resiudals {cas}")

                plt.savefig(image_path)
                plt.close(fig)
                logging.info(f"saved image {image_name}.")
            


    def multi_plot(self, update_conf=True):
        """
        Function that plots variables over the radius for multiple timesteps
        """
        if update_conf == True:
            self.update_plot_cfg()

        for cas in self.cases:

            self.case = cas
            plot_cfg = self.update_plot_cfg(cas)

            path = sys.path[0]
            path = os.path.join(path, "assets")
            sub_path = os.path.join(path, "plots")
            if os.path.exists(path) == False:
                os.mkdir(path)
            if os.path.exists(sub_path) == False:
                os.mkdir(sub_path)

            if update_conf:
                image_name = "plot_" + self.case + "_" + "_".join(self.plot_vars) + "." + self.config["plot_file_type"]
            else:
                image_name = self.config["plot_file_name"] + self.config["plot_file_type"]
            image_path = os.path.join(sub_path, image_name)

            if os.path.exists(image_path):
                logging.info(f"{self.plot_vars} field for case {self.case} already created")
                continue

            logging.info(f"Creating {self.plot_vars} field {self.plots} for case {self.case}")


            if self.one_plot == False:
                
                title = ""

                fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))

                fig.suptitle(title)

                legend = []

                for var in list(plot_cfg["legend"]):
                    legend.append(plot_cfg["legend"][var])
                

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
                        axs[idx].legend(legend)
                        if self.export_times != "flow_time":
                            axs[idx].set_title("t = {}s".format(round(ele * self.case_conf["timestep"],1)))
                        else:
                            axs[idx].set_title("t = {}s".format(ele))


                    else:
                        for tmp_var in data_tmp.keys():
                            col = plot_cfg["colors"][tmp_var]
                            cax = axs.plot(x_tmp, data_tmp[tmp_var], color=col)
                            # add axis description

                        axs.set_xlabel("radius r [m]")
                        axs.set_ylabel("height z [m]")
                        axs.legend(legend)
                        if self.export_times != "flow_time":
                            axs.set_title("t = {}s".format(round(ele* self.case_conf["timestep"],1)))
                        else:
                            axs.set_title("t = {}s".format(ele))

            else:
                
                self.check_plot_cfg(plot_cfg)

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
                            if self.export_times != "flow_time":
                                l_conf.append(plot_cfg["legend"][tmp_var] + ", t={}s".format(round(ele * self.case_conf["timestep"],1)))
                            else:
                                l_conf.append(plot_cfg["legend"][tmp_var] + ", t={}s".format(ele))

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
                
           

            plt.savefig(image_path)
            plt.close(fig)
            logging.info(f"saved image {image_name}.")
    
    def multi_field(self, update_conf=True):
        """
        Function that shows ansys fields for multiple timesteps within one image
        """
        if update_conf == True:
            self.update_plot_cfg()

        if 'velocity-field' in self.config["field_var"]:
            self.vel_field()

        for cas in self.cases:

            self.case = cas

            self.update_plot_cfg(cas)

            if 'velocity-field' in self.field_var:
                self.field_var.remove('velocity-field')

            for var in self.field_var:

                var = [var]

                path = sys.path[0]
                path = os.path.join(path, "assets")
                sub_path = os.path.join(path, var[0])
                if os.path.exists(path) == False:
                    os.mkdir(path)
                if os.path.exists(sub_path) == False:
                    os.mkdir(sub_path)

                if update_conf == True:
                    image_name = "field_" + self.case + "_" + var[0] + "." + self.config["image_file_type"]
                else:
                    image_name = self.config["image_file_name"] + "." + self.config["image_file_type"]
                image_path = os.path.join(sub_path, image_name)

                if os.path.exists(image_path) and self.config["ignore_exsisting"] == True:
                    logging.info(f"{var} field for case {self.case} already exsists.")
                    continue

                logging.info(f"Creating {var} field {self.plots} for case {self.case}")

                title = self.case

                fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5, 2.0*len(self.plots)+2.5))
                fig.suptitle(title, size=12)
                # axs = fig.add_subplot(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))

                for idx, ele in enumerate(self.plots):

                    X, Y, res = self.convert2field(self.data[ele], var)
                    # mirror across diagonal
                    x_tmp = Y
                    y_tmp = X
                    Vals = res[var[0]]
                    logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))
                    Vals = np.rot90(np.fliplr(Vals))            
                    # plot n

                    if len(self.plots) != 1:
                        if self.image_conf["set_custom_range"]:
                            cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=self.image_conf["min"], vmax=self.image_conf["max"])
                        else:
                            cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
                        # add axis description
                        if idx == len(self.plots)-1 :
                            axs[idx].set_xlabel("radius r [m]")
                        axs[idx].set_ylabel("height z [m]")
                        axs[idx].set_title("t = {}s".format(round(ele * self.case_conf["timestep"], 1)))
                        # add colorbar
                        cbar = fig.colorbar(cax, ax=axs[idx])
                        cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)

                    else:
                        if self.image_conf["set_custom_range"]:
                            cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=self.image_conf["min"], vmax=self.image_conf["max"])
                        else:
                            cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))

                        axs.set_xlabel("radius r [m]")
                        axs.set_ylabel("height z [m]")
                        if self.export_times != "flow_time":
                            axs.set_title("t = {}s".format(round(ele * self.case_conf["timestep"], 1)))
                        else:
                            axs.set_title("t = {}s".format(ele))

                        # add colorbar
                        cbar = fig.colorbar(cax, ax=axs)
                        cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)
                
                plt.savefig(image_path)
                plt.close(fig)
                logging.info(f"saved image {image_name}.")

    def setup_journal(self, exit=True, split_cases=False):
        """
        Function that creates journal files
        """

        journal_path = os.path.join(sys.path[0], "..", "ansys", "journals")

        files = glob.glob(r'**/*.jou', root_dir=journal_path, recursive=True)
        files.remove("gui_template.jou")
        for file in files:
            os.remove(os.path.join(journal_path, file))
        logging.info(f"Removed {len(files)} journals")

        build_journal(self.config["cases_dir_path"], split_cases, exit)


    def delete_gif_imgs(self):

        """
        Function that deletes all images used to create the gifs and videos
        """

        path = sys.path[0]
        img_path = os.path.join(path, "assets", self.field_var[0])

        gifs = glob.glob('*_gif*', root_dir=img_path)

        if gifs == []:
            logging.info("No gif Images to delete")
        else:
            
            for ele in gifs:
                os.remove(os.path.join(img_path, ele))
            logging.info(f"Deleted {len(gifs)} image files")

    def create_gif(self):
        
        """
        Function that creates missing Images if necessary and then creates .gif out of all obtained images
        """
        if not hasattr(self, 'cases'):
            self.update_plot_cfg()

        for cas in self.cases:

            self.case = cas

            self.update_plot_cfg(cas)

            path = sys.path[0]
            img_path = os.path.join(path, "assets", self.field_var[0])

            if self.gif_conf["new"]:
                self.delete_gif_imgs()

            logging.info("Creating images for .gif ...")

            path = os.path.join(path, "gifs")
            if os.path.exists(path) == False:
                os.mkdir(path)

            # create plots

            cases = get_cases(self.config["cases_dir_path"], self.case)


            raw_cases = list(range(int(self.gif_conf["cases"]["start"]), int(self.gif_conf["cases"]["end"]) + 1, int(self.gif_conf["cases"]["step"])))

            if raw_cases != []:
                start_end = get_closest_plots(np.array(raw_cases)/self.case_conf["timestep"], self.case_conf["timestep"] ,self.config["cases_dir_path"], self.case, self.export_times)
                cases = list(set(start_end))
                cases.sort()

            digits = len(str(int(max(cases))))
            plot_images = []
            field_images = []

            for cas in cases:
                self.config["plots"] = [cas * self.case_conf["timestep"]]
                plot_name = "_".join(["plot_gif", self.case, f"{int(cas):0{digits}d}"])
                
                self.config["plot_file_name"] = plot_name
                field_name = "_".join(["img_gif", self.case, f"{int(cas):0{digits}d}"])
                
                self.config["image_file_name"] = field_name
                
                if self.gif_conf["gif_plot"]:

                    number = int(cas)
                    search_res = glob.glob(f'*{number}.png', root_dir=img_path)
                    if search_res != [] and int(re.findall('\d+', search_res[0])[0]) == number:
                        logging.info(f"Image {plot_name} already exsists")
                        plot_images.append(search_res[0])
                    else:
                        self.multi_field(update_conf=False)
                        plot_images.append(plot_name)

                if self.gif_conf["gif_image"]:
                    number = int(cas)
                    search_res = glob.glob(f'*{number}.png', root_dir=img_path)
                    if search_res != [] and int(re.findall('\d+', search_res[0])[0]) == number:
                        logging.info(f"Image {field_name} already exsists")
                        field_images.append(search_res[0])
                    else:
                        self.multi_field(update_conf=False)
                        field_images.append(field_name)

            # create Images
            if self.gif_conf["gif_plot"]:
                
                gif_name = "_".join([self.gif_conf["name"], "plot"]) + ".gif"
                gif_path = os.path.join(path, gif_name)

                video_name = "_".join([self.gif_conf["name"], "plot"]) + ".avi"
                video_path = os.path.join(path, video_name)

                if os.path.exists(gif_path):
                    logging.info(f"Deleting existing gif {gif_name}")
                if os.path.exists(video_path):
                    logging.info(f"Deleting existing video {video_name}")
                
                imgs = (Image.open(os.path.join(img_path,f)) for f in plot_images)
                img = next(imgs)  # extract first image from iterator
                img.save(gif_path, format="GIF", append_images=imgs,
                        save_all=True, duration=self.gif_conf["frame_duration"], loop=self.gif_conf["loop"])
                logging.info(f"Created gif for variable {self.plot_vars} for case {self.case}")

                if self.gif_conf["videos"]:

                    logging.info(f"Creating plot video for case {self.case}")
                    
                    # images = sorted(glob.glob('plot_gif_*png', root_dir=img_path))
                    images = plot_images

                    frame = cv2.imread(os.path.join(img_path, images[0]))

                    height, width, layers = frame.shape
                    fps = 1000/self.gif_conf["frame_duration"]
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
                    

                    for image in images:
                        img = cv2.imread(os.path.join(img_path, image))
                        out.write(img)

                    cv2.destroyAllWindows()
                    out.release()
                    logging.info(f"Video created for var {self.plot_vars} for case {self.case}.")


            if self.gif_conf["gif_image"]:

                logging.info(f"Creating field video for case {self.case}")
                
                gif_name = "_".join([self.gif_conf["name"], "image"]) + ".gif"
                video_name = "_".join([self.gif_conf["name"], "image"]) + ".avi"
                
                gif_path = os.path.join(path, gif_name)
                video_path = os.path.join(path, video_name)

                if os.path.exists(gif_path):
                    logging.info(f"Deleting existing gif {gif_name}")
                if os.path.exists(video_path):
                    logging.info(f"Deleting existing video {video_name}")

                tmp_imgs = field_images

                if tmp_imgs == []:
                    logging.error(f"No images for gif found at {img_path}")
                    exit()

                imgs = (Image.open(os.path.join(img_path,f)) for f in field_images)
                img = next(imgs)  # extract first image from iterator
                img.save(gif_path, format="GIF", append_images=imgs,
                        save_all=True, duration=self.gif_conf["frame_duration"], loop=self.gif_conf["loop"])
                logging.info(f"Created gif for variable {self.field_var} for case {self.case}")

                if self.gif_conf["videos"]:
                    
                    # images = sorted(glob.glob('img_gif_*png', root_dir=img_path))
                    images = field_images

                    frame = cv2.imread(os.path.join(img_path, images[0]))

                    height, width, layers = frame.shape
                    fps = 1000/self.gif_conf["frame_duration"]
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
                    

                    for image in images:
                        img = cv2.imread(os.path.join(img_path, image))
                        out.write(img)

                    cv2.destroyAllWindows()
                    out.release()
                    logging.info(f"Video created for var {self.field_var} for case {self.case}.")

            if self.gif_conf["keep_images"] == False:
                self.delete_gif_imgs()    

def do_plots():
    cfg_path = os.path.join(sys.path[0], "conf.json")

    with open(cfg_path) as f:
        config = json.load(f)

    field = flowfield(config)
    
    if config["create_image"]:
        field.multi_field()
    
    if config["create_plot"]:
        field.multi_plot()

    if config["create_gif"]:
        field.create_gif()

    if config["create_resi_plot"]:
        field.resi_plot()

if __name__ == "__main__":

    do_plots()

    
    