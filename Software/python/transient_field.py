import json
import os
import sys
import logging
import glob
import cv2
import math

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ansys_utils import *

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


class flowfield:

    def __init__(self, config=None, cases_cfg=None):

        if config == None or cases_cfg == None:
            logging.error("No cases.json or conf.json provided")
            exit()

    def convert2field(self, data, vars, vel_field=False):
        """
        Function that converts list of x,y,<var> to matrix of dimensions x,y with <var> as values inside
        """

        if vel_field == False:
            X = sorted(set(data['x-coordinate']))
            Y = sorted(set(data['y-coordinate']))
            if len(X) == len(Y):
                logging.error(f"Something went completly wrong.")
                exit()
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

    def update_plot_cfg(self, config, cases_cfg, case=None, update_from_file=False):

        """
        Function that updates plot config and returns plot configuration (return value is used for checking cfg in one_plot case)
        """

        if update_from_file:
            cf_path = os.path.join(sys.path[0],"conf.json")
            with open(cf_path) as f:
                config = json.load(f)
        plots = config["plots"]
        field_var = config["field_var"]
        plots = config["plots"]
        do_plots = config["create_plot"]
        do_image = config["create_image"]

        if case != None:
            if case in list(cases_cfg.keys()):
                case_conf = cases_cfg[case]
            else:
                logging.error(f"Case {case} not in provided config.")
                exit()

            export_times = case_conf["export_times"]
            
            if plots == []:
                logging.info("Plots are not set. Creating default ones ...")
                config["plots"] = get_default_cases(config, case)

            else:
                # config["plots"] = get_closest_plots(np.array(plots)/case_conf["data_export_interval"], case_conf["data_export_interval"] ,config["cases_dir_path"], config["data_path"], case, export_times)
                config["plots"] = get_closest_plots(config, cases_cfg, case, export_times)
                
            cols = get_case_vars(config, case)
            cols.append('velocity-field')
            if do_plots == True:
                for var in config["plot_vars"]:
                    if not var in cols:
                        logging.warning("Declared var {} for plot not within data. Please use one of {}".format(var, cols))
                        exit()
                    
            if do_image == True:
                for var in config["field_var"]:
                    if not var in cols:
                        logging.warning("Declared var {} for field not within data. Please use one of {}".format(field_var, cols))
                        exit()

        return config

    def front_width(self, config=None, cases_cfg=None):
        """
        Function that calculates the width of the front at half gap height for every timestep. Threshold is the minimum value to determine front and back positions.
        """

        if config == None or cases_cfg == None:
            logging.error("No cases.json or conf.json provided")
            exit()

        cases = config["cases"]
        cases_dir_path = config["cases_dir_path"]

        for cas in cases:

            widths = {}
            
            widths["time [s]"] = []
            widths["FWHM [mm]"] = []

            var = "concentration-fluid_c"
            if config["hpc_calculation"]:
                tmp_path = cases_dir_path[1:]
                tmp_path[0] = "/" + tmp_path[0]
                data_path = os.path.join(*tmp_path, cas, "widths_" + ".csv")
            else:
                data_path = os.path.join(*cases_dir_path, cas, "widths_" + ".csv")
            if os.path.exists(data_path):
                logging.info(f"Already created front for case {cas}. Continuing with next case")
                # continue

            logging.info(f"Creating reaction width data for case {cas}")

            times = get_cases(config, cas)
            times.sort()

            for idx, time in enumerate(times):

                config["plots"] = [time]
                config["field_var"] = [var]

                data = read_transient_data(config, cas)
                
                X, Y, res = self.convert2field(data[time], [var])
                # mirror across diagonal
                Vals = res[var]
                logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))
                Vals = np.rot90(np.fliplr(Vals))

                for id, ele in enumerate(X):
                    row_name = "w [mm] (h=" + str(ele * 1e3).replace(".", ",") + "mm)"
                    if not row_name in widths.keys():
                        widths[row_name]  = []
                    tmp_front = Vals[id]
                    threshold = max(tmp_front) * 1e-6
                    positions = np.array(np.where(tmp_front >= threshold))
                    if positions.size != 0:
                        front = positions[0][-1]
                        back = positions[0][0]
                        r_front = Y[front]
                        r_back = Y[back]
                        width = abs(r_front - r_back)*1e3 # mm
                    else:
                        width = math.nan
                    widths[row_name].append(width)
                
                widths["time [s]"].append(time)

                # calculating FWHM
                # logging.info(f"Calculating FWHM for time {time}")
                Vals_tmp = np.rot90(np.fliplr(Vals))
                avg_fluid_c_vals = []
                for idx, ele in enumerate(Y):
                    avg_fluid_c_vals.append(np.mean(Vals_tmp[idx]))
                threshold = max(avg_fluid_c_vals)/2
                positions = np.array(np.where(avg_fluid_c_vals >= threshold))
                if positions.size != 0:
                    front = positions[0][-1]
                    back = positions[0][0]
                    r_front = Y[front]
                    r_back = Y[back]
                    width = abs(r_front - r_back)*1e3 # mm
                else:
                    width = math.nan
                widths["FWHM [mm]"].append(width)

                logging.info(f"Added widths for time {time}s for case {cas}")

            df = pd.DataFrame(widths)
            df.to_csv(data_path, index=False)
            logging.info(f"Saved width data for case {cas}")

    def reaction_front(self, config=None, cases_conf=None):
        """
        Function calculates reaction front data and stores it in front.csv file within case directory
        """
        

        if config == None or cases_conf == None:
            logging.error("No cases.json or conf.json provided")
            exit()

        cases = config["cases"]
        cases_dir_path = config["cases_dir_path"]

        for cas in cases:
            
            front = {}
            var = "concentration-fluid_c"
            if config["hpc_calculation"]:
                tmp_path = cases_dir_path[1:]
                tmp_path[0] = "/" + tmp_path[0]
                data_path = os.path.join(*tmp_path, cas, "front_" + var + ".csv")
            else:
                data_path = os.path.join(*cases_dir_path, cas, "front_" + var + ".csv")
            
            if os.path.exists(data_path):
                logging.info(f"Already created front for case {cas}. Continuing with next case")
                continue

            logging.info(f"Creating reaction front data for case {cas}")
            
            times = get_cases(config, cas)
            times.sort()
            
            for id, time in enumerate(times):

                config["plots"] = [time]
                config["field_var"] = [var]

                data = read_transient_data(config, cas)
                
                X, Y, res = self.convert2field(data[time], [var])
                data_tmp = {}
                
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

                if "r [m]" not in front.keys():
                    front["r [m]"] = Y

                front["t= " + str(time) +" [s]"] = tmp

                pct = round((id+ 1)/len(times) * 100.0, 1)
                logging.info(f"Added time {time} to front data. {pct} % done.")

            df = pd.DataFrame(front)


            df.to_csv(data_path, index=False)
            logging.info(f"Saved front data for case {cas}")

    def formed_product(self, config, cases_cfg):
        """
        Functions that calculates the total amount of prodct formed for a case
        """

        if config == None or cases_cfg == None:
            logging.error("No cases.json or conf.json provided")
            exit()

        cases = config["cases"]
        cases_dir_path = config["cases_dir_path"]

        for cas in cases:

            prod = {}
            
            prod["time [s]"] = []
            prod["product [kmol]"] = []
            
            if config["hpc_calculation"]:
                tmp_path = cases_dir_path[1:]
                tmp_path[0] = "/" + tmp_path[0]
                data_path = os.path.join(*tmp_path, cas, "total_product_formed" + ".csv")
            else:
                data_path = os.path.join(*cases_dir_path, cas, "total_product_formed" + ".csv")
            if os.path.exists(data_path):
                logging.info(f"Already created formed product data for case {cas}. Continuing with next case")
                continue

            logging.info(f"Creating formed product data for case {cas}")

            times = get_cases(config, cas)
            times.sort()

            config["plots"] = times
            for idx, time in enumerate(times):
                config["plots"] = [time]
                data = read_transient_data(config, cas)[time]
                if not "cell-volume" in list(data.columns):
                    logging.warning(f"No Cell-Volume Data in case {cas}")
                    break

                prod["product [kmol]"].append(sum(data["cell-volume"] * data["concentration-fluid_c"]))
                prod["time [s]"].append(time)
                logging.info(f"Added product formed for time {time}s for case {cas}")

            if prod != {}:
                prod = pd.DataFrame(prod)
                prod.to_csv(data_path, index=False)



    def vel_field(self, config, cases_cfg):
        """
        Function that plots velocity filed with directional info
        """
        cases = config["cases"]
        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]
        

        for cas in cases:

            config = self.update_plot_cfg(config, cases_cfg, cas)
            plots = config["plots"]
            data = read_transient_data(config, cas)
            var = "velocity-field"

            if config["hpc_calculation"]:
                folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "vel_field", cas)
            else:
                path = sys.path[0]
                path = os.path.join(path, "assets")
                folder_path = os.path.join(path, var)
    
            if os.path.exists(folder_path) == False:
                os.makedirs(folder_path)

            image_name = "field_" + cas + "_" + var + "." + config["image_file_type"]
            image_path = os.path.join(folder_path, image_name)

            if os.path.exists(image_path) and config["ignore_exsisting"] == True:
                logging.info(f"{var} field for case {cas} already exsists.")
                continue

            logging.info(f"Creating {var} field {plots} for case {cas}")

            title = cas

            fig, axs = plt.subplots(len(plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(plots)))
            fig.suptitle(title, size=12)
            # axs = fig.add_subplot(len(plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(plots)))

            for idx, ele in enumerate(plots):
                elements = ['x-coordinate', 'y-coordinate', 'axial-velocity', 'radial-velocity']
                data_header = list(data[ele].columns)
                if all(elem in data_header for elem in elements) == False:
                    logging.error(f"Not all needed elements {elements} for plot {ele} in data for case {cas}")
                    continue

                X, Y, res = self.convert2field(data[ele], ['axial-velocity', 'radial-velocity'], vel_field=True)
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

                if len(plots) != 1:
                    if self.image_conf["set_custom_range"]:
                        # cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=self.image_conf["min"], vmax=self.image_conf["max"])
                        cax = axs[idx].quiver(y_final, x_final, v_r_vals, v_z_vals, scale=scl_val)
                    else:
                        cax = axs[idx].quiver(y_final, x_final, v_r_vals, v_z_vals, scale=scl_val)
                        # cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
                    # add axis description
                    if idx == len(plots)-1 :
                        axs[idx].set_xlabel("radius r [m]")
                    axs[idx].set_ylabel("height z [m]")
                    axs[idx].set_title("t = {}s".format(round(ele * cases_cfg[cas]["timestep"], 1)))
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
                    axs.set_title("t = {}s".format(round(ele * cases_cfg[cas]["timestep"], 1)))
                    # add colorbar
                    cbar = fig.colorbar(cax, ax=axs)
                    cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)
            
            plt.savefig(image_path)
            plt.close(fig)
            logging.info(f"saved image {image_name}.")



    def resi_plot(self, config):
        """
        Function that generates residuals plot for a case
        """
        # self.update_plot_cfg()
        cases_dir = config["cases_dir_path"]
        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]
        cases = config["cases"]

        for cas in cases:

            image_name = "residuals_" + cas + "." + config["plot_file_type"]
            if config["hpc_calculation"]:
                folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "residuals")
                
            else:
                path = sys.path[0]
                path = os.path.join(path, "assets")
                folder_path = os.path.join(path, "residuals", cas)
            
            
            if os.path.exists(folder_path) == False:
                os.makedirs(folder_path)
            
            image_path = os.path.join(folder_path, image_name)
            logging.debug(f"image path = {image_path}")

            if os.path.exists(image_path):
                logging.info(f"Residuals for case {cas} already created")
                continue
        
            logging.info(f"Creating residuals plot for case {cas}")

            if config["hpc_calculation"]:


                resid_file = glob.glob(r'*residuals.csv', root_dir=os.path.join(*hpc_cases_dir, cas))
                if resid_file == []:
                    logging.info(f"There is no residual file for case {cas}. Start processing .trn file")
                    parse_log_file(cas, config)
                resid_file = glob.glob(r'*residuals.csv', root_dir=os.path.join(*hpc_cases_dir, cas))
                data_path = os.path.join(*hpc_cases_dir, cas, resid_file[0])
                data = pd.read_csv(data_path)
            else:
                resid_file = glob.glob(r'*residuals.csv', root_dir=os.path.join(*cases_dir, cas))
                if resid_file == []:
                    logging.info(f"There is no residual file for case {cas}. Start processing .trn file")
                    parse_log_file(cas, config)
                resid_file = glob.glob(r'*residuals.csv', root_dir=os.path.join(*cases_dir, cas))
                data_path = os.path.join(*cases_dir, cas, resid_file[0])
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
            logging.info(f"saved image {image_name} to {image_path}.")

    def plot_prod(self, config, use_exp=True):
        """
        Function that plots the total amount of product
        """

        data = read_prod_data(config)

        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]

        i = 0
        legend = []

        cols = ["r", "g", "b", "y"]

        title = "total_product"
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,4.5))
        fig.suptitle(title)

        for cas in config["cases"]:
            logging.info(f"Creating total_product image for case {cas}")
            if config["hpc_calculation"]:
                folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "plots")
            else:
                path = sys.path[0]
                path = os.path.join(path, "assets", "plots", cas)
                folder_path = os.path.join(path)
    
            if os.path.exists(folder_path) == False:
                os.makedirs(folder_path)

            image_name = f"total_product" + "." + config["plot_file_type"]
            image_path = os.path.join(folder_path, image_name)

            if os.path.exists(image_path) and config["ignore_exsisting"]:
                logging.info(f"Already created total_product image for case {cas}")
                continue
            
            if len(config["cases"]) == 1:
                exps = ["ground", "PF"]          
                title = "total_product_" + cas
                fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,4.5))
                fig.suptitle(title)

                if use_exp:
                    image_name = f"total_product_sim_vs_{exps}" + "." + config["plot_file_type"]
                    for exp in exps:
                        image_path = os.path.join(folder_path, image_name)
                        cax = axs.plot(data["exp"][cas][exp][f"{exp} t [s]"], data["exp"][cas][exp][f"{exp} n_C [mol]"], f"{cols[i]}d")
                        i +=1
                        legend.append(f"exp_data {exp}")
                cax = axs.plot(data["sim"][cas]["time [s]"], data["sim"][cas]["product [kmol]"]*1e3*2*math.pi, f"{cols[i]}x")
                legend.append("sim_data")
                axs.set_xlim(0, 380)
                axs.legend(legend)
                axs.set_xlabel("time [s]")
                axs.set_ylabel("product [mol]")
                plt.savefig(image_path)
                plt.close(fig)
                logging.info(f"saved image {image_name} at {image_path}.")
                return
            else:
                if config["hpc_calculation"]:
                    folder_path = os.path.join(*hpc_cases_dir, "..", "Results" ,"plots")
                else:
                    path = sys.path[0]
                    path = os.path.join(path, "assets", "plots")
                    folder_path = os.path.join(path)
                    
        
                if os.path.exists(folder_path) == False:
                    os.makedirs(folder_path)

                image_path = os.path.join(folder_path, image_name)
                
            cax = axs.plot(data["sim"][cas]["time [s]"], data["sim"][cas]["product [kmol]"]*1e3*2*math.pi, f"{cols[i]}x")
            i += 1
            
            legend.append(cas)
        
        if os.path.exists(image_path) and config["ignore_exsisting"]:
            cases = config["cases"]
            logging.info(f"Already created total_product image for cases {cases}")
        else:    
            # axs.set_xlim(0, 380)
            axs.legend(legend)
            axs.set_xlabel("time [s]")
            axs.set_ylabel("product [mol]")
            plt.savefig(image_path)
            plt.close(fig)
            logging.info(f"saved image {image_name}.")

            
    def plot_front(self, config, front_data, max_data, exp_data):
        """
        Function to plot front position
        """

        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]

        for cas in list(front_data.keys()):
            logging.info(f"Creating front image for case {cas}")
            if config["hpc_calculation"]:
                folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "plots")
            else:
                path = sys.path[0]
                path = os.path.join(path, "assets", "plots", cas)
                folder_path = os.path.join(path)
    
            if os.path.exists(folder_path) == False:
                os.makedirs(folder_path)

            exps = ['ground', 'PF']

            for exp in exps:

                image_name = f"front_{exp}_{str(self.front_threshhold).replace('.', ',')}" + "." + config["plot_file_type"]
                image_path = os.path.join(folder_path, image_name)

                if os.path.exists(image_path) and config["ignore_exsisting"]:
                    logging.info(f"Already created front image for {exp} for case {cas}")
                    continue

                title = "front_" + cas

                fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,3.5))

                fig.suptitle(title)

                legend = ["sim_front_data", "sim_max_data", "exp_front_data", "exp_max_data"]

                
                tmp_exp = exp_data[cas]

                match exp:
                    case 'ground':
                        tmp_front_exp = exp_data[cas]['r_f_ground']
                        tmp_max_exp = exp_data[cas]['r_c_ground']
                        time_front_name = 'r_f_ground t [s]'
                        time_max_name = 'r_c_ground t [s]'

                    case 'PF':
                        tmp_front_exp = exp_data[cas]['r_f_PF']
                        tmp_max_exp = exp_data[cas]['r_c_PF']

                sim_data = pd.DataFrame(front_data[cas])
                front_exp_t = list(tmp_front_exp[f"r_f_{exp} t [s]"].round(0))
                front_exp_r = list(tmp_front_exp[f"r_f_{exp} r [mm]"])
                sim_matched = sim_data[sim_data["times [s]"].isin(front_exp_t)]

                if sim_matched.empty:
                    logging.warning(f"No matching times between experiment {exp} and simulation found.")

                mx_data = pd.DataFrame(max_data[cas])
                max_matched = mx_data[mx_data["times [s]"].isin(list(tmp_max_exp[f"r_c_{exp} t [s]"].round(0)))]
                max_exp_t = list(tmp_max_exp[f"r_c_{exp} t [s]"].round(0))
                max_exp_r = list(tmp_max_exp[f"r_c_{exp} r [mm]"])

                cax = axs.plot(sim_matched["times [s]"], sim_matched["r_s [m]"]*1e3, "bd")
                cax = axs.plot(max_matched["times [s]"], max_matched["r_s [m]"]*1e3, "yd")

                cax = axs.plot(front_exp_t, front_exp_r, "rx")
                cax = axs.plot(max_exp_t, max_exp_r, "gx")

                axs.set_xlabel("time [s]")
                axs.set_ylabel("radius [mm]")
                axs.set_xlim(225, 380)
                axs.legend(legend)

                plt.savefig(image_path)
                plt.close(fig)
                logging.info(f"saved image {image_name}.")


    def multi_plot(self, config=None, cases_cfg=None):
        """
        Function that plots variables over the radius for multiple timesteps
        """

        cases = config["cases"]
        plot_vars = config["plot_vars"]

        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]
        
        for cas in cases:

            config = self.update_plot_cfg(config, cases_cfg, cas)
            plots = config["plots"]
            plot_cfg = config["plot_conf"]
            one_plot = config["one_plot"]

            logging.info(f"Creating {plot_vars} plot {plots} for case {cas}")

            if config["hpc_calculation"]:
                folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "plots", "animation_images")
            else:
                path = sys.path[0]
                path = os.path.join(path, "assets", "plots", cas)
                folder_path = os.path.join(path)
    
            if os.path.exists(folder_path) == False:
                os.makedirs(folder_path)


            if "plot_file_name" in config.keys():
                image_name = config["plot_file_name"]
                if config["hpc_calculation"] == False:
                    path = sys.path[0]
                    folder_path = os.path.join(path, "assets" , "plots", "animation_images", cas)
                    image_name = config["plot_file_name"] + "." + config["plot_file_type"]
            else:
                image_name = "plot_" + cas + "_" + "_".join(plot_vars) + "." + config["plot_file_type"]
                if config["hpc_calculation"]:
                    folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "plots")
            image_path = os.path.join(folder_path, image_name)

            if os.path.exists(image_path) and config["ignore_exsisting"]:
                logging.info(f"{plot_vars} plot for case {cas} already created")
                continue
            
            data = read_transient_data(config, cas)
            if one_plot == False:
                
                title = ""

                fig, axs = plt.subplots(len(plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(plots)))

                fig.suptitle(title)

                legend = []

                for var in list(plot_cfg["legend"]):
                    legend.append(plot_cfg["legend"][var])
                

                for idx, ele in enumerate(plots):

                    X, Y, res = self.convert2field(data[ele], plot_vars)
                    data_tmp = {}

                    for var in plot_vars:
                        
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

                    if len(plots) != 1:

                        for tmp_var in data_tmp.keys():
                            col = plot_cfg["colors"][tmp_var]
                            cax = axs[idx].plot(x_tmp, data_tmp[tmp_var], color=col)
                            # add axis description
                        if idx == len(plots)-1 :
                            axs[idx].set_xlabel("radius r [m]")
                        axs[idx].set_ylabel("height z [m]")
                        axs[idx].legend(legend)
                        export_times = cases_cfg[cas]["export_times"]
                        if export_times != "flow_time":
                            axs[idx].set_title("t = {}s".format(round(ele * cases_cfg["timestep"],1)))
                        else:
                            axs[idx].set_title("t = {}s".format(ele))


                    else:
                        export_times = cases_cfg[cas]["export_times"]
                        for tmp_var in data_tmp.keys():
                            col = plot_cfg["colors"][tmp_var]
                            cax = axs.plot(x_tmp, data_tmp[tmp_var], color=col)
                            # add axis description

                        axs.set_xlabel("radius r [m]")
                        axs.set_ylabel("height z [m]")
                        axs.legend(legend)
                        if export_times != "flow_time":
                            axs.set_title("t = {}s".format(round(ele * cases_cfg["timestep"],1)))
                        else:
                            axs.set_title("t = {}s".format(ele))

            else:
                fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(plots)))
                
                l_conf = []
                for idx, ele in enumerate(plots):

                    X, Y, res = self.convert2field(data[ele], plot_vars)
                    data_tmp = {}
                    
                    for var in plot_vars:
                        
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
                    
                    if len(plots) != 1:
                        
                        for tmp_var in data_tmp.keys():
                            l_style = plot_cfg["linestyles"]["t" + str(idx + 1)]
                            col = plot_cfg["colors"][tmp_var]
                            cax = axs.plot(x_tmp, data_tmp[tmp_var], linestyle=l_style, color=col)
                            if self.export_times != "flow_time":
                                l_conf.append(plot_cfg["legend"][tmp_var] + ", t={}s".format(round(ele * cases_cfg["timestep"],1)))
                            else:
                                l_conf.append(plot_cfg["legend"][tmp_var] + ", t={}s".format(ele))

                            # add axis description
                        if idx == len(plots)-1 :
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
    
    def multi_field(self, config=None, cases_cfg=None):
        """
        Function that shows ansys fields for multiple timesteps within one image
        """

        cases = config["cases"]
        field_var = config["field_var"]
        field_var_tmp = config["field_var"]
        plots_tmp = config["plots"]
        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]
        
        image_conf = config["image_conf"]
        for cas in cases:
            config["plots"] = plots_tmp
            field_var = field_var_tmp[:]
            export_times = cases_cfg[cas]["export_times"]
            case_conf = cases_cfg[cas]

            if field_var != []:
                config = self.update_plot_cfg(config, cases_cfg, cas)
                data = read_transient_data(config, cas)
                plots = config["plots"]

            for var in field_var:

                if 'velocity-field' in field_var:
                    self.vel_field(config, cases_cfg)
                    field_var.remove('velocity-field')
        
                # var = [var]
                logging.info(f"Creating {var} field {plots} for case {cas}")

                if config["hpc_calculation"]:
                    folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "fields", "animation_images", var)
                else:
                    path = sys.path[0]
                    path = os.path.join(path, "assets", "fields", cas)
                    folder_path = os.path.join(path, var)
        
                if os.path.exists(folder_path) == False:
                    os.makedirs(folder_path)

                if "image_file_name" in config.keys():
                    image_name = config["image_file_name"]
                    
                    if config["hpc_calculation"] == False:
                        path = sys.path[0]
                        folder_path = os.path.join(path, "assets" , "fields", "animation_images", cas ,var)
                        image_name = config["image_file_name"] + "." + config["image_file_type"]

                else:
                    if config["hpc_calculation"]:
                        folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "fields")
                    image_name = "field_" + cas + "_" + var + "." + config["image_file_type"]

                image_path = os.path.join(folder_path, image_name)

                if os.path.exists(image_path) and config["ignore_exsisting"] == True:
                    logging.info(f"{var} field for case {cas} already exsists.")
                    continue

                title = cas

                fig, axs = plt.subplots(len(plots), 1, sharex=True, sharey=True, figsize=(6.5, 2.0*len(plots)+2.5))
                fig.suptitle(title, size=12)
                # axs = fig.add_subplot(len(plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(plots)))

                for idx, ele in enumerate(plots):

                    X, Y, res = self.convert2field(data[ele], [var])
                    # mirror across diagonal
                    x_tmp = Y
                    y_tmp = X
                    Vals = res[var]
                    logging.debug("Size X = {}, Size Y = {}, Size Vals = {}".format(len(X), len(Y), Vals.shape))
                    Vals = np.rot90(np.fliplr(Vals))            
                    # plot n

                    if len(plots) != 1:
                        if image_conf["set_custom_range"]:
                            cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=image_conf["min"], vmax=image_conf["max"])
                        else:
                            cax = axs[idx].pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))
                        # add axis description
                        if idx == len(plots)-1 :
                            axs[idx].set_xlabel("radius r [m]")
                        axs[idx].set_ylabel("height z [m]")
                        if export_times != "flow_time":
                            axs[idx].set_title("t = {}s".format(round(ele * case_conf["timestep"], 1)))
                        else:
                            axs[idx].set_title("t = {}s".format(ele))
                        
                        # add colorbar
                        cbar = fig.colorbar(cax, ax=axs[idx])
                        cbar.set_label(config["c_bar"], rotation=90, labelpad=7)

                    else:
                        if image_conf["set_custom_range"]:
                            cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'), vmin=image_conf["min"], vmax=image_conf["max"])
                        else:
                            cax = axs.pcolormesh(x_tmp, y_tmp, Vals, shading='nearest', cmap=plt.cm.get_cmap('jet'))

                        axs.set_xlabel("radius r [m]")
                        axs.set_ylabel("height z [m]")
                        if export_times != "flow_time":
                            axs.set_title("t = {}s".format(round(ele * case_conf["timestep"], 1)))
                        else:
                            axs.set_title("t = {}s".format(ele))

                        # add colorbar
                        cbar = fig.colorbar(cax, ax=axs)
                        cbar.set_label(config["c_bar"], rotation=90, labelpad=7)
                
                plt.savefig(image_path)
                plt.close(fig)
                logging.info(f"saved image {image_name}.")

    def setup_journal(self, config, cases_cfg, mode, exit=True, update_exsisting=False):
        """
        Function that creates journal files
        """

        journal_path = os.path.join(sys.path[0], "..", "ansys", "journals")

        files = glob.glob(r'**/*.jou', root_dir=journal_path, recursive=True)
        files.remove("gui_template.jou")
        files.remove("cmd_template.jou")
        files.remove("cmd_creation_template.jou")
        for file in files:
            os.remove(os.path.join(journal_path, file))
        logging.info(f"Removed {len(files)} journals")

        build_journal(config, cases_cfg, exit, mode, update_exsisting)


    def delete_gif_imgs(self, config):

        """
        Function that deletes all images used to create the gifs and videos
        """
        field_var = config["field_var"]
        path = sys.path[0]

        for var in field_var:
            img_path = os.path.join(path, "assets", var)

            gifs = glob.glob('*_gif*', root_dir=img_path)

            if gifs == []:
                logging.info("No gif Images to delete")
            else:
                
                for ele in gifs:
                    os.remove(os.path.join(img_path, ele))
                logging.info(f"Deleted {len(gifs)} image files")

    def create_gif(self, config, cases_conf):
        
        """
        Function that creates missing Images if necessary and then creates .gif out of all obtained images
        """
        
        cases = config["cases"]
        hpc_cases_dir = config["cases_dir_path"][1:]
        hpc_cases_dir[0] = "/" + hpc_cases_dir[0]

        for cas in cases:
            
            config = self.update_plot_cfg(config, cases_conf, cas)
            field_vars = config["field_var"]
            gif_conf = config["gif_conf"]
            config["cases"] = [cas]

            for var in config["animation_vars"]:

                if config["hpc_calculation"]:
                    img_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "fields", "animation_images", var)
                    plot_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "plots", "animation_images")
                    folder_path = os.path.join(*hpc_cases_dir, cas, *config["hpc_results_path"], "animations", var)
                else:
                    path = sys.path[0]
                    img_path = os.path.join(path, "assets" , "fields", "animation_images", cas, var)
                    plot_path = os.path.join(path, "assets", "plots", "animation_images", cas)    
                    folder_path = os.path.join(path, "animations", var)

                if os.path.exists(folder_path) == False:
                    os.makedirs(folder_path)
                if os.path.exists(img_path) == False:
                    os.makedirs(img_path)
                if os.path.exists(plot_path) == False:
                    os.makedirs(plot_path)
                
                logging.info("Creating images for .gif ...")
                # create plots

                cases_tmp = get_cases(config, cas)

                # raw_cases = list(range(int(gif_conf["cases_tmp"]["start"]), int(gif_conf["cases_tmp"]["end"]) + 1, int(gif_conf["cases_tmp"]["step"])))
                start = float(gif_conf["cases"]["start"])
                end = float(gif_conf["cases"]["end"])
                steps = int((gif_conf["cases"]["end"] - gif_conf["cases"]["start"])/gif_conf["cases"]["step"] +1)
                raw_cases = list(np.linspace(start, end, steps))

                if raw_cases != []:
                    start_end = get_closest_plots(config, cases_conf, cas)
                    cases_tmp = list(set(start_end))
                    cases_tmp.sort()

                digits = len(str(max(cases_tmp)))
                plot_images = []
                field_images = []

                # Check for exsisting images
                raw_plots = []
                raw_imgs = []

                for tmp_cas in raw_cases:
                    config["plots"] = [tmp_cas]

                    plot_name = "_".join(["plot_gif", cas, f"{tmp_cas:0>{digits}}".replace(".", ",")])
                    
                    config["plot_file_name"] = plot_name
                    field_name = "_".join(["img_gif", cas, f"{tmp_cas:0>{digits}}".replace(".", ",")])
                    
                    config["image_file_name"] = field_name

                    if gif_conf["gif_plot"] and os.path.exists(os.path.join(plot_path, plot_name + ".png")) == False:
                        raw_plots.append(tmp_cas)
                    else:
                        
                        logging.info(f"Plot {plot_name} already created")
                    plot_images.append(plot_name + ".png")

                    if gif_conf["gif_image"] and os.path.exists(os.path.join(img_path, field_name + ".png")) == False:
                        raw_imgs.append(tmp_cas)
                    else:
                        logging.info(f"Field {field_name} already created")
                    field_images.append(field_name + ".png")

                for tmp_cas in raw_plots:
                    config["plots"] = [tmp_cas]
                    plot_name = "_".join(["plot_gif", cas, f"{tmp_cas:0>{digits}}".replace(".", ",")])
                    config["plot_file_name"] = plot_name
                    config["plot_vars"] = config["animation_plot_vars"]
                    self.multi_plot(config, cases_conf)

                for tmp_cas in raw_imgs:

                    # config["plots"] = [tmp_cas * tmp_case_conf["timestep"]]
                    config["plots"] = [tmp_cas]
                    field_name = "_".join(["img_gif", cas, f"{tmp_cas:0>{digits}}".replace(".", ",")])
                    config["image_file_name"] = field_name
                    config["field_var"] = [var]
                    self.multi_field(config, cases_conf)

                # create Images
                if gif_conf["gif_plot"]:

                    plot_vars = config["plot_vars"]
                    gif_name = "_".join([cas, *plot_vars, "plot"]) + ".gif"
                    gif_path = os.path.join(folder_path, gif_name)

                    video_name = "_".join([cas, *plot_vars, "plot"]) + ".avi"
                    video_path = os.path.join(folder_path, video_name)

                    if os.path.exists(gif_path):
                        logging.info(f"Deleting existing gif {gif_name}")
                    if os.path.exists(video_path):
                        logging.info(f"Deleting existing video {video_name}")
                    
                    logging.info(f"Plot imgs 1 to 3 = {plot_images[0:3]}, Amount={len(plot_images)}")
                    logging.info(f"Plot path={plot_path}")
                    imgs = None
                    imgs = (Image.open(os.path.join(plot_path, f)) for f in plot_images)
                    img = next(imgs)  # extract first image from iterator
                    img.save(gif_path, format="GIF", append_images=imgs,
                            save_all=True, duration=gif_conf["frame_duration"], loop=gif_conf["loop"])
                    logging.info(f"Created gif for variable {plot_vars} for cas {cas}")

                    if gif_conf["videos"]:

                        logging.info(f"Creating plot video for cas {cas}")
                    
                        images = plot_images
                        frame_path = os.path.join(plot_path, images[0])
                        logging.debug(f"Image Path={frame_path}")
                        frame = cv2.imread(frame_path)


                        height, width, layers = frame.shape
                        fps = 1000/gif_conf["frame_duration"]
                        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
                        

                        for image in images:
                            img = cv2.imread(os.path.join(plot_path, image))
                            out.write(img)

                        cv2.destroyAllWindows()
                        out.release()
                        logging.info(f"Video created for var {plot_vars} for cas {cas}.")


                if gif_conf["gif_image"]:

                    logging.info(f"Creating field video for cas {cas}")
                    
                    gif_name = "_".join([cas, var, "image"]) + ".gif"
                    video_name = "_".join([cas , var, "image"]) + ".avi"
                    
                    gif_path = os.path.join(folder_path, gif_name)
                    video_path = os.path.join(folder_path, video_name)

                    if os.path.exists(gif_path):
                        logging.info(f"Deleting existing gif {gif_name}")
                    if os.path.exists(video_path):
                        logging.info(f"Deleting existing video {video_name}")

                    logging.info(f"Field imgs 1 to 3 = {field_images[0:3]}, Amount={len(field_images)}")
                    logging.info(f"Field path {img_path}")

                    tmp_imgs = field_images

                    if tmp_imgs == []:
                        logging.error(f"No images for gif found at {img_path}")
                        exit()
                    imgs = None
                    imgs = (Image.open(os.path.join(img_path,f)) for f in field_images)
                    img = next(imgs)  # extract first image from iterator
                    img.save(gif_path, format="GIF", append_images=imgs,
                            save_all=True, duration=gif_conf["frame_duration"], loop=gif_conf["loop"])
                    logging.info(f"Created gif for variable {var} for cas {cas}")

                    if gif_conf["videos"]:
                        
                        # images = sorted(glob.glob('img_gif_*png', root_dir=img_path))
                        images = field_images
                        logging.debug(f"Image Path={img_path}")
                        logging.debug(f"Images={field_images}")

                        frame = cv2.imread(os.path.join(img_path, images[0]))

                        height, width, layers = frame.shape
                        fps = 1000/gif_conf["frame_duration"]
                        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
                        

                        for image in images:
                            img = cv2.imread(os.path.join(img_path, image))
                            out.write(img)

                        cv2.destroyAllWindows()
                        out.release()
                        logging.info(f"Video created for var {var} for cas {cas}.")

                if gif_conf["keep_images"] == False:
                    self.delete_gif_imgs(config)    

def parse_log_file(case, config):

    hpc_cases_dir = config["cases_dir_path"][1:]
    hpc_cases_dir[0] = "/" + hpc_cases_dir[0]
    
    if config["hpc_calculation"]:
        case_path = os.path.join(*hpc_cases_dir, case)

    else:
        case_path = os.path.join(*config["cases_dir_path"], case)


    files = glob.glob('*.trn', root_dir=case_path, recursive=False)
    if files == []:
        files = glob.glob('**/*.trn', root_dir=case_path, recursive=True)
        
    if files == []:
        logging.warning(f"No .trn file found")
        exit()
        # return logs

    log_path = os.path.join(case_path, files[0])
    parse_logs(log_path, journal=False, case=case)

def do_plots():

    # local machine
    path = os.path.join(sys.path[0], "..", "ansys", "cases.json")
    if os.path.exists(path) == False:
        path = os.path.join(sys.path[0], "cases.json")

    with open(path) as f:
        cases_cfg = json.load(f)

    cfg_path = os.path.join(sys.path[0], "conf.json")

    with open(cfg_path) as f:
        config = json.load(f)

    field = flowfield(config, cases_cfg)
    
    if config["create_image"]:
        field.multi_field(config, cases_cfg)
    
    if config["create_plot"]:
        field.multi_plot(config, cases_cfg)

    if config["create_gif"]:
        field.create_gif(config, cases_cfg)

    if config["create_resi_plot"]:
        field.resi_plot(config)

    if config["create_front"]:
        field.reaction_front(config, cases_cfg)
    
    if config["create_widths"]:
        field.front_width(config, cases_cfg)
    
    if config["create_prod"]:
        field.formed_product(config, cases_cfg)

    if config["plot_front"]:
        field.front_threshhold = 1e-6
        data = read_front_data(config)
        front = calc_front(data["sim"], field.front_threshhold, False)
        mx = calc_front(data["sim"], 0.0, True)
        field.plot_front(config, front, mx, data["exp"])

    if config["plot_prod"]:
        field.plot_prod(config)
if __name__ == "__main__":

    do_plots()