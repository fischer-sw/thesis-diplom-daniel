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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")


class flowfield:

    def __init__(self, config):

        self.config = config
        check_data_format()

    def convert2field(self, data, vars):
        """
        Function that converts list of x,y,<var> to matrix of dimensions x,y with <var> as values inside
        """
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

        """
        Function that checks plot configuration for defined vars
        """

        if len(self.config["plot_vars"]) != len(conf["linestyles"]):
            logging.info("Plot vars and colordefinitions don't match. Please check if all plotted variables have a respective color defined")
            exit()

        if len(self.plots) != len(conf["linestyles"]):
            logging.info("Please define a linestyle for all times that are plotted under linestyles")
            exit()

    def update_plot_cfg(self):

        """
        Function that updates plot config and returns plot configuration (return value is used for checking cfg in one_plot case)
        """

        plot_cfg = self.config["plot_conf"]
        self.plots = self.config["plots"]

        self.case = self.config["case"]
        self.field_var = self.config["field_var"]
        self.plots = self.config["plots"]
        self.one_plot = self.config["one_plot"]
        self.plot_vars = self.config["plot_vars"]
        self.do_plots = self.config["create_plot"]
        self.do_image = self.config["create_image"]
        self.image_conf = self.config["image_conf"]
        self.gif_conf = self.config["gif_conf"]

        self.case_conf = get_case_info(self.config["cases_dir_path"], self.case)
        
        if self.plots == []:
            logging.info("Plots are not set. Creating default ones ...")
            self.plots = get_default_cases(self.config["cases_dir_path"], self.case)

        else:
            self.plots = get_colsest_plots(np.array(self.plots)/self.case_conf["timestep"], self.case_conf["timestep"] ,self.config["cases_dir_path"], self.case)
        
        self.data = read_transient_data(self.config["cases_dir_path"], self.case, self.plots)

        cols = list(self.data[list(self.data.keys())[0]].columns)
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

    def multi_plot(self):
        """
        Function that plots variables over the radius for multiple timesteps
        """


        plot_cfg = self.update_plot_cfg()
    
        logging.info(f"Creating {self.plot_vars} field {self.plots} for case {self.case}")


        if self.one_plot == False:

            fig, axs = plt.subplots(len(self.plots), 1, sharex=True, sharey=True, figsize=(6.5,2.4*len(self.plots)))

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
                    axs[idx].set_title("t = {}s".format(round(ele * self.case_conf["timestep"],1)))
                    

                else:
                    for tmp_var in data_tmp.keys():
                        col = plot_cfg["colors"][tmp_var]
                        cax = axs.plot(x_tmp, data_tmp[tmp_var], color=col)
                        # add axis description

                    axs.set_xlabel("radius r [m]")
                    axs.set_ylabel("height z [m]")
                    axs.legend(legend)
                    axs.set_title("t = {}s".format(ele* self.case_conf["timestep"]))

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
                        l_conf.append(plot_cfg["legend"][tmp_var] + ", t={}s".format(round(ele * self.case_conf["timestep"],1)))
                        
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
        path = os.path.join(path, "assets")
        sub_path = os.path.join(path, "transient")
        if os.path.exists(path) == False:
            os.mkdir(path)
        if os.path.exists(sub_path) == False:
            os.mkdir(sub_path)

        image_name = self.config["plot_file_name"] + "." + self.config["plot_file_type"]
        image_path = os.path.join(sub_path, image_name)

        plt.savefig(image_path)
        plt.close(fig)
    
    def multi_field(self):
        """
        Function that shows ansys fields for multiple timesteps within one image
        """
        
        self.update_plot_cfg()
        

        logging.info(f"Creating {self.field_var} field {self.plots} for case {self.case}")

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
                axs.set_title("t = {}s".format(round(ele * self.case_conf["timestep"], 1)))
                # add colorbar
                cbar = fig.colorbar(cax, ax=axs)
                cbar.set_label(self.config["c_bar"], rotation=90, labelpad=7)
            
        path = sys.path[0]
        path = os.path.join(path, "assets")
        sub_path = os.path.join(path, "transient")
        if os.path.exists(path) == False:
            os.mkdir(path)
        if os.path.exists(sub_path) == False:
            os.mkdir(sub_path)

        image_name = self.config["image_file_name"] + "." + self.config["image_file_type"]
        image_path = os.path.join(sub_path, image_name)

        
        plt.savefig(image_path)
        plt.close(fig)


    def delete_gif_imgs(self):

        path = sys.path[0]
        img_path = os.path.join(path, "assets", "transient")

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

        path = sys.path[0]
        img_path = os.path.join(path, "assets", "transient")

        if self.gif_conf["new"]:
            self.delete_gif_imgs()

        logging.info("Creating images for .gif ...")

        path = os.path.join(path, "gifs")
        if os.path.exists(path) == False:
            os.mkdir(path)

        # create plots

        cases = get_cases(self.config["cases_dir_path"], self.case)
        digits = len(str(int(max(cases))))

        for cas in cases:
            self.config["plots"] = [cas * self.case_conf["timestep"]]
            plot_name = "_".join(["plot_gif", self.case, f"{int(cas):0{digits}d}"])
            self.config["plot_file_name"] = plot_name
            field_name = "_".join(["img_gif", self.case, f"{int(cas):0{digits}d}"])
            self.config["image_file_name"] = field_name
            if self.gif_conf["gif_plot"]:
                if os.path.exists(os.path.join(img_path, plot_name + ".png")) == False:
                    self.multi_plot()
                else:
                     logging.debug(f"Image {plot_name} already exsists")

            if self.gif_conf["gif_image"]:
                if os.path.exists(os.path.join(img_path, field_name + ".png")) == False:
                    self.multi_field()
                else:
                    logging.debug(f"Image {field_name} already exsists")

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
           
            imgs = (Image.open(os.path.join(img_path,f)) for f in sorted(glob.glob('*plot_gif_*png', root_dir=img_path)))
            img = next(imgs)  # extract first image from iterator
            img.save(gif_path, format="GIF", append_images=imgs,
                    save_all=True, duration=self.gif_conf["frame_duration"], loop=self.gif_conf["loop"])

            if self.gif_conf["videos"]:

                logging.info(f"Creating plot video for case {self.case}")
                
                images = sorted(glob.glob('plot_gif_*png', root_dir=img_path))

                frame = cv2.imread(os.path.join(img_path, images[0]))

                height, width, layers = frame.shape
                fps = 1000/self.gif_conf["frame_duration"]
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
                

                for image in images:
                    img = cv2.imread(os.path.join(img_path, image))
                    out.write(img)

                cv2.destroyAllWindows()
                out.release()    

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

            imgs = (Image.open(os.path.join(img_path,f)) for f in sorted(glob.glob('*img_gif_*png', root_dir=img_path)))
            img = next(imgs)  # extract first image from iterator
            img.save(gif_path, format="GIF", append_images=imgs,
                    save_all=True, duration=self.gif_conf["frame_duration"], loop=self.gif_conf["loop"])

            if self.gif_conf["videos"]:
                
                images = sorted(glob.glob('img_gif_*png', root_dir=img_path))

                frame = cv2.imread(os.path.join(img_path, images[0]))

                height, width, layers = frame.shape
                fps = 1000/self.gif_conf["frame_duration"]
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
                

                for image in images:
                    img = cv2.imread(os.path.join(img_path, image))
                    out.write(img)

                cv2.destroyAllWindows()
                out.release()

        if self.gif_conf["keep_images"] == False:
            self.delete_gif_imgs()    
        
if __name__ == "__main__":

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

    
    