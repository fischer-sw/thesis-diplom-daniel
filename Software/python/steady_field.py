import math
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

from ansys_utils import *

class flowfield:
    def __init__(self, pars):
        self.pars = self.check_pars( pars)

    def check_pars(self, pars):

        """
        Function that checks that all parameters needed to calculate flowfield are set

        example = {

            "input_vel" : 0.1,

            "geometry" : {
                "r" : {
                    "min" : 0.001,
                    "max" : 0.101

                },
                "z" : {
                    "min" : 0,
                    "max" : 0.01
                }
            }
        }
        """


        missing = []
        error = False

        for key in pars.keys():
            if not key in ["input_vel", "geometry"]:
                missing.append(key)

        if missing != []:
            error = True
            print("Parameters {} in pars dict missing".format(missing))

        if error == False:
            return pars
        else:
            exit()

    def show_field(self, x, y, data, conf):
        """
        Function that shows calculation results as image
        """

        # mirror across diagonal
        x_tmp = y
        y_tmp = x
        data = np.rot90(np.fliplr(data))

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(x_tmp, y_tmp, data, shading='nearest', cmap=plt.cm.get_cmap('jet'))

        # add axis description
        ax.set_xlabel("radius r [m]")
        ax.set_ylabel("height z [m]")

        # add colorbar
        cbar = fig.colorbar(cax)
        cbar.set_label(conf["c_bar"], rotation=90, labelpad=7)

        path = sys.path[0]
        path = os.path.join(path, "Images")

        if os.path.exists(path) == False:
            os.mkdir(path)

        image_name = conf["name"] + ".png"
        image_path = os.path.join(path, image_name)

        plt.savefig(image_path)

    def calc_v(self, r, z, pars):
        """
        Function that calculates velocity v dependent on positon (r = radius, z = height, pars = parameters)
        """

        h = pars["geometry"]["z"]["max"] - pars["geometry"]["z"]["min"]

        Q_rate = math.pi * pars["geometry"]["r"]["min"] * 2 * h * pars["input_vel"]

        v_max = (3*Q_rate)/(4*math.pi*h*r)

        return v_max * (1 - ((4 * z**2)/(h**2)))


if __name__ == "__main__":

    example = {

        "input_vel" : 0.1,

        "geometry" : {
            "r" : {
                "min" : 0.001,
                "max" : 0.101

            },
            "z" : {
                "min" : 0,
                "max" : 0.01
            }
        }
    }

    test = flowfield(example)

    raw_data = read_steady_data('export.csv')
    X = sorted(set(raw_data['X [ m ]']))
    Y = sorted(set(raw_data['Y [ m ]']))
    V = np.zeros((len(Y), len(X)))

    for i in range(len(raw_data['Velocity [ m s^-1 ]'])):
        x = raw_data['X [ m ]'][i]
        y = raw_data['Y [ m ]'][i]
        v = raw_data['Velocity [ m s^-1 ]'][i]
        ix = X.index(x)
        iy = Y.index(y)
        V[iy,ix] = v

    config = {
        "name" : "ansys_field",
        "c_bar" : "Velocity [m/s]"
    }

    test.show_field(X, Y, V, config)

    Vr = np.zeros((len(Y), len(X)))
    for i in range(len(raw_data['Velocity [ m s^-1 ]'])):
        x = raw_data['X [ m ]'][i]
        y = raw_data['Y [ m ]'][i]
        r = y
        z = x - (X[-1] - X[0]) / 2
        vr = test.calc_v(r, z, test.pars)
        ix = X.index(x)
        iy = Y.index(y)
        Vr[iy,ix] = vr
    
    config = {
        "name" : "calc_field",
        "c_bar" : "Velocity [m/s]"
    }

    test.show_field(X, Y, Vr, config)

    # d = (V - Vr)/example["input_vel"] * 100
    d = (V - Vr)/0.1 * 100

    config = {
        "name" : "diff_field",
        "c_bar" : "Error [%]"
    }

    test.show_field(X, Y, d, config)
