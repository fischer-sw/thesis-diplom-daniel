import math
from tkinter import image_names
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from ansys_utils import *

class flowfield:
    def __init__(self, pars):
        self.pars = self.check_pars( pars)
        self.flowfield_data, self.flowfield = self.setup_flowfield()


    def check_pars(self, pars):

        """
        Function that checks that all parameters needed to calculate flowfield are set

        example = {

            "flowrate" : 3,

            "z_res" : {
                "length" : 1,
                "devisions" : 10
            },

            "r_res" : {
                "length" : 0.5,
                "devisions" : 5
            }
        }
        """


        missing = []
        error = False

        for key in pars.keys():
            if not key in ["flowrate", "r_res", "z_res"]:
                missing.append(key)

        if missing != []:
            error = True
            print("Parameters in pars {} dict missing").format(missing)

        # if pars["r_res"]["length"]%pars["r_res"]["devisions"] != 0:
        #     error = True
        #     print("Radius length and devisions don't match")

        # if pars["z_res"]["length"]%pars["z_res"]["devisions"] != 0:
        #     error = True
        #     print("Height length and devisions don't match")

        if error == False:
            return pars
        else:
            exit()

    def setup_flowfield(self):
        """
        Function that creates flowfield array
        """

        r_elements = int(self.pars["r_res"]["devisions"])

        z_elements = int(self.pars["z_res"]["devisions"])

        field = []

        v_field = np.zeros((z_elements+1, r_elements))

        for i in range(z_elements+1):

            line = []

            for j in range(1, r_elements):

                r_len = self.pars["r_res"]["length"]
                z_len = self.pars["z_res"]["length"]

                r_divs = self.pars["r_res"]["devisions"]
                z_divs = self.pars["z_res"]["devisions"]

                r_div_len = r_len/r_divs
                z_div_len = z_len/z_divs

                r = j * r_div_len
                z_val = i * z_div_len
                z = z_val - self.pars["z_res"]["length"]/2

                v_r = self.calc_v(r, z, self.pars)

                tmp = {
                    "r" : r,
                    "z" : z,
                    "v" : v_r
                }

                v_field[i,j] = v_r

                line.append(tmp)

            field.append(line)

        # v_field = np.delete(v_field, [0] , 0)

        return field, v_field

    def show_field(self, x, y, data):
        """
        Function that shows calculation results as image
        """

        fig = plt.figure(figsize=(4,10))
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(x, y, data, shading='nearest', cmap=plt.cm.get_cmap('jet'))


        # set correct x_ticks labels
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        test = ax.get_xticklabels()
        test1 = ax.get_yticks()

        x_div_len = self.pars["r_res"]["length"]/len(x_ticks)

        new_x_ticks = []

        for x in range(len(x_ticks)):
            new_x_ticks.append(x * x_div_len)

        # ax.set_xticklabels(new_x_ticks)

        y_ticks = ax.get_yticks()

        z_div_len = self.pars["z_res"]["length"]/len(y_ticks)

        new_y_ticks = []

        for z in range(len(y_ticks)):
            new_y_ticks.append(round(self.pars["z_res"]["length"]/2 - z * z_div_len, 2))

        # ax.set_yticklabels(new_y_ticks)

        # add axis description
        ax.set_xlabel("radius r [m]")
        ax.set_ylabel("height z [m]")

        # add colorbar


        cbar = fig.colorbar(cax)
        cbar.set_label('velocity [m/s]', rotation=90)

        path = sys.path[0]

        path = os.path.join(path, "Images")

        if os.path.exists(path) == False:
            os.mkdir(path)

        image_name = "Test.png"

        image_path = os.path.join(path, image_name)

        plt.savefig(image_path)

    def calc_v(self, r, z, pars):
        """
        Function that calculates velocity v dependent on positon (r = radius, z = height, pars = parameters)
        """

        Q_rate = pars["flowrate"]
        h = pars["z_res"]["length"]

        v_max = (3*Q_rate)/(4*math.pi*h*r)

        return v_max * (1 - ((4 * z**2)/(h**2)))


if __name__ == "__main__":

    example = {

        "flowrate" : 1000,

        "z_res" : {
            "length" : 1,
            "devisions" : 60
        },

        "r_res" : {
            "length" : 1,
            "devisions" : 80
        }
    }

    test = flowfield(example)

    raw_data = read_csv('export.csv')
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

    #test.show_field(X, Y, V)

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
    
    #test.show_field(X, Y, Vr)

    d = V - Vr
    test.show_field(X, Y, d)

    #shape = np.shape(test.flowfield)
    #print(test.flowfield[round(shape[1]/2)])
