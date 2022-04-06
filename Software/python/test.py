import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        if pars["r_res"]["length"]%pars["r_res"]["devisions"] != 0:
            error = True
            print("Radius length and devisions don't match")

        if pars["z_res"]["length"]%pars["z_res"]["devisions"] != 0:
            error = True
            print("Height length and devisions don't match")

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

        v_field = np.zeros((r_elements, z_elements))

        for i in range(1, z_elements):

            line = []

            for j in range(1, r_elements):
                
                r = j * int(self.pars["r_res"]["length"]/self.pars["r_res"]["devisions"])
                z_val = i * int(self.pars["z_res"]["length"]/self.pars["z_res"]["devisions"])
                z = z_val - self.pars["z_res"]["length"]/2

                v_r = self.calc_v(r, z, self.pars)

                tmp = {
                    "r" : j,
                    "z" : i,
                    "v" : v_r
                }

                v_field[i,j] = v_r 

                line.append(tmp)
            
            field.append(line)


        return field, v_field

    def show_field(self):
        """
        Function that shows calculation results as image
        """
        plt.matshow(self.flowfield)

        plt.show()
        


    def calc_v(self, r, z, pars):
        """
        Function that calculates velocity v dependent on positon (r = radius, z = height, pars = parameters)
        """
        
        Q = pars["flowrate"]
        h = pars["z_res"]["length"]

        v_max = (3*Q)/(4*math.pi*h*r)

        return v_max * (1 - ((4 * z**2)/(h**2)))


if __name__ == "__main__":

    example = {
        
        "flowrate" : 3,
        
        "z_res" : {
            "length" : 100,
            "devisions" : 10
        },

        "r_res" : {
            "length" : 50,
            "devisions" : 25
        }
    }

    test = flowfield(example)
    test.show_field()
