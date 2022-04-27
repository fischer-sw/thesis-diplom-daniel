import os
import re
import sys

def read_steady_data(file_name):
    """
    Read Ansys CSV export file.
    """

    path = os.path.join(sys.path[0], ".." , ".." , "Daten", "steady", file_name)

    result = {}
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            m = re.match('^\[(.*)\]', line)
            if m:
                tag = m.group(1)
                continue
            if not tag in result.keys():
                result[tag] = line.split(', ')
                continue
            if tag == 'Data':
                for i, val in enumerate(line.split(', ')):
                    key = result[tag][i]
                    if not key in result.keys():
                        result[key] = []
                    result[key].append(round(float(val), 6))

    return result

def read_transient_data():
    pass

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
