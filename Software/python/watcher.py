import time
import os
import sys
from tkinter import E
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transient_field import *
from add_data import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")


class Watcher:
    

    def __init__(self):

        self.case = "reaction_test_04_grob"

        cf_path = os.path.join(sys.path[0],"conf.json")
        with open(cf_path) as f:
            cf = json.load(f)

        self.directory = os.path.join(*cf["cases_dir_path"], self.case)

        if os.path.exists(self.directory) == False:
            logging.info(f"Folder for case {self.case} does not exsist. Path = {self.directory}")
            exit()

        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        try:
            logging.info("Started watcher...")
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            os.remove("tmp.txt")
            logging.info("Closed watcher.")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):

        

        if os.path.exists("tmp.txt") == False:
            counter = 0
            with open("tmp.txt", "w") as f:
                f.write("0")
        else:
            with open("tmp.txt") as f:
                counter = int(f.read())

        if event.is_directory:
            return None

        # elif event.event_type == 'created':

        #     # Take any action here when a file is first created.
        #     print("Received created event - {}.".format(event.src_path)) 

        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            counter += 1

            if counter == 2:
                print("Received modified event - {}.".format(event.src_path))

                # add_case("tmp", "watcher")

                cfg_path = os.path.join(sys.path[0], "watcher_conf.json")

                with open(cfg_path) as f:
                    config = json.load(f)

                time.sleep(1)

                field = flowfield(config)
                
                if config["create_image"]:
                    field.multi_field()
                
                if config["create_plot"]:
                    field.multi_plot()

                counter = 0

            with open("tmp.txt", "w") as f:
                f.write(str(counter))

if __name__ == '__main__':
    w = Watcher()
    w.run()