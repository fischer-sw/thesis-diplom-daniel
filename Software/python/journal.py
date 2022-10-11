import os
import sys
import json
import logging

from transient_field import *
from ansys_utils import *

"""
File that calls build journal function 
"""
def create_journals(exit=True, mode="cmd", update_exsisting=False):

    cfg_path = os.path.join(sys.path[0],".." ,"ansys","cases.json")
    with open(cfg_path) as f:
        cases_cfg = json.load(f)

    cases = list(cases_cfg.keys())

    cf_path = os.path.join(sys.path[0],"conf.json")
    with open(cf_path) as f:
        config = json.load(f)

    config["cases"] = cases

    field = flowfield(config, cases_cfg)

    field.setup_journal(config, cases_cfg, mode ,exit, update_exsisting)

def run_journals(run_gui_case=False):

    jou_path = os.path.join(sys.path[0], "..", "ansys", "journals", "cases")
    journals = glob.glob("*.jou", root_dir=jou_path)

    if journals == []:
        logging.warning("No Journals to run")
    else:
        for jour in journals:
            run_fluent(jou_name=jour, run_gui_case=run_gui_case)

if __name__ == "__main__":

    # Use gui to run cases on local machine
    # Use cmd to create journal for commandline execution on hpc cluster
    # mode = "gui"
    mode = "cmd"

    create_journals(exit=True, mode=mode, update_exsisting=False)
    if mode == "cmd":
        run_journals(False)
    else:
        run_journals(True)