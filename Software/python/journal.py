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

def prep_cases():

    cases = {}
    path = os.path.join(sys.path[0], "..", "..", "Notes", "simulationen.ods")
    cfg = pd.read_excel(path, engine="odf", header=3)
    # cfg = cfg[cfg.success == "nein"]
    
    for idx in range(cfg.shape[0]):
        row = cfg.iloc[idx]
        cas_name = "_".join([row.case[0:2]+ row.case[3:5], "P" + f"{row.Pe:.2E}"[0:5].replace(".", "") + f"{row.Pe:.2E}"[-1], "S" + f"{row.Sc:.2E}"[0:5].replace(".", "") + f"{row.Sc:.2E}"[-1]])
        cases[cas_name] = {
            "export_times" : row["export_times"],
            "total_time" : float(row["total_time_[s]"]),
            "fixed_timesteps" : int(row["fixed_timesteps"]),
            "timestep_min" : float(row["timestep_min_[s]"]),
            "data_export_interval" : float(row["export_interval"]),
            "input_vel" : float(row["u_[m/s]"]),
            "inlet_xA" : float(row["x_A"]),
            "init_phiB" : float(row["phi_B"]),
            "viscosity" : float(row["ν_[m²/s]"]),
            "diffusion" : float(row["D_[m²/s]"]),
            "iterations" : int(row["iterations"]),
            "queue" : row["queue"],
            "tasks_p_node" : int(row["tasks_p_node"]),
            "wall_time" : str(row["wall_time_[h]"]),
            "post_wall_time" : str(row["post_wall_time_[h]"]),
            "case" : row["case"]
        }

    cases_path = os.path.join(sys.path[0], "..", "ansys", "cases.json")
    with open(cases_path, "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    # Use gui to run cases on local machine
    # Use cmd to create journal for commandline execution on hpc cluster
    # mode = "gui"
    mode = "cmd"

    prep_cases()

    run_jour = True
    create_journals(exit=True, mode=mode, update_exsisting=True)
    
    if run_jour:
        if mode == "cmd":
            run_journals(False)
        else:
            run_journals(True)