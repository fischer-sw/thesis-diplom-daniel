import os
import sys


from transient_field import *
from ansys_utils import *

"""
File that calls build journal function 
"""
def create_journals(exit=True, split_cases=True):

    cfg_path = os.path.join(sys.path[0], "conf.json")

    with open(cfg_path) as f:
        config = json.load(f)

    field = flowfield(config)

    field.setup_journal(exit, split_cases)

def run_journals(journals=[]):
    if journals == []:
        logging.warning("No Journals to run")
    else:
        for jour in journals:
            run_fluent(jou_name=jour)

if __name__ == "__main__":
    create_journals(True, True)
    run_journals(["v_3e-3"])