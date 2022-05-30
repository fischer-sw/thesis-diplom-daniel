from transient_field import *

"""
File that calls build journal function 
"""

cfg_path = os.path.join(sys.path[0], "conf.json")

with open(cfg_path) as f:
    config = json.load(f)

field = flowfield(config)

field.setup_journal(exit=False)