import sys
import os
import json
import base64
import logging

import plotly.graph_objects as go
import numpy as np

from dash import *
from plotly.tools import mpl_to_plotly
from ansys_utils import *
from transient_field import flowfield
import dash_bootstrap_components as dbc

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")

PACKAGE_PARENT = "../.."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

external_scripts = [{'src': 'https://cdn.plot.ly/plotly-locale-de-latest.js'}]


app = dash.Dash(__name__, external_scripts=external_scripts,external_stylesheets=[dbc.themes.GRID, dbc.themes.BOOTSTRAP], assets_folder="./assets")

title = html.Div(
    html.H2("Ansys Fluent Post Processing")
)

primary_row = html.Div(hidden=False, id="case-selector",children=
    [       
        dbc.Button("Select Case", color="primary", id="setup-case", className="me-1"),
    ]
)

secondary_row = html.Div(hidden=True, id="image-setup", children=
    [
        dbc.Button("Setup Field", color="primary", id="setup-field", className="me-1"),
        dbc.Button("Setup Plot", color="primary", id="setup-plot", className="me-1"),
    ]
)

third_row = html.Div(hidden=True, id="image-create", children=
    [
        dbc.Button("Create Field", color="primary", id="create-field", className="me-1"),
        dbc.Button("Create Plot", color="primary", id="create-plot", className="me-1"),
    ]
)

download_row = html.Div(hidden=True, children=
    [
        dbc.Button("Download Image",
            href="/static/data_file.txt",
            download="my_data.txt",
            external_link=True,
            color="primary",
        ),
        dbc.Button("Download Plot",
            href="/static/data_file.txt",
            download="my_data.txt",
            external_link=True,
            color="primary",
        ),
    ]
)

fig = go.Figure()

image_row = html.Div(
    [
        dash.html.Img(id="img", alt="no image yet created"),
        dbc.Spinner(html.Div(id="loading-output")),
    ]
)

case_modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Please select case"), close_button=True),
                dbc.ModalBody([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                dcc.Dropdown([], None, id='dd-cases')
                                ],
                                width=12,
                            )
                        ],
                        className="g-3",
                    ),
                ]),
                
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-case-modal",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="case-modal",
            centered=True,
            is_open=False,
        ),
    ]
)

plot_modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Please setup the plot"), close_button=True),
                dbc.ModalBody([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                
                                ],
                                width=12,
                            )
                        ],
                        className="g-3",
                    ),
                ]),
                
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-plot-modal",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="plot-modal",
            centered=True,
            is_open=False,
        ),
    ]
)

image_modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Please setup the image"), close_button=True),
                dbc.ModalBody([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    
                                    html.Div(id="time-list-container", children=[dcc.Dropdown(options=[], value=None, id='dd-times', placeholder="times", multi=True)]),

                                    dcc.Dropdown(options=[], value=None, id='var-dd', placeholder="variable"),

                                ],
                                width=12,
                            )
                        ],
                        className="g-3",
                    ),
                ]),
                
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-image-modal",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="image-modal",
            centered=True,
            is_open=False,
        ),
    ]
)


app.layout = dbc.Container( children=[
    dbc.Col(
        [
            title,
            primary_row,
            secondary_row,
            third_row,
            download_row,
            image_row,
            case_modal,
            image_modal,
            plot_modal,
        ],
        width=12,
    )
]
)


cfg_path = os.path.join(sys.path[0], "conf.json")

if os.path.exists(cfg_path) == False:
    logging.info(f"no conf.json found at {cfg_path}")
    exit()
with open(cfg_path) as f:
    config = json.load(f)

data_cfg_path = os.path.join(sys.path[0], "..", "ansys", "cases.json")

if os.path.exists(data_cfg_path) == False:
    logging.info(f"no cases.json found at {data_cfg_path}")
    exit()
with open(data_cfg_path) as f:
    cases_config = json.load(f)

@app.callback(
    Output("image-modal", "is_open"),
    [Input("setup-field", "n_clicks"), Input("close-image-modal", "n_clicks")],
    [State("image-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("plot-modal", "is_open"),
    [Input("setup-plot", "n_clicks"), Input("close-plot-modal", "n_clicks")],
    [State("plot-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("case-modal", "is_open"),
    [Input("setup-case", "n_clicks"), Input("close-case-modal", "n_clicks")],
    [State("case-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



@app.callback(
    Output("dd-cases", "options"),
    Input("setup-case", "n_clicks"),
    # Input("dd-cases", "value"),
)

def get_options(clicks):
    options = []
    path = os.path.join(*config["cases_dir_path"])

    if os.path.exists(path) == False:
        logging.info(f"Path {path} does not exsist")
        exit()

    files = os.listdir(path)
    
    rm = []
    for file in files:
        if ".sh" in file:
            rm.append(file)

    for file in rm:
        files.remove(file)

    for case_name in files:
        options.append({"label" : case_name, "value" : case_name})

    return options

@app.callback(
    Output("image-setup", "hidden"),
    Output("case-selector", "hidden"),
    Output("image-create", "hidden"),
    Input("dd-cases", "value"),
    Input("close-image-modal", "n_clicks")
)

def update_btn_hide(value, clicks):
    if value == None:
        hide1 = True
        hide2 = False
        hide3 = True
    else:
        hide1 = False
        if clicks == 0:
            hide2 = False
            hide3 = True
        else:	
            hide2 = True
            hide3 = False

    config["cases"] = [value]
    return [hide1, hide2, hide3]

@app.callback(
    Output("dd-times", "options"),

    Input("dd-cases", "value",),
    prevent_initial_call=True
)

def get_time_options(value):
    cases = get_cases(config, value)
    # cases = get_cases(config["hpc_data_path"], config["cases_dir_path"], value)

    # times = cases_config[value]["timestep"] * np.array(cases)

    times = np.round(cases,1)
    times.sort()

    mini = min(times)
    maxi = max(times)

    stepsize = round((times[-1] - times[0])/10,0)

    options = []

    for idx, ele in enumerate(times):

        tmp = {
            "label" : f"t = {str(ele).replace('.', ',')}s",
            "value" : ele
        }
        
        options.append(tmp)

    return options


@app.callback(
    Output("var-dd", "options"),

    Input("dd-cases", "value",),
    prevent_initial_call=True
)

def update_vars(case):
    vars = get_case_vars(config, case)

    options = []

    vars = vars[2:]

    for idx, ele in enumerate(vars):

        tmp = {
            "label" : f"{ele}",
            "value" : f"{ele}"
        }
        
        options.append(tmp)
    
    return options

@app.callback(
    Output("img", "src"),
    Output("loading-output", "children"),
    Input("create-field", "n_clicks"),
    State("dd-times", "value"),
    State("var-dd", "value"),
    prevent_initial_call=True
)

def create_image(clicks, time_values, var):
    config["field_var"] = [var]
    config["ignore_exsisting"] = True
    config["plots"]= time_values
    config["create_front"]= False
    config["create_image"]= True
    config["create_resi_plot"]= False
    config["create_gif"]= False
    field = flowfield(config, cases_config)

    path = sys.path[0]
    path = os.path.join(path, "assets", "fields", config["cases"][0])
    sub_path = os.path.join(path, var)
    
    
    if config["create_image"]:
        image_name = "field_" + config["cases"][0] + "_" + var + "." + config["image_file_type"]
        image_path = os.path.join(sub_path, image_name)
        field.multi_field(config, cases_config)
    
    if config["create_plot"]:
        field.multi_plot(config, cases_config)

    if config["create_plot"] or config["create_image"]:

        encoded_image = base64.b64encode(open(image_path, 'rb').read())
        base_src = 'data:image/png;base64,{}'.format(encoded_image.decode())
    
        return [base_src, None] 
    
    return "Output not reloaded yet"

if __name__ == "__main__":
    app.run_server(debug=False)
    # app.run_server(host='127.0.0.1', port=8050, debug=True, use_debugger=True, use_reloader=True)