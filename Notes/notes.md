# Notes

# Running a new case

1. Make shure the directory the auto export writes files to is empty
    - if not run add_data.py

2. Make shure the case is setup within cases.json (you can put all relevant parameters here. helps if you need to look them up later ;-) )

3. Run the case

4. Take a look at the latest results with running watcher.py (Config is done in watcher.json)

## Ansys Configuration

* Change UI language
    - go to ansys installation location
    - open (Ansys Inc -> Ansys Student -> v221(version number) -> ProductConfig.exe)
    - change language to desired one
## Ansys Fluent

User Interface Basics (DesignModeler + Mesher)

++strg++ + ++scroll wheel++ -> move
++scroll wheel++ -> tilt

### Workflow

0. First time opening
    - Drop **Fluent Flow(Fluent)** Workflow to Project
    - Doubleclick on Geometry
    - Close Window

1. Create Geometry in DesignModeler
    - rightclick **Geometry** -> **New DesignModeler Geometry...**
    - click on XY-Plane (Z arrow)
        - Axisymetric Problems
            - **radial direction**: y
            - **axial direction**: x (axis of rotation)
    - change from **Modelling** to **Sketching**
        - create sketch with drawing and constraints
        - add dimensions with dimensions
    - Highlight Sketch in Model tree -> Click **Concepts** -> **Surfaces from Sketch**
    - Click **Generate**
    - Check for errors in Model tree

2. Open mesher
    - name **Edges** (give usefull names e.g. inlet, outlet, wall_1, wall_2, ...) (shortkey ++N++)
        - select edge -> rightclick -> **create named selection**
    - create divisions (generate mesh near faces)
        - rightclick Mesh -> **Insert** -> **Sizing**
            - select edges -> Click Apply in Geometry tab
            - select type (e.g. Number of Devisions)
    - create inflation (create higher resolution near walls)
        - rightclick Mesh -> **Insert** -> **Inflation**
            - select Face -> click **Apply** under Geometry
            - select edges -> click **Apply** under Boundary
            - leave rest as default settings
    - create method (create inner mesh)
        - rightclick Mesh -> **Insert** -> **Method**
            - highlight body -> Click **Apply**
            - Definition Tab
                - Method: MultiZone Quad/Tri
    - click **Generate**
    - view mesh by clicking mesh in model tree

3. Setup
    - General
        - ![Image not found](img/general.png)
    - Models 
        - ![model.png](img/models.png)
    - Materials
        - Add fluids
            - RightClick Fluids -> **New**
            - Highlight all needed fluids from Database -> Click **Copy**
        - Delete all fluids not used
            - RightClick Fluid -> Click **Delete**
        - Create Mixture
            - under models --> **Species** --> activate **Species Transport**
            - under materials --> **Mixture** --> ++rightclick++ --> mixture-template --> **Edit**
            - under **Mixture Species** --> Edit
                - add all created fluids to mixture
                - remove non used species from mixture
                - delete non used materials under materials tab

    - Cell Zone Conditions
        - RightClick surface_body -> **Edit** -> Material Name: "fluid to use"

    - Boundry Conditions
        - inlet
            - ![Image not found](img/inlet_yorgos.png)
        - outlet
            - ![Image not found](img/outlet_yorgos.png)

    - Methods
        - ![Image not found](img/methods_yorgos.png)

    - Contorls
        - ![Image not found](img/controls_yorgos.png)

    - Initialization
        - ![Image not found](img/init_yorgos.png)

    - Run calculation
        - ![Image not found](img/calc_yorgos.png)

4. Solution
    - get a first impression of results
        - expand **Results** tab within model tree
        - expand **Graphics** tab
        - ++rightclick++ **Contours** -> **New**
        - select variable of interest from dropdown menus
        - click **Save/Display**
        - close settings window with **Close**

5. Results

    - export results 
        - create plane
            - highlight **Insert** in menu bar -> **Location** -> Plane
            - default settings are ok for this case
            - click **Apply**
            - plane should be visible within viewport
        - add results to plane
            - hihghlight **Insert** in menu bar -> **Contour**
            - select the just created plane at Locations
            - select the desired variable within Variable
            - click **Apply**
        - export data
            - highlight **File** in menu bar -> **Export** -> **Export...**
            - select just created plane at Locations
            - highlight variables that need to be exported (geometry information is already there due to ticked **Export Geometry Information**)
            - select export file location at **File**
            - click **save**

# Log Files
-  Console output will be saved within `Solution.trn` file

# Animations

## Create Animations
- click **Solution** tab --> Activities --> Create --> Solution Animations
    - ![Image not found](img/animation_setup.png)
    - Storage Type (PNG or JPEG for easier use later)
- run calculation (Initialize & Calculate)

## Remove Animation creation
- expand **Calculation Activities** within model tree
- expand **Solution Animations**
- delete Animation objects

# Data Export

## Create data export at every timestep (Transient Simulation)
- click **Solution** tab --> Activities --> Manage --> Create (under automatic Export)
    - ![Image not found](img/data_export.png)
    - File Type ASCII to be able to read data
    - File Name (including path) can have **.csv** extension for easier post processing but doesn't have to

# Intialize with species
- under **Initialize** menu --> Patch... --> set values --> click ++Patch++


# Parabolic velocity at inlet

- under **inlet** setup click on dropdown menu --> select **expression** --> click on **f(x)** button and enter expression
- expression: 

# Add Reaction

## general
- think about reaction e.g. $ 1A + 1B \rightarrow 1C$
- think about reation rate e.g. $r = k \cdot c_A^{n_{c_A}} \cdot c_B^{n_{c_B}} $
    - e.g. $n_{c_A} = n_{c_B} = 1 $
    - $ k = k_{\infty} \cdot e^{\frac{-E_A}{R \cdot T}}$

## ansys configuration
- under Models --> Species --> ++rightclick++ --> edit
- under Reactions --> tick Volumetric
- under Mixture Properties --> click ++Edit++
- under Reaction (finite rate) --> click ++Edit++
    - ![Image not found](img/reaction.png)
    - Rate Exponent e.g. fluid a: $n_{c_A}$
    - Pre-Exponential Factor: $k_\infty$
    - Activation Energy: $E_A$

## troubleshooting
- No or few product C
    - activation energy $E_A$ to high ?
    - temperature to low ?
- No dispersion
    - input velocity and diffusion rate match? (diffusion rate << input vel ?)
    - flow time high enough (dependant on input velocity)

# Automation

## Commandline (holy grail)

### Run fluent from cmdline

- find flunet installation path (windows installation)
    - e.g. `C:\Program Files\ANSYS Inc\v221\fluent\ntbin\win64`
    - add this path to PATH variable (windows 10)
        - Go to **Settings** --> **System** --> **Info** --> **System Information** --> **Advanced System Settings**
        - login as root
        - go to tab **advanced** --> **Environment variables**
        - Doubleclick **Path** under system variables
        - click **new** --> paste your path here
        - click ok

- cmd: `fluent 3ddp -tX -i <journal>.jou -g`
    - X: number of processors

## GUI Application

- to automate the gui application these steps need to be done:
    1. Create case file (e.g. <name>.cas.gz)
        - ! Make shure [automatic data export](#data-export) is enabled or you will not get any results !
        - Click **File** in Menubar --> **Export** --> **Case**
        - Choose **.cas.gz** as filetype

    2. Create journal for case
        - Start journal (**File** in Menubar --> **Write** --> **Journal** (to record your steps))
        - do things (Change all parameters you want to change)
            - changing the data export path is necessary ortherwise fluent will ask for permission to overwrite existing results
        - Stop recording (**File** in Menubar --> **Write** --> **Stop Journal**)
    3. Look at your journal (should look somewhat similar to gui_template.jou)

    4. Create journal or append existing one

    5. Run new journal
        - Click **File** in Menubar --> **Read** --> **Journal** --> Choose your journal

# Mesh dependancy study

1. Create mesh
2. Run steady simulation
3. Export variable
4. Refine mesh (e.g. number of divisions * 1.5)
5. Run simulation again
6. Check if variable has changed
    - if so refine again

# Scripts explaination

## Workflow to run case

1. Setup anys case within ansys workbench
2. Create journal file that edits all needed variables
3. Add marker (%variable name%) to base journal file
4. Add case to cases.json (within ansys folder)
5. Create and run journals with `journal.py`

## Workflow to create images

1. change `conf.json` within python directory acording to your needs
2. run `transient_field.py`
3. resulting images will be stored under `assets\<var_name>`


### Cases.json

| Variable|Explaination|Example|
|:--------:|:------------:|:------:|
|case_name|name of case to calculate. directory (see case_dir_path) will be created under data path if it doesn't exsist already | test |
|timestep|number of seconds per timestep| 0.01|
|data_export_interval| interval at which results will be created| 10|
|iterations| number of iterations per timestep | 30|
|timestep_number|number of timesteps to calculate|1e2|
|case|case to import first and then apply all variables| case_name.cas.gz|

### conf.json

|Variable|Explaination|Example|
|:-----:|:-----:|:-----:|
|cases_dir_path|path where cases are stored| ["\\\\gssnas", "bigdata", "fwdt", "DFischer" ,"Data"]
|cases| cases to create results for | ["test", "test1] |
|plots| timestamps that plots are created for | [], [-1, -2], [0, 10, 30] |
|create_resi_plot| if residual plots need to be created | false, true |
|create_image| if field needs to be created | false, true |
|field_var| variables that fields are created for | ["just put something here. if wrong u get shown all options"] |
|c_bar| colorbar label | velocity [m/s]
|image_file_type| image filetype | pmg, pdf, ...|
|set_custom_range| if custom range needs to be applied to field | true, false |
|min| minimum field_var value | 0 |
|max| maximum filed_var value | 1 |
|create_plot| if plots need to be created (averaged values over radius) | true, false |
|plot_vars| variables that are ploted| ["molef-fluid_a"]|
|plot_file_type| file type of plot | png, pdf, ... |
|one_plot| if multiple timesteps are ploted into one plot| true, false|
|plot_conf| plot config | |
|create_gif| if animation needs to be created (.gif file| true, false |
|video| if animation needs to be created (.avi file) | true, false |
|cases| animation image selection | |
|new| if existing images should be replaced | true, false |
|keep_images | if gif images are deleted after gif creation | true, false |
|gif_plot | if animation needs to be done for plots | true, false |
|gif_image | if animation needs to be done for fields | true, false |
|name | gif and video name | "bla"|
|loop | how many times gif and video needs to loop | 0 |
|frame_duration| how long in [ms] a frame should be displayed | 200 |
