# thesis-diplom-daniel

diplom thesis on numerical simulation of reaction-diffusion-advection fronts

## Steps to follow

- [x] create geometry in Ansys
- [x] solve flow field for geometry
- [x] compute analytical solution for flow field
- [x] compare analytical with computed solution
- [x] add species to model
- [x] solve model for species
- [x] add reaction to model
- [x] solve model with reaction
- [ ] setup correct model
- [ ] do parameter studies

# Structure

- introduction
    - motivation
    - objectives
    - report outline
- theory
    - cfd in general
    - Navier-Stokes-equations
    - finite volume method
    - solution methods
        - PISO
    - participating phenomena
        - RDA stuff (Alan Turing)
        - Taylor-dispersion
- model
    - governing equations
        - mass conservation
        - energy equation
        - reaction equation
    - model setup
        - geometry
        - meshing
        - setup
    - model evolution
        - solving flowfield
        - adding fluids
        - adding reaction
        - mesh gird dependency
- validation
    - design studies (parametric)
    - experimental setup
    - model results
- limitations and errors
- outlook/conculsion



aspect ratio = 1...5
avg wall shear
repeat 1-3 times