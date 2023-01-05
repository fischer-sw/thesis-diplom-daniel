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
- [x] setup correct model
- [x] do parameter studies

# Structure

- introduction
    - motivation
    - objectives and report outline
- theory
    - cfd in general
    - governing equations
        - mass conservation
        - energy equation
        - reaction equation
    - finite volume method
    - solution methods
        - PISO
    - flow observations
        - Taylor-dispersion
        - reaction details
        - dimensionless variables
- model
    - model setup
        - geometry
        - meshing
        - solver settings
    - model evolution (mesh independancy)
    
- validation
    - experimental setup
    - model results
        - front
        - width
        - product
    - experimental results
    - comparison
- parameter variation
    - 
    - front width
    - total amount of product formed
    - (production rate)
    
- limitations and errors
- outlook/conculsion
- appendix
    - model implementation
        - parallel computing etc

# Open Questions and Todos