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
    - model implementation
        - parallel computing etc
- validation
    - design studies (parametric)
    - experimental setup
    - model results
- parameter variation
    - Pe number
    - Sc number
    - front width
    - total amount of product formed
    - (production rate)
- limitations and errors
- outlook/conculsion

# Open Questions

- threshold for front position --> widths calculation
- Sc and Pe numbers
    - Pe [500, 931, 5000]?
    - Sc [500, 1219, 2500]?

case_name | Pe | Sc |
| - | - | - |
| h4r3_P500_S1219_gs | 500 | 1219 |
| h4r3_P931_S1219_gs | 931 | 1219 |
| h4r3_P5e3_S1219_gs | 5000 | 1219 |
| h6r3_P500_S1219_ms | 500 | 1219 |
| h6r3_P931_S1219_ms | 931 | 1219 |
| h6r3_P5e3_S1219_ms | 5000 | 1219 |

- investigations:
    - front position
        - threshold as pct value of maximum
        - substract offset from values
    - front width
        - Full width at half maximum
    - total amount of product
