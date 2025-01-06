Adiabatic geotherms are calculated by numercically integrating the following equation using the 4th order Runge-Kutta method:

$$\frac{dT}{dP} =  \frac{\alpha T}{\rho C_{p}} - \frac{H_{fus}}{C_{p}} \cdot \frac{dF}{dP} $$

The thermal expansivity ($\alpha$), density, and specific heat capacity ($C_p$) of the sytems are calculated using Gibbs free energy minimization software (MAGEMin). Under conditions where both solid and melt are stable, the enthalpy of fusion, and hence the latent heat term, is non-zero and needs to be calculated. If the systems is in chemical equlibrium, then:

$$\Delta G_{fus} = G_{liquid}(P,T) - G_{solids}(P,T) = 0 $$

$$\Delta G_{fus} = \Delta h_{fus} - T\Delta s_{fus} + \int_{P0}^{P} [v_{liquid} - v_{solid}]dP $$

$$ \Delta h_{fus} = T\Delta s_{fus} - \int_{P0}^{P} [v_{liquid} - v_{solid}]dP $$

where $h_{fus}$ is the specific enthalpy of fusion, $s_{fus}$ is the specific entropy of fusion, $v = 1/\rho$ and denotes the specific volume of the liquid and solid phases.

$$ \Delta s_{fus} = s_{liquid} - \sum_{i=0}^{n} x_{i}s_{i} $$

$$ \Delta v = v_{liquid} - \sum_{i=0}^{n} x_{i}v_{i} = v(P) $$

and $x_{i}$ is the mass fraction of each solid mineral phase.

The pressure-volume integral is approximated using the Trapezoid rule:

$$ \int_{P0}^{P} v(P) \cdot dP \approx \sum_{i=0}^{n-1} \frac{v(P_k) + v(P_{k+1})}{2} \cdot [P_{k+1}-P_{k}]$$

