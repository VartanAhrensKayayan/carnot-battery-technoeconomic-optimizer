# carnot-battery-technoeconomic-optimizer

The goal is to use a genetic algorithm to find suitable sizes for Molten Salt based Carnot Batteries for repurposing coal-fired power plants in the European electricity market. This process would support the transition from coal-fired boilers to variable renewable power (i.e. wind and solar). As renewables push out coal-fired power plants, Carnot Batteries can make use of the existing infrastructure and existing power generators avoiding stranded assets. Furthermore, storage of energy can be one solution for the mismatch between production and demand which is in part communicated with price signals. However, this solution will not be implemented if it is not economically feasible. Thus this repository creates a method to systematically test different carnot batteries performance in different European electricity markets. There are three design parameters: direct heater size, storage size, and turbine size. The direct heater size determines the speed at which the battery can be recharged. The storage is the size of the molten salt storage and determines the maximum charge the battery can hold. Finally the turbine determines the rate at which the battery can be discharged and unlike the other design factors, it is constrained to the possible sizes of power blocks in existing but announced or planned to retire coal-fired power plants.

## Dependencies 
### Solver:

CBC (recommended see https://github.com/coin-or/Cbc)

or

GLPK

### Carbon Data (optional):


Electricity Map


## Workflow:

Choose a country within the European Union which has coal-fired power plants announced or planned to be closed.

The overarching workflow is to create a first population of solutions based on the established ranges of configurations from the literature. 

Then run the genetic algorithm to further optimize the design parameters. 

This genetic optimization is supportd by a simplified (but lenghty) optimization of the dispatch to estimate the income based on the hourly energy maket and the design settings.

Finally, the postprocessing can be used to display the results in graphs.
