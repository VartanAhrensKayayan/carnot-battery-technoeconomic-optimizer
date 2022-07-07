# carnot-battery-technoeconomic-optimizer

A model to optimize the dispatch and sizing of Carnot Battery in the European electricity market. The goal is to use a genetic algorithm to find suitable sizes for Molten Salt based Carnot Batteries. The overarching workflow is to create a first population of solutions based on the established ranges of configurations from the literature. Then use the genetic algorithm to further optimize a solution. This higher level optimization is supportd by a simplified optimization of the dispatch to estimate the income based on the hourly energy maket and the design settings. 

Dependencies 
Solver:
  CBC (recommended see https://github.com/coin-or/Cbc)
  GLPK
Carbon Data (optional):
  Electricity Map

