# carnot-battery-technoeconomic-optimizer
A model to optimize the dispatch and sizing of Carnot Battery in the European electricity market.
It is based on ENTSO-E and its python wrapper data and requires a ENTSO-E API to run successfully (see https://github.com/EnergieID/entsoe-py).
The goal is to use a genetic algorithm to find suitable sizes for Molten Salt based Carnot Batteries. The overarching workflow is to create a first population of solutions based on the established ranges of configurations. Then use the genetic algorithm to further optimize a solution.
