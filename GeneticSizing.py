import gc

import numpy as np
import pygad
from financial import sizingOptimizer
import pandas as pd
import os
from ExploringCoal import CoalPowerPlant


def fitness_func(solution, solution_idx):
    print(f'powerin: {solution[0]} Capacity:{solution[1]} and powerout: {solution[2]}')

    oldresults = pd.read_csv(f'{filename}.csv')
    # columnnames = sizingOptimizer()
    # columnnames = columnnames.new.keys()
    oldresults.columns = ['Storage', 'Power', 'PowerOut', 'CAPEX', 'Profit', 'Costs', 'Yearly Cashflow', 'LCOE', 'IRR',
                          'NPV', 'Capacity Factor', 'Utilization Factor', 'CO2 Avoided', 'RoundTrip', 'WACC']
    # print(columnnames)
    # del columnnames
    # This loops avoids that the same design runs several times
    for x in range(len(oldresults)):
        if solution[0] == oldresults.Power.at[x] and solution[1] == oldresults.Storage.at[x] and solution[2] == \
                oldresults.PowerOut.at[x]:
            fitness = oldresults[f'{optimizerfor}'].at[x]
            print(fitness)
            if np.isnan(fitness):
                fitness = -9999999.0
            return fitness
    # If the design is unique the dispatch is optimized for and the financials are calculated
    else:
        new = sizingOptimizer()
        new.biddingZone = bidding
        new.opexPercentage = opex / 100
        new.payForPowerBlock = payforpowerout / 100
        new.initialPowerIn = solution[0]
        new.initialCapacity = solution[1]
        new.initialDischargePower = solution[2]
        new.demandFile = 'DemandPulp.csv'
        new.buildFinancial()

        results = pd.DataFrame([new.output])

        results.to_csv(f'{filename}.csv', index=False, header=False, mode='a')
        fitness = new.output[f'{optimizerfor}']
        print(fitness)
        return fitness
        # if pd.isnull(fitness):
        #    fitness=-99
        # print(gc.garbage)
        # if fitness<0:
        #    print('-----------------------------------------------------------')
        # else:
        #    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # return fitness


def on_start(ga_instance):
    print("on_start()")


def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")


def on_parents(ga_instance, selected_parents):
    print("on_parents()")


def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")


def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")


def on_generation(ga_instance):
    print("on_generation()")


def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


if __name__ == "__main__":
    fitness_function = fitness_func
    bidding = 'DE_LU'
    opex = 1
    payforpowerout = 100
    optimizerfor = 'Profit'

    filename = 'NewResults/DE_LU_opex1_powerout100_Pulp'  # f'NewResults/{bidding}_opex{opex}_powerout{payforpowerout}_pulp'
    if not os.path.isfile(f'{filename}.csv'):
        dummydata = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        pd.DataFrame(dummydata, index=[0]).to_csv(f'{filename}.csv', header=False, index=False)

    CPP = CoalPowerPlant()
    coalPowerPlants = CoalPowerPlant.getOneCountryList(CPP, countrytarget=bidding)

    firstpopulation = pd.read_csv(f'NewResults/{bidding}_2021_results_pulp.csv')
    firstpopulation = firstpopulation[['ChargePower', 'Storage', 'DischargePower']]
    firstpopulation = firstpopulation.to_numpy()
    print(firstpopulation)

    ga_instance = pygad.GA(num_generations=10,
                           num_parents_mating=2,
                           fitness_func=fitness_function,
                           sol_per_pop=6,
                           num_genes=3,
                           K_tournament=6,
                           # random_mutation_min_val=-100.0,
                           # random_mutation_max_val=100.0,
                           gene_space=[range(101), range(1_000),range(101)], # list(np.arange(0.1,5.1,0.1))+[3.72875]],  # coalPowerPlants],

                           initial_population=firstpopulation,
                           mutation_probability=[0.75, 0.50],
                           mutation_type="adaptive",
                           gene_type=float,
                           allow_duplicate_genes=False,
                           parent_selection_type='tournament',
                           on_start=on_start,
                           on_fitness=on_fitness,
                           on_parents=on_parents,
                           on_crossover=on_crossover,
                           on_mutation=on_mutation,
                           on_generation=on_generation,
                           on_stop=on_stop,
                           )

    for generations in range(11):
        # ga_instance =pygad.load(filename=filename)
        print('here')
        ga_instance.run()
        # solution, solution_fitness, solution_idx = ga_instance.best_solution()
        # print(f'Parameters of the best solution : {solution[0]} MW electric heater and {solution[1]} MWh sized storage')
        # print(f'Fitness value of the best solution = {solution_fitness}')
        # print(f'Index of the best solution : {solution_idx}')
        ga_instance.save(filename=filename)
        gc.collect()
