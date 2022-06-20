import gc
import pygad
from sizingOptimizer import sizingOptimizer
import pandas as pd
import os



def fitness_func(solution, solution_idx):
    print(f'powerin: {solution[0]} Capacity:{solution[1]} and powerout: {solution[2]}')

    oldresults = pd.read_csv(f'{filename}.csv')
    columnnames = sizingOptimizer()
    columnnames = columnnames.output.keys()
    oldresults.columns = columnnames
    del columnnames
    for x in range(len(oldresults)):
        if solution[0] == oldresults.Power.at[x] and solution[1] == oldresults.Storage.at[x] and solution[2]==oldresults.PowerOut.at[x]:
            fitness = oldresults.IRR.at[x]
            return fitness

    else:
        output = sizingOptimizer()
        output.biddingZone=bidding
        output.opexPercentage=opex/100
        output.payForPowerBlock=payforpowerout/100
        output.initialPowerIn = int(solution[0])
        output.initialCapacity = int(solution[1])
        output.initialDischargePower=int(solution[2])
        output.buildFinancial()
        data = output.output
        results = pd.DataFrame([data])

        results.to_csv(f'{filename}.csv', index=False, header=False, mode='a')
        print(output.WACC)
        fitness = output.output[f'{optimizerfor}']
        if pd.isnull(fitness):
            fitness=-99
        print(gc.garbage)
        if fitness<0:
            print('-----------------------------------------------------------')
        else:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return fitness


fitness_function = fitness_func


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


ga_instance = pygad.GA(num_generations=10,
                       num_parents_mating=2,
                       fitness_func=fitness_function,
                       sol_per_pop=6,
                       num_genes=3,
                       K_tournament=6,
                       random_mutation_min_val=-100.0,
                       random_mutation_max_val=100.0,
                       gene_space=[range(1, 1001, 10), range(10, 630 * 24, 10),
                                   [630,625,64,58]],
                       #[358, 380, 394, 390, 370, 858, 25, 12, 200, 28, 50, 215, 225, 228, 560, 1075, 32, 55, 105, 100, 130, 386, 474]],

                       #[150,60,315,210,235,210,50,330,150]],#RO
                       #            [15,16,29,30,37,38,49,57,60,76,78,82,85,96,100,112,128,129,135,136,140,144,148,167,
                       #             205,267,290,300,306,365,380,466,474,500,553,600,724,780,790,795,816,820,855,
                       #             923,980,1066,1100,1462,1595,1600,1868,2146,2582,3021,3210,4112]],

                       initial_population=firstpopulation,
                       mutation_probability=[0.75, 0.50],
                       mutation_type="adaptive",
                       gene_type=int,
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
if __name__ == "__main__":
    bidding='FR'
    opex=1
    payforpowerout=25
    optimizerfor='IRR'
    filename = f'NewResults/{bidding}_opex{opex}_powerout{payforpowerout}_{optimizerfor}'
    firstpopulation = pd.read_csv(f'NewResults/{bidding}_2021_results.csv')
    firstpopulation = firstpopulation[['ChargePower', 'Storage', 'DischargePower']]
    firstpopulation = firstpopulation.to_numpy()
    if not os.path.isfile(f'{filename}.csv'):
        dummydata = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,13:0,14:0,15:0}
        pd.DataFrame(dummydata, index=[0]).to_csv(f'{filename}.csv', header=False, index=False)


    for fiveGenerations in range(11):
        # ga_instance =pygad.load(filename=filename)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f'Parameters of the best solution : {solution[0]} MW electric heater and {solution[1]} MWh sized storage')
        print(f'Fitness value of the best solution = {solution_fitness}')
        print(f'Index of the best solution : {solution_idx}')
        ga_instance.save(filename=filename)
        gc.collect()
