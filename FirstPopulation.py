from pyomo.common.errors import ApplicationError
import ExploringCoal
from financial import sizingOptimizer
import os
import pandas as pd
from ExploringCoal import CoalPowerPlant

DEPowerBlocks =[#144,148,167,15,16,29,30,37,49,57,60,76,82,85,96,100,112,128,129,135,140,205,267,290,300,306,365,380,466,474,500,553,600,
                #724,780,790,795,816,820,855,923,980,
                1066,1100,1462,1595,1600,1868,2146,2582,3021,3210,4112, 38, 78,136]
PLPowerBlocks=[358, 380, 394, 390, 370, 858, 25, 12, 200,
               28, 50, 215, 225, 228, 560, 1075, 32, 55, 105, 100, 130, 386, 474]
ROPowerBlocks=[150,60,315,235,210,50,330]
FRPowerBlocks=[630,625,64,58]
CZPowerBlocks=[200, 110, 24, 55, 20, 41, 660, 205, 50, 57, 60, 250, 135, 136, 4,
               12, 1, 35, 32, 22, 72, 25, 19, 70, 67, 6, 38, 13, 105,
               30, 26, 9, 16, 33, 18, 34]


powerBlocks={'DE':DEPowerBlocks,'PL':PLPowerBlocks,'RO':ROPowerBlocks,'FR':FRPowerBlocks}

class Populate:
    def __init__(self):
        self.impossibles = []
        self.bidding = 'DE_LU'
        self.opex = 1
        self.payforpowerout = 100

        self.year=2021
        #self.optimizerfor = 'IRR'
        self.filename = f'NewResults/{self.bidding}_opex{self.opex}_powerout{self.payforpowerout}_Pulp'

        CPP = CoalPowerPlant()
        self.coalPowerPlants = CPP.getOneCountryList(countrytarget = self.bidding)

        if not os.path.isfile(f'{self.filename}.csv'):
            dummydata = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
            pd.DataFrame(dummydata, index=[0]).to_csv(f'{self.filename}.csv', header=False, index=False)

    def dispatchFirstPopulation(self):
        for powerBlock in [3.72875,0.01,1,2,3]:# self.coalPowerPlants:
            for heaterFactor in [0.2, 0.3, 1, 3, 10]:
                for storageHours in [0.1, 0.3, 1, 3, 6, 8, 10, 12, 24]:
                    old_results = pd.read_csv(f'{self.filename}.csv')
                    column_names = sizingOptimizer()
                    column_names = column_names.output.keys()
                    old_results.columns = column_names
                    print(column_names)
                    del column_names
                    for row in range(len(old_results)):
                        if powerBlock*heaterFactor == old_results.Power.at[row] \
                                and powerBlock*storageHours == old_results.Storage.at[row] \
                                and powerBlock == old_results.PowerOut.at[row]:
                            pass

                    output = sizingOptimizer()
                    output.biddingZone = self.bidding
                    output.opexPercentage=self.opex/100
                    output.payForPowerBlock=self.payforpowerout/100
                    output.demandFile ='DemandPulp.csv'
                    output.initialPowerIn = round(powerBlock * heaterFactor)
                    output.initialCapacity = round(powerBlock * storageHours)
                    output.initialDischargePower=powerBlock
                    output.year=self.year
                    try:
                        output.buildFinancial()

                    except ApplicationError:
                        self.impossibles.append([powerBlock, heaterFactor, storageHours])
                        print(f'{self.impossibles}')
                        pass
                    data = output.output
                    results = pd.DataFrame([data])
                    results.to_csv(f'{self.filename}.csv', index=False, header=False, mode='a')
                    print(self.impossibles)
if __name__ == "__main__":
    firstpop= Populate()
    firstpop.dispatchFirstPopulation()