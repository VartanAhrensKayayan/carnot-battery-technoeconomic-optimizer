from sizingOptimizer import sizingOptimizer
import os
import pandas as pd

impossibles=[]
bidding='DE_LU'
opex=1
payforpowerout=50
optimizerfor='IRR'
filename = f'NewResults/{bidding}_opex{opex}_powerout{payforpowerout}_{optimizerfor}'
if not os.path.isfile(f'{filename}.csv'):
    dummydata = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,13:0,14:0,15:0}
    pd.DataFrame(dummydata, index=[0]).to_csv(f'{filename}.csv', header=False, index=False)
DEPowerBlocks =[15,16,29,30,37,38,49,57,60,76,78,82,85,96,100,112,128,129,135,136,140,144,148,167,
                205,267,290,300,306,365,380,466,474,500,553,600,724,780,790,795,816,820,855,
                923,980,1066,1100,1462,1595,1600,1868,2146,2582,3021,3210,4112]
PLPowerBlocks=[358, 380, 394, 390, 370, 858, 25, 12, 200,
               28, 50, 215, 225, 228, 560, 1075, 32, 55, 105, 100, 130, 386, 474]
ROPowerBlocks=[150,60,315,235,210,50,330]
FRPowerBlocks=[630,625,64,58]

powerBlocks={'DE':DEPowerBlocks,'PL':PLPowerBlocks,'RO':ROPowerBlocks,'FR':FRPowerBlocks}

for powerblock in DEPowerBlocks:
    for heaterfactor in [0.1,0.3,1,3,10]:
        for storagehours in [0.1,1,2,3,4,5,6,7,8,9,10,11,12,24]:
            oldresults = pd.read_csv(f'{filename}.csv')
            columnnames = sizingOptimizer()
            columnnames = columnnames.output.keys()
            oldresults.columns = columnnames
            del columnnames
            for x in range(len(oldresults)):
                if powerblock*heaterfactor == oldresults.Power.at[x] \
                        and powerblock*storagehours == oldresults.Storage.at[x] \
                        and powerblock == oldresults.PowerOut.at[x]:
                    pass
            output = sizingOptimizer()
            output.biddingZone=bidding
            output.opexPercentage=opex/100
            output.payForPowerBlock=payforpowerout/100
            output.initialPowerIn = round(powerblock*heaterfactor)
            output.initialCapacity = round(powerblock*storagehours)
            output.initialDischargePower=powerblock
            #output.buildFinancial()
            try:
                output.buildFinancial()
            except:
                impossibles.append([powerblock,heaterfactor,storagehours])
                print(f'{impossibles}//////////////////////////////////////////////////////////////////////////')
                pass
            data = output.output
            results = pd.DataFrame([data])
            results.to_csv(f'{filename}.csv', index=False, header=False, mode='a')