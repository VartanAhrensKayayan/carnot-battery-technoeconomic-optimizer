import os
import sys

from dispatcher import dispatchOptimizer
import numpy_financial as npf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from entsoe import EntsoePandasClient
import pygad
import random
import csv
import gc


class sizingOptimizer:
    def __init__(self):

        self.costBuying = None
        self.hoursDischarging = None
        self.hoursCharging = None
        self.energyOut = None
        self.energyIn = None
        self.decomissionRatio = 0.0
        print('Sizing Started.')
        self.irr = None
        self.lcoe = None

        self.opexPercentage = 0.05  # The opex is calculated as a percentage of capex which varies between 1 and 5%
        self.payForPowerBlock=25/100

        self.initialPowerIn = 1500
        self.initialCapacity = 3600
        self.initialDischargePower = 150


        self.capex = 0
        self.capexBreakDown ={}


        self.constructionTime = 2



        self.referenceCostSize = 500_000
        self.referenceSizeSize = 10
        self.alphaSize = 0.75
        self.dispatch = 0

        self.biddingZone='DE_LU'
        self.year=2021
        self.taxCorporateChoose={'DE_LU':.22,'RO':.16}
        self.lifetime = 30
        self.npvValue = -999999999999999
        self.roundTripEfficiency= None

        self.WACCMethod='Lantz'
        if self.WACCMethod=='Lantz':
            # WACC calculating factors as per Lantz(2020)
            self.equityPercent = .2
            self.equityIRR = 0.1
            self.debtPercent = 0.8
            self.debtInterest =0.06
            self.taxCorporate = 0.30
        else:
            # WACC calculating factors as per (Wagner & Rubin)
            self.equityPercent = .60
            self.equityIRR = 0.0874
            self.debtPercent = .40
            self.debtInterest = .08
            self.taxCorporate = 0.22

        self.WACC = self.equityPercent * self.equityIRR + self.debtPercent * self.debtInterest * (1 - self.taxCorporate)

        self.degradation = 0

        self.profit = None
        self.yearlyValues = 0
        self.cashflow = None
        self.cost = [0, 0, 0, 0]
        self.energyflow = None
        self.capacityFactor = None
        self.utilizationFactor = None
        self.CO2Avoided = None
        self.output = {'Storage': self.initialCapacity,
                       'Power': self.initialPowerIn,
                       'PowerOut': self.initialDischargePower,
                       'CAPEX': self.capex,
                       'Profit': self.profit,
                       'Costs': int(self.cost[self.constructionTime + 1]),
                       'Yearly Cashflow': float(self.yearlyValues),
                       'LCOE': self.lcoe,
                       'IRR': self.irr,
                       'NPV': self.npvValue,
                       'Capacity Factor': self.capacityFactor,
                       'Utilization Factor': self.utilizationFactor,
                       'CO2 Avoided': self.CO2Avoided,
                       'RoundTrip':self.roundTripEfficiency,
                       'WACC':self.WACC}

        self.possible = True
        self.capexForOpex=None
        self.filename=f'NewResults/{self.biddingZone}_{self.year}_results_pulp.csv'

    def buildFinancial(self):
        self.checkDispatchSolution()
        self.getCapex()

        self.getCashflow()
        self.getRoundTripEfficiency()
        if self.WACCMethod == 'Own':
            self.getWACC()
        if self.possible:
            self.getIRR()
            self.getNPV()
            self.getCostforLCOE()
            self.getEnergy()
            self.getLCOE()
            self.getCapacityFactor()
            self.getUtilizationFactor()
            self.getCO2Avoided()
            self.output = {'Storage': self.initialCapacity,
                           'Power': self.initialPowerIn,
                           'PowerOut': self.initialDischargePower,
                           'CAPEX': self.capex,
                           'Profit': self.profit,
                           'Costs': self.cost[self.constructionTime + 1],
                           'Yearly Cashflow': self.yearlyValues[0],
                           'LCOE': self.lcoe,
                           'IRR': self.irr,
                           'NPV': self.npvValue,
                           'Capacity Factor': self.capacityFactor,
                           'Utilization Factor': self.utilizationFactor,
                           'CO2Avoided': self.CO2,
                           'RoundTrip':self.roundTripEfficiency,
                           'WACC':self.WACC}
            self.output = {k: round(v, 2) for k, v in self.output.items()}
        else:
            self.getCostforLCOE()
            self.getEnergy()
            self.getLCOE()
            self.output['LCOE']=self.lcoe


    def getCapex(self):
        self.capex = capexMoltenSalt(self.initialCapacity,
                                     targetPowerOut=self.initialDischargePower*self.payForPowerBlock).value
        directHeater=  self.referenceCostSize * (self.initialPowerIn / self.referenceSizeSize) \
                      ** self.alphaSize
        self.capex += directHeater
        self.capex *= - 1
        self.capex = round(self.capex)

        self.capexBreakDown= capexMoltenSalt(self.initialCapacity,
                                     targetPowerOut=self.initialDischargePower*self.payForPowerBlock).valueBreakDown
        self.capexBreakDown['Direct Heater']= directHeater


        self.capexForOpex = capexMoltenSalt(self.initialCapacity,
                                     targetPowerOut=self.initialDischargePower).value
        self.capexForOpex += self.referenceCostSize * (self.initialPowerIn / self.referenceSizeSize) \
                      ** self.alphaSize
        self.capexForOpex *= - 1
        self.capexForOpex = round(self.capexForOpex)


        return self.capex

    def checkDispatchSolution(self):
        if not os.path.isfile(self.filename):
            dummydata = {'ChargePower': 0, 'Storage': 0, 'DischargePower': 0, 'EnergyIn': 0, 'EnergyOut': 0,
                         'HoursCharging': 0, 'HoursDischarging': 0, 'CostBuying': 0, 'Profit': 0, 'CO2': 0}
            pd.DataFrame(dummydata, index=[0]).to_csv(f'{self.biddingZone}_{self.year}_results.csv', index=False)
        readysolutions =pd.read_csv(self.filename)

        for rows in range(len(readysolutions)):
            if (readysolutions['ChargePower'].at[rows] ==self.initialPowerIn) \
                    & (readysolutions['Storage'].at[rows] == self.initialCapacity) \
                    & (readysolutions['DischargePower'].at[rows] == self.initialDischargePower):
                self.energyIn=float(readysolutions['EnergyIn'].at[rows])
                self.energyOut=float(readysolutions['EnergyOut'].at[rows])
                self.hoursCharging=float(readysolutions['HoursCharging'].at[rows])
                self.hoursDischarging=float(readysolutions['HoursDischarging'].at[rows])
                self.costBuying=float(readysolutions['CostBuying'].at[rows])
                self.profit=float(readysolutions['Profit'].at[rows])
                self.CO2=float(readysolutions['CO2'].at[rows])
                print('An old result was found and applied')
                return
        print('This is a new configuration.')
        self.getDispatchSolution()


    def getDispatchSolution(self):
        income = dispatchOptimizer()
        income.biddingCode=self.biddingZone
        income.maximumCapacity = self.initialCapacity
        income.maximumChargePower = self.initialPowerIn
        income.maximumDischargePower = self.initialDischargePower
        income.demandFile='DemandPulp.csv'
        income.stitchTime()
        self.dispatch = income.results
        del income
        gc.collect()
        self.saveResults()

    def saveResults(self):
        newRow= {'ChargePower': self.initialPowerIn,
                 'Storage': self.initialCapacity,
                 'DischargePower': self.initialDischargePower,
                 'EnergyIn':self.dispatch['buyingElectricity'].sum(),
                 'EnergyOut':self.dispatch['soldElectricity'].sum(),
                 'HoursCharging':self.dispatch['buyingElectricity'].gt(0).sum(),
                 'HoursDischarging':self.dispatch['soldElectricity'].gt(0).sum(),
                 'CostBuying':self.dispatch["Cost"].sum(),
                 'Profit':self.dispatch["Objective"].sum() * -1,
                 'CO2':(self.dispatch['CO2']*self.dispatch['buyingElectricity']).sum()
                       -(self.dispatch['CO2']*self.dispatch['soldElectricity']).sum(),
                 }
        newRowdf=pd.DataFrame(newRow, index=[0])
        newRowdf.to_csv(self.filename, mode='a',header=False,index=False)
        self.checkDispatchSolution()

    def getCashflow(self):
        investment = [round(self.capex / self.constructionTime)] * self.constructionTime
        self.yearlyValues = [round((self.profit) + (self.capexForOpex * self.opexPercentage))]
        if self.yearlyValues[0] < 0:
            self.infeasible()
        else:
            #self.profit = round((self.profit))
            self.cashflow = investment + (self.yearlyValues * self.lifetime) + [self.capex * self.decomissionRatio]
            self.possible=True

    def infeasible(self):
        self.output = {'Storage': self.initialCapacity, 'Power': self.initialPowerIn,
                       'PowerOut': self.initialDischargePower,'CAPEX': self.capex,
                       'Profit': self.profit, 'Costs': self.cost[self.constructionTime + 1],
                       'Yearly Cashflow': self.yearlyValues[0],
                       'LCOE': -9999999, 'IRR': -9999999, 'NPV': self.npvValue,
                       'Capacity Factor': -9999999, 'Utilization Factor': -9999999,
                       'CO2 Avoided': -9999999, 'RoundTrip':self.roundTripEfficiency,
                       'WACC':self.WACC}
        self.possible = False

    def getWACC(self):
        loan=(self.capex*self.debtPercent)/self.constructionTime
        repaymentSchedule=[0]*(1+self.constructionTime+self.lifetime+1)
        for years in range(1,self.constructionTime+1):
            oneloanschedule= [0] + ([0] * years) \
                            + [npf.pmt(self.debtInterest, self.lifetime + self.constructionTime - years, loan)]\
                            * (self.lifetime + self.constructionTime - years)+[0]
            repaymentSchedule= np.add(oneloanschedule,repaymentSchedule).tolist()

        investmentEquity = [0]+ ([(self.capex * self.equityPercent) / self.constructionTime] * self.constructionTime) \
                           + self.yearlyValues*(self.lifetime) + [self.capex*self.decomissionRatio]

        equityCashflow = np.subtract(investmentEquity,repaymentSchedule).tolist()
        print(sum(equityCashflow))

        self.equityIRR = npf.irr(equityCashflow)
        print(f'equity IRR is {self.equityIRR}')

        self.WACC = (self.equityPercent * self.equityIRR )+ (self.debtPercent * self.debtInterest * (1 - self.taxCorporate))
        if np.isnan(self.WACC) or self.equityIRR<0:
            self.infeasible()


    def getNPV(self):
        self.npvValue = round(npf.npv(self.WACC,
                                      self.cashflow))

    def getCostforLCOE(self):
        investment = [self.capex / self.constructionTime]
        cost = [0 - self.costBuying + (self.capexForOpex * self.opexPercentage)]
        decomissioning = [self.capex * self.decomissionRatio]

        self.cost = [0] + (investment * self.constructionTime) + (cost * self.lifetime) + decomissioning
        self.cost = [round(abs(number)) for number in self.cost]

    def getEnergy(self):
        energy = round(self.energyIn)
        # No energy production for the construction or the decomissioning
        self.energyflow = [0] + ([0] * self.constructionTime) + ([energy] * self.lifetime) + [0]

    def getLCOE(self):
        if sum(self.energyflow) == 0:  # todo: maybe try/catch is better fitted
            self.lcoe = -9999999  # if no power is sold, then lcoe is set to a high value to avoid zero division
            return
        else:
            lcoe_cost = 0
            lcoe_energy = 0
            for year in range(1+self.constructionTime+ self.lifetime+1):
                lcoe_cost += (self.cost[year] * -1) / (1 + self.WACC) ** year
                lcoe_energy += self.energyflow[year] * (1 - self.degradation) ** year / (1 + self.WACC) ** year
            self.lcoe = (lcoe_cost / lcoe_energy)  # LCOE is set to negative so that it can be maximized by PYGAD

    def getIRR(self):
        self.irr = npf.irr(self.cashflow)
        self.irr *= 100

    def getCapacityFactor(self):
        energy_production = self.energyOut
        theoretical_maximum = self.initialDischargePower*8760

        try:
            self.capacityFactor = energy_production / theoretical_maximum
        except ZeroDivisionError:
            self.capacityFactor=0
        self.capacityFactor *= 100

    def getUtilizationFactor(self):
        self.utilizationFactor = self.energyIn + self.energyOut
        try:
            self.utilizationFactor /= (8760 / 24) * 2 * self.initialCapacity
        except ZeroDivisionError:
            self.utilizationFactor =0
        self.utilizationFactor *= 100

    def getCO2Avoided(self):
        self.CO2 = round(self.CO2)

    def getRoundTripEfficiency(self):
        try:
            self.roundTripEfficiency=(self.energyOut*.8)/(self.energyIn)
        except:
            self.roundTripEfficiency=0#todo:find a way to circumvent the outgoing efficiency here set to 0.8
    def getLCOSBreakDown(self):
        levalizedCAPEX = 0
        for year in range(1,1+self.constructionTime):
            levalizedCAPEX += ((self.capex*-1)/self.constructionTime) / (1 + self.WACC) ** year
        levalizedBuyEnergy = 0
        levalizedOPEX = 0
        for year in range(self.constructionTime+1,self.constructionTime+self.lifetime+1):
            levalizedBuyEnergy += (self.costBuying) / (1 + self.WACC) ** year
            levalizedOPEX += (self.capexForOpex*self.opexPercentage*-1)/ (1 + self.WACC) ** year
        lcoe_energy = 0
        for year in range(1 + self.constructionTime + self.lifetime + 1):
            lcoe_energy += self.energyflow[year] * (1 - self.degradation) ** year / (1 + self.WACC) ** year

        levalizedCAPEX/= lcoe_energy
        levalizedBuyEnergy /= lcoe_energy
        levalizedOPEX /= lcoe_energy
        setwidth=0.5
        fig, ax = plt.subplots()
        ax.bar(1, (levalizedOPEX + levalizedBuyEnergy + levalizedCAPEX) , width=setwidth, label='OPEX')
        ax.bar(1, (levalizedBuyEnergy + levalizedCAPEX) , width=setwidth, label='Buying Energy')
        ax.bar(1, levalizedCAPEX , width=setwidth, label='CAPEX')
        ax.set_xlim(0,2.5)
        ax.get_xaxis().set_ticks([])
        ax.set_ylabel('LCOS (€/MWh)')
        ax.set_title('LCOS Breakdown')
        rects = ax.patches

        # Make some labels.
        labels = [round(levalizedOPEX),round(levalizedBuyEnergy),round(levalizedCAPEX)]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height -6, label, ha="center", va="bottom", fontsize=18
            )
        plt.legend()
        plt.show()


    def plotCapex(self):
        val = sorted(self.capexBreakDown.values(), reverse=False)[2]
        res = {k: v for k, v in self.capexBreakDown.items() if v >= val}

        others = sum(sorted(self.capexBreakDown.values(),reverse=False)[:2])
        res['Others']=others


        fig,ax=plt.subplots()
        ax.pie(res.values(),labels=res.keys(), autopct='%1.1f%%')

        plt.show()
    #def getPaybackTime(self):

    def plotEarnings(self):
        MILLION=1_000_000
        fig, ax= plt.subplots()
        negativecost =  [number*-1/MILLION for number in self.cost]
        #negativecost = [number+self.costBuying/MILLION for number in negativecost]
        negativecost[0]=0
        negativecost[33]=0
        ax.bar(range(1+self.constructionTime+self.lifetime+1),(negativecost),label='Outflows', color='r')
        balance=(self.profit+self.costBuying)/MILLION +negativecost[self.constructionTime+1]
        ax.bar(range(self.constructionTime+1,self.constructionTime+self.lifetime+1),(self.profit+self.costBuying)/MILLION,
               label='Inflow', color='b')
        ax.bar(range(self.constructionTime + 1, self.constructionTime + self.lifetime + 1),
               balance,label='Total', color='y')
        runningNPV=[0]
        print(self.WACC)
        cashflowmillion = [number/MILLION for number in self.cashflow]
        for year in range(1,1+self.constructionTime+self.lifetime+1):
            runningNPV.append(npf.npv(self.WACC,cashflowmillion[:year]))
        print(runningNPV)
        ax.plot(runningNPV,label='NPV')
        ax.set_title('Cash Flow and NPV Over Lifetime')
        ax.set_xlabel('Year')
        ax.set_ylabel('Cash Flow (M€)')
        discountPayback=[]
        for x in range(len(runningNPV)-1):
            if runningNPV[x+1]>0 and runningNPV[x]<0:
                print(x)
                discountPayback.append(x)

        if runningNPV[len(runningNPV)-1]>0:
            ax.annotate(f'DPP={discountPayback[0]} years', (discountPayback[0], 0),
                            xycoords='data',
                            xytext=(0.7, 0.4), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='-', facecolor='black', linewidth=.1),
                            horizontalalignment='right', verticalalignment='top'
                            )
        ax.annotate(f'NPV={round(runningNPV[len(runningNPV)-1]) } M€', (len(runningNPV)-1, runningNPV[len(runningNPV)-1]),
                    xycoords='data',
                    xytext=(0.9, 0.4), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', facecolor='black', linewidth=.1),
                    horizontalalignment='right', verticalalignment='top'
                    )
        print(self.WACC)



        plt.legend()
        plt.show()



class capexMoltenSalt:
    def __init__(self, targetCapacity, targetPowerOut=0):
        self.value = None
        self.CEPCI20122022 = 806.3 / 584.6
        self.CEPCI20212022 =806.2/686.7
        self.storageReferenceSize = 1745  # MWh
        self.dollarsToEuros2012 = 0.8207
        self.dollarsToEuros2021 = 0.8458
        self.tanks = 42.888_000 * self.CEPCI20122022 * self.dollarsToEuros2012  # M$
        self.pipes = 1.418_000 * self.CEPCI20122022 * self.dollarsToEuros2012
        self.foundation = .520_000 * self.CEPCI20122022 * self.dollarsToEuros2012
        self.pumpsHEX = 29.766_000 * self.CEPCI20122022 * self.dollarsToEuros2012
        self.instruments = 5.677_000 * self.CEPCI20122022 * self.dollarsToEuros2012
        self.salt = 2.65 * 0.003600 *self.CEPCI20212022*self.dollarsToEuros2021 # M$2021/MW
        self.targetCapacity = targetCapacity
        self.alphaStorage = 0.8

        self.powerBlockUnit=(941 *self.CEPCI20212022*self.dollarsToEuros2012*1_000)/(1_000_000) #M€/MW
        # It was originally in dollar/kw
        self.targetPowerOut =targetPowerOut

        self.buildCapex()
        return

    def sizingFunction(self, cost):
        return cost * ((self.targetCapacity / self.storageReferenceSize) ** self.alphaStorage)

    def buildCapex(self):
        self.value = round(self.sizingFunction(self.tanks)
                           + self.sizingFunction(self.pipes)
                           + self.sizingFunction(self.foundation)
                           + self.sizingFunction(self.pumpsHEX)
                           + self.sizingFunction(self.instruments)
                           + (self.salt * self.targetCapacity))
        self.value += self.powerBlockUnit*self.targetPowerOut
        self.value *= 1_000_000

        self.valueBreakDown={'Tanks':self.sizingFunction(self.tanks),'Pipes':self.sizingFunction(self.pipes),
                             'Foundation':self.sizingFunction(self.foundation),
                             'PumpHEX':self.sizingFunction(self.pumpsHEX),
                             'Instruments':self.sizingFunction(self.instruments),
                             'Salt':(self.salt * self.targetCapacity),
                             'PowerBlock':self.powerBlockUnit*self.targetPowerOut}
        self.valueBreakDown={k:round(v*1_000_000) for (k,v) in self.valueBreakDown.items()}


if __name__ == "__main__":
    """x=sizingOptimizer()
    ROPowerBlocks = [150, 60, 315, 235, 210, 50, 330]

    for powerblock in ROPowerBlocks:
        for heaterfactor in [0.1, 0.3, 1, 3, 10]:
            for storagehours in [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24]:"""
    x=sizingOptimizer()
    x.initialCapacity= 7560
    x.initialPowerIn= 3150
    x.initialDischargePower=315
    #x.WACCMethod='Lantz'
    x.payForPowerBlock=25/100
    x.buildFinancial()
    x.plotEarnings()
    x.plotCapex()
    print(x.capex/1_000_000)
    print(x.npvValue/1000000)

