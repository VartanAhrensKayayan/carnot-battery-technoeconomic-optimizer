# import CoolProp.CoolProp as CP
# import CoolProp.Plots as CPP
import datetime
import gc
import math
import os.path
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from entsoe import EntsoePandasClient
import random
from math import ceil
import openpyxl
import pprint
import numpy_financial as npf
from mpl_toolkits import mplot3d
import os.path
import csv
from pyomo.opt.results import SolverStatus



class dispatchOptimizer():

    def __init__(self):
        self.threads = 4
        print('Dispatch started')


        self.solveFor = 'Cost' #The model can be solved to minized 'Costs' or 'Carbon'

        self.timelimit = 300  # int below 999 which limits the model running time


        self.electricityPrice = pd.DataFrame()
        self.electricityPriceSlice = pd.DataFrame()
        self.carboneq = pd.DataFrame()
        self.carboneqSlice = pd.DataFrame()
        f = open("MyAPI.txt", "r")
        self.myapi = f.read()
        f.close()
        self.length = 8760
        # Running for the full year overflows so it is divided into weekly periods
        self.lengthBreakdown = 168

        self.model = None
        self.timezone = 'Europe/Stockholm'
        self.year = 2021
        self.startDateTimestr = f'{self.year}-01-01T00'
        self.startDateTime = pd.Timestamp(f'{self.startDateTimestr}', tz=f'{self.timezone}')
        self.biddingCode = 'RO'


        self.intoEfficiency = 0.8
        self.outtoEfficiency = 0.8
        self.maximumCapacity = 3600
        self.minimumCapacity = self.maximumCapacity * 0.1
        self.maximumChargePower = 1500
        self.minimumChargePower = self.maximumChargePower * 0.1
        self.startCharge = self.minimumCapacity * 1.1
        self.hoursProcessed = 0
        self.maximumDischargePower = 150
        self.minimumDischargePower = self.maximumDischargePower * 0.2
        self.maximumDischargePowerToLoad = 0

        self.MaxRampUp = 0.5 * self.maximumChargePower
        self.MaxRampDown = 0.5 * self.maximumChargePower

        self.MaxRampUpDischarge = 0.5 * self.maximumDischargePower
        self.MaxRampDownDischarge = 0.5 * self.maximumDischargePower

        self.mintimeup = 4
        self.mintimedown = 3

        self.mintimeupDischarge = 5
        self.mintimedownDischarge = 5

        self.currentDateTime = self.startDateTime

        self.opt = None
        self.results = pd.DataFrame()

        self.NEOS =False

    def getElectricityPriceQuickly(self):
        if os.path.isfile(f'{self.biddingCode}_{self.year}.csv'):
            temp = pd.read_csv(f'{self.biddingCode}_{self.year}.csv')
            if self.length != 8760:
                temp.drop(temp.index[self.length:8760], inplace=True)
            Timespan = pd.date_range(start=self.startDateTime,
                                     periods=self.length,
                                     freq="H", tz='Europe/Stockholm')
            temp['Timespan'] = Timespan
            temp.set_index('Timespan', inplace=True)
            self.electricityPrice = temp
        else:
            self.prepareElectricityPrice()
        return self.electricityPrice

    def getElectricityPrice(self):

        """

        :param apikey: str API given out by ENTSO-E
        :param startdatetime: str in the format yyyy-mm-ddThh
        :param length: int number of days
        :param bidding_code: the bidding zone code for the researched area
        :param info: Boolean if the data summary should be presented.
        :return: dataframe for the electricity pricing
        """

        client = EntsoePandasClient(api_key=self.myapi)
        start = self.startDateTime

        Base = pd.DataFrame()
        Timespan = pd.date_range(start=self.startDateTime,
                                 periods=self.length + 1,
                                 freq="H", tz='Europe/Stockholm')
        end = Timespan.max()
        Base['Timespan'] = Timespan
        Base.set_index('Timespan', inplace=True)
        Base['Price'] = client.query_day_ahead_prices(self.biddingCode, start=start, end=end)
        Base.drop(end, inplace=True)

        self.electricityPrice = Base
        return self.electricityPrice

    def prepareElectricityPrice(self):
        if not os.path.isfile(f'{self.biddingCode}_{self.year}.csv'):
            self.getElectricityPrice()
            self.electricityPrice.to_csv(f'{self.biddingCode}_{self.year}.csv', index=False)

    def sliceElectricityPrice(self): #todo: combine the slicing and fix the 53 weeks problem
        slicerStart = self.currentDateTime
        slicerEnd = self.currentDateTime + datetime.timedelta(hours=self.lengthBreakdown - 1)

        self.electricityPriceSlice = \
            self.electricityPrice.loc[slicerStart:
                                      min(slicerEnd, self.startDateTime + datetime.timedelta(hours=self.length))]
        return self.electricityPriceSlice

    def sliceCarbonEmissions(self):
        slicerStart = self.currentDateTime
        slicerEnd = self.currentDateTime + datetime.timedelta(hours=self.lengthBreakdown - 1)

        self.carboneqSlice = \
            self.carboneq.loc[slicerStart:
                              min(slicerEnd, self.startDateTime + datetime.timedelta(hours=self.length))]
        return self.carboneqSlice

    def getCarbonEmissions(self):
        temp = pd.read_csv('CarbonDummy.csv')
        if self.length != 8760:
            temp.drop(temp.index[self.length:8760], inplace=True)
        Timespan = pd.date_range(start=self.startDateTime,
                                 periods=self.length,
                                 freq="H", tz='Europe/Stockholm')
        temp['Timespan'] = Timespan
        temp.set_index('Timespan', inplace=True)
        self.carboneq = temp



    def presentPriceInformation(self):
        plt.figure(0)
        plt.plot(self.electricityPrice['Price'])
        plt.axhline(np.nanmean(self.electricityPrice['Price']), color='r')
        largest = self.electricityPrice['Price'].max()
        smallest = self.electricityPrice['Price'].min()
        mean = self.electricityPrice['Price'].mean()
        print(
            f'The maximum price is {largest} EUR/MWh and the minimum is {smallest} EUR/MWh. '
            f'The mean is {mean:.2f} EUR/MWh.')
        plt.plot(self.electricityPrice)
        plt.figure(1)
        plt.hist(standard.electricityPrice, bins=50)
        plt.show()
        return

    def setParemeters(self):
        self.maximumStateofCharge = self.maximumCapacity*0.9
        self.minimumCapacity = int(self.maximumCapacity * 0.1)
        self.minimumChargePower = self.maximumChargePower * 0.1

        self.minimumDischargePower = round(self.maximumDischargePower * 0.2)
        #if self.maximumCapacity < self.minimumDischargePower:
        #    self.minimumDischargePower = round(self.minimumCapacity*0.1)

        self.MaxRampUp = 0.5 * self.maximumChargePower
        self.MaxRampDown = 0.5 * self.maximumChargePower
        self.startCharge = int(math.ceil(self.minimumCapacity * 1.1))

        self.MaxRampUpDischarge = 0.5 * self.maximumDischargePower
        self.MaxRampDownDischarge = 0.5 * self.maximumDischargePower

    def buildDispatchModel(self):
        # Set time period
        self.model = pyo.ConcreteModel()
        self.model.T = pyo.Set(initialize=pyo.RangeSet(len(self.electricityPriceSlice)), ordered=True)

        # PARAMETERS:
        self.model.maximumStateofCharge = pyo.Param(initialize=self.maximumStateofCharge)
        self.model.intoEfficiency = pyo.Param(initialize=self.intoEfficiency)
        self.model.outtoEfficiency = pyo.Param(initialize=self.outtoEfficiency)
        self.model.minimumCapacity = pyo.Param(initialize=self.minimumCapacity, mutable=True)
        self.model.maximumCapacity = pyo.Param(initialize=self.maximumCapacity, mutable=True)
        self.model.maximumChargePower = pyo.Param(initialize=self.maximumChargePower, mutable=True)
        self.model.minimumChargePower = pyo.Param(initialize=self.minimumChargePower, mutable=True)
        self.model.maximumDischargePower = pyo.Param(initialize=self.maximumDischargePower)
        self.model.minimumDischargePower = pyo.Param(initialize=self.minimumDischargePower)
        if self.hoursProcessed != 0:
            self.startCharge = float(self.results["SOC"].tail(n=1))
        self.model.startCharge = pyo.Param(initialize=self.startCharge)
        self.model.hourlyLoss = pyo.Param(initialize=0.95 ** (1 / 24))
        self.model.demand = pyo.Param(self.model.T, initialize=0)
        self.model.spotPrices = pyo.Param(self.model.T,
                                          initialize=dict(enumerate(self.electricityPriceSlice["Price"], 1)),
                                          within=pyo.Reals, mutable=True)
        self.model.carboneq = pyo.Param(self.model.T,
                                        initialize=dict(enumerate(self.carboneqSlice["CO2"], 1)),
                                        within=pyo.Reals, mutable=True)
        self.model.MaxRampUp = pyo.Param(initialize=self.MaxRampUp)
        self.model.MaxRampDown = pyo.Param(initialize=self.MaxRampDown)
        self.model.MinTimeUp = pyo.Param(initialize=self.mintimeup)
        self.model.MinTimeDown = pyo.Param(initialize=self.mintimedown)

        self.model.MaxRampUpDischarge = pyo.Param(initialize=self.MaxRampUpDischarge)
        self.model.MaxRampDownDischarge = pyo.Param(initialize=self.MaxRampDownDischarge)
        self.model.MinTimeUpDischarge = pyo.Param(initialize=self.mintimeupDischarge)
        self.model.MinTimeDownDischarge = pyo.Param(initialize=self.mintimedownDischarge)

        # Variables : also the variable has to  be indexed with the time T
        self.model.SOC = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                 bounds=(self.model.minimumCapacity, self.model.maximumStateofCharge),
                                 initialize=self.startCharge)

        self.model.dischargeToLoad = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                             bounds=(0, self.maximumDischargePowerToLoad),
                                             initialize=0)

        self.model.charge = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                    bounds=(0, self.model.maximumChargePower),
                                    initialize=0)

        self.model.buyingElectricity = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                               bounds=(0, self.model.maximumChargePower / self.model.intoEfficiency),
                                               initialize=0)

        self.model.soldElectricity = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                             bounds=(0,
                                                     self.model.maximumDischargePower),
                                             initialize=0)
        self.model.onOffSwitch = pyo.Var(self.model.T, domain=pyo.Binary, initialize=1)
        self.model.dischargeOnOff = pyo.Var(self.model.T, domain=pyo.Binary, initialize=1)
        self.model.noDischargeAllowed = pyo.Var(self.model.T, domain=pyo.Binary, initialize=1)

    def constrainModel(self):

        def charge_indicator(model, t):
            return self.model.maximumChargePower * self.model.onOffSwitch[t] >= model.charge[t]

        self.model.switchCharge = pyo.Constraint(self.model.T, rule=charge_indicator)

        def charge_indication_floor(model, t):
            return self.model.onOffSwitch[t] * self.model.minimumChargePower <= self.model.charge[t]

        self.model.switchChargefloor = pyo.Constraint(self.model.T, rule=charge_indication_floor)

        def charge_leads_no_discharge(model, t):
            return self.model.onOffSwitch[t] <= self.model.noDischargeAllowed[t]

        self.model.charge_leads_no_discharge_c = pyo.Constraint(self.model.T, rule=charge_leads_no_discharge)

        def discharge_constraint_maximum(model, t):
            return self.model.maximumDischargePower * ((self.model.noDischargeAllowed[t] - 1) * -1) \
                   >= self.model.dischargeToLoad[t] + self.model.soldElectricity[t]
        self.model.discharge_constraint_maximum =pyo.Constraint(self.model.T, rule=discharge_constraint_maximum)

        def discharge_on_off(model, t):
            return self.model.maximumDischargePower * self.model.dischargeOnOff[t] \
                   >= self.model.dischargeToLoad[t] + self.model.soldElectricity[t]

        self.model.discharge_max = pyo.Constraint(self.model.T, rule=discharge_on_off)

        def discharge_on_off_min(model, t):
            return self.model.minimumDischargePower * self.model.dischargeOnOff[t] \
                   <= self.model.dischargeToLoad[t] + self.model.soldElectricity[t]

        self.model.discharge_min = pyo.Constraint(self.model.T, rule=discharge_on_off_min)

        def nocharge_discharge_constrain(model, t):
            return ((self.model.noDischargeAllowed[t] - 1) * -1) * self.model.minimumDischargePower \
                   <= self.model.dischargeToLoad[t] + self.model.soldElectricity[t]

        self.model.nocharge_discharge_c = pyo.Constraint(self.model.T, rule=nocharge_discharge_constrain)

        def purchase_constraint(model,
                                t):  # TODO change this equation so there are two dischargeToLoad, toLoad and to Grid
            return self.model.buyingElectricity[t] == self.model.demand[t] \
                   - (self.model.dischargeToLoad[t] * self.model.outtoEfficiency) \
                   + (self.model.charge[t])

        self.model.purchase_c = pyo.Constraint(self.model.T, rule=purchase_constraint)

        def soc_constraint(model, t):
            if t == self.model.T.first():
                return self.model.SOC[t] == int(math.ceil(self.startCharge))
            else:
                return self.model.SOC[t] == self.model.SOC[t - 1] - self.model.dischargeToLoad[t] \
                       + (self.model.charge[t] * self.model.intoEfficiency) \
                       - (self.model.SOC[t - 1] * (1 - self.model.hourlyLoss)) \
                       - self.model.soldElectricity[t]

        self.model.soc_c = pyo.Constraint(self.model.T, rule=soc_constraint)

        def discharge_constraint_emptying(model, t):
            return self.model.dischargeToLoad[t] + self.model.soldElectricity[t] <= self.model.SOC[t] \
                   - self.model.minimumCapacity

        self.model.discharge_empty = pyo.Constraint(self.model.T, rule=discharge_constraint_emptying)

        def charge_constraint(model, t):
            return self.model.charge[t] <= self.model.maximumStateofCharge - self.model.SOC[t]

        self.model.charge_c = pyo.Constraint(self.model.T, rule=charge_constraint)

        def rampUpLimit(model, t):
            if t == self.model.T.first():
                return self.model.charge[t] <= self.model.MaxRampUp
            else:
                return self.model.charge[t] - self.model.charge[t - 1] <= self.model.MaxRampUp

        self.model.rampUp_c = pyo.Constraint(self.model.T, rule=rampUpLimit)

        def rampDownLimit(model, t):
            if t == self.model.T.first():
                return self.model.charge[t] <= self.model.MaxRampDown
            else:
                return self.model.charge[t - 1] - self.model.charge[t] <= self.model.MaxRampDown

        self.model.rampDown_c = pyo.Constraint(self.model.T, rule=rampDownLimit)

        def timeUpCommitment(model, t):
            commitUp = 0

            for t_star in range(t, min(t + int(model.MinTimeUp), len(self.electricityPriceSlice) - 1)):
                commitUp += self.model.onOffSwitch[t_star]
            if t == self.model.T.first():
                return model.MinTimeUp * (model.onOffSwitch[t]) <= commitUp
            else:
                return model.MinTimeUp * (model.onOffSwitch[t] - model.onOffSwitch[t - 1]) <= commitUp

        self.model.timeUpCommitmenCharge = pyo.Constraint(self.model.T, rule=timeUpCommitment)


        def timeDownCommitment(model, t):
            commitDownCharge = 0
            for t_star in range(t, min(t + self.model.MinTimeDown, len(self.electricityPriceSlice) - 1)):
                commitDownCharge += (self.model.onOffSwitch[t_star]-1)*-1
            if t == self.model.T.first():
                return self.model.MinTimeDown * self.model.onOffSwitch[t] <= commitDownCharge
            else:
                return self.model.MinTimeDown * (self.model.onOffSwitch[t-1]
                                                          - self.model.onOffSwitch[t]) <= commitDownCharge

        self.model.timeDownCommitmentCharge = pyo.Constraint(self.model.T, rule=timeDownCommitment)

        def rampUpLimitDischarge(model, t):
            if t == self.model.T.first():
                return self.model.dischargeToLoad[t] + self.model.soldElectricity[t] <= self.model.MaxRampUpDischarge
            else:
                return (self.model.dischargeToLoad[t] + self.model.soldElectricity[t]) \
                       - (self.model.dischargeToLoad[t - 1] + self.model.soldElectricity[t - 1]) \
                       <= self.model.MaxRampUpDischarge

        self.model.rampUpDischarge = pyo.Constraint(self.model.T, rule=rampUpLimitDischarge)

        def rampDownLimitDischarge(model, t):
            if t == self.model.T.first():
                return self.model.dischargeToLoad[t] + self.model.soldElectricity[t] <= self.model.MaxRampDownDischarge
            else:
                return (self.model.dischargeToLoad[t - 1] + self.model.soldElectricity[t - 1]) \
                       -(self.model.dischargeToLoad[t] + self.model.soldElectricity[t]) \
                       <= self.model.MaxRampDownDischarge

        self.model.rampDownDischarge = pyo.Constraint(self.model.T, rule=rampDownLimitDischarge)

        def timeUpCommitmentDischarge(model, t):
            commitUpDischarge = 0
            for t_star in range(t, min(t + int(self.model.MinTimeUpDischarge), len(self.electricityPriceSlice) - 1)):
                commitUpDischarge += self.model.dischargeOnOff[t_star]
            if t == self.model.T.first():
                return self.model.MinTimeUpDischarge * self.model.dischargeOnOff[t] <= commitUpDischarge
            else:
                return self.model.MinTimeUpDischarge * (self.model.dischargeOnOff[t]
                                                   - self.model.dischargeOnOff[t-1]) <= commitUpDischarge

        self.model.timeUpCommitmentDischarge = pyo.Constraint(self.model.T, rule=timeUpCommitmentDischarge)

        def timeDownCommitmentDischarge(model, t):
            commitDownDischarge = 0
            for t_star in range(t, min(t + self.model.MinTimeDownDischarge, len(self.electricityPriceSlice) - 1)):
                commitDownDischarge += (self.model.dischargeOnOff[t_star]-1)*-1
            if t == self.model.T.first():
                return self.model.MinTimeDownDischarge * self.model.dischargeOnOff[t] <= commitDownDischarge
            else:
                return self.model.MinTimeDownDischarge * (self.model.dischargeOnOff[t-1]
                                                          - self.model.dischargeOnOff[t]) <= (commitDownDischarge)

        self.model.timeDownCommitmentDischarge = pyo.Constraint(self.model.T, rule=timeDownCommitmentDischarge)

        def costs(model):
            return sum(((self.model.buyingElectricity[t] * self.model.spotPrices[t])
                        - (self.model.soldElectricity[t] * self.model.spotPrices[t] * self.model.outtoEfficiency))
                       for t in model.T)

        def carbon(model):
            return sum((self.model.buyingElectricity[t] * self.model.carboneq[t])
                       + (self.model.soldElectricity[t] * self.model.carboneq[t] * -1) for t in model.T)

        if self.solveFor == 'Cost':
            self.model.objective = pyo.Objective(rule=costs, sense=pyo.minimize)
        elif self.solvefor == 'Carbon':
            self.model.objective = pyo.Objective(rule=carbon, sense=pyo.minimize)
        else:
            self.model.objective = pyo.Objective(rule=costs, sense=pyo.minimize)


        if pyo.SolverFactory('cbc').available():
            self.opt = pyo.SolverFactory('cbc')
            self.opt.options['seconds'] = self.timelimit
            self.opt.options['threads'] = self.threads




        else:
            self.opt = pyo.SolverFactory('glpk')
            self.opt.options['tmlim'] = self.timelimit

    def solveModel(self):

        if pyo.SolverFactory('cbc').available():
            results = self.opt.solve(self.model, warmstart=True)
            print(results.Solver.Status)
            results.Solver.Status = SolverStatus.warning

            self.model.solutions.load_from(results)
        else:
            results = self.opt.solve(self.model)

        # getting the results out so that the further optimization can be carried out
        resultsdf = pd.DataFrame()

        for v in self.model.component_objects(pyo.Var, active=True):
            for index in v:
                resultsdf.at[index, v.name] = pyo.value(v[index])

        self.results = pd.concat([self.results, resultsdf])
        return

    def stitchTime(self):
        self.setParemeters()
        self.hoursProcessed = 0
        self.getElectricityPriceQuickly()
        self.getCarbonEmissions()
        self.sliceElectricityPrice()
        self.sliceCarbonEmissions()
        self.buildDispatchModel()
        self.constrainModel()
        while self.hoursProcessed < self.length:#todo: for loop with hours processed in length breakdown steps to length
            self.sliceElectricityPrice()
            self.sliceCarbonEmissions()
            self.resetModel()
            try:
                self.solveModel()
            except:
                self.solveModel()
            self.currentDateTime += datetime.timedelta(hours=self.lengthBreakdown)
            self.hoursProcessed += self.lengthBreakdown

            gc.collect()

        Timespan = pd.date_range(start=self.startDateTime,
                                 periods=self.length,
                                 freq="H", tz='Europe/Stockholm')
        self.results=self.results.head(n=self.length)
        self.results.reset_index(inplace=True, drop=True)
        # self.results.drop("index", axis='columns', inplace=True)
        self.results.set_index(Timespan, inplace=True)
        self.results["Price"] = self.electricityPrice["Price"]
        self.results["Cost"] = self.results["buyingElectricity"] * self.results["Price"]
        self.results["Incomes"] = self.results["soldElectricity"] * self.results["Price"] * self.outtoEfficiency
        self.results["Objective"] = self.results["Cost"] - self.results["Incomes"]
        self.results["Objective"] = self.results["Objective"].astype(int)
        self.results["CO2"]=self.carboneq
        self.dropModel()
        gc.collect()
        return self.results

    def dropModel(self):
        self.model = None

    def resetModel(self):
        self.electricityPriceSlice.reset_index(inplace=True, drop=True)
        self.carboneqSlice.reset_index(inplace=True, drop=True)
        for t in range(1, len(self.electricityPriceSlice) + 1):
            self.model.spotPrices[t] = \
                self.electricityPriceSlice["Price"].at[t - 1]
            self.model.carboneq[t] = self.carboneqSlice['CO2'].at[t - 1]

    def presentResults(self):
        plt.figure(0)
        plt.plot(self.results["SOC"] / self.maximumCapacity)
        plt.ylim((0, 1))
        plt.xlabel('Time')
        plt.ylabel('Normalized SoC (MWh/MWh)')
        plt.figure(1)
        self.results.reset_index(inplace=True)
        plt.plot(self.results.loc[168*51:, "SOC"] / self.maximumCapacity)
        plt.ylim((0,1))
        plt.xlabel('Time (hours)')
        plt.ylabel('Normalized SoC (MWh/MWh)')  # Todo: make the timeslice in timestamps
        plt.figure(2)
        plt.plot(self.results["Objective"])
        plt.title('Cashflow in France')
        plt.xlabel('Time (hours)')
        plt.ylabel('Cost (€)')
        print(f' the objective is {self.results["Objective"].sum()}')
        print(list(self.results.columns))
        plt.figure(3)
        fig, ax = plt.subplots()
        ax.bar(np.linspace(1, len(self.results['charge']), num=self.length), self.results["soldElectricity"]*-1,
               label='Discharge', color='g')
        ax.bar(np.linspace(1, len(self.results['charge']), num=self.length), self.results["charge"], label='Charge')

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (MW)')
        ax2 = ax.twinx()
        ax2.plot(self.results["SOC"] / self.maximumCapacity, label='State of Charge', color ='r')
        ax2.set_ylim((0, 1))
        ax2.set_ylabel('Normalized SoC (MWh/MWh)')
        ax.legend()
        ax2.legend()

        plt.show()
    def plotAllWeeks(self):
        self.results.reset_index(inplace=True)
        # standard.results=pd.read_csv('quick_results.csv')
        fig, axs = plt.subplots(13, 4, sharey=True)
        firstrow = 0
        secondrow = 0
        for each in range(52):
            axs[firstrow, secondrow].plot((self.results['SOC'].loc[((firstrow * secondrow) + firstrow) * 168:
                                                                       ((firstrow * secondrow) + firstrow) * 168 + 168])
                                          / self.results.SOC.max())
            axs[firstrow,secondrow].get_xaxis().set_ticks([])
            firstrow += 1
            if firstrow == 13:
                secondrow += 1
                firstrow = 0

        plt.show()


if __name__ == "__main__":

    standard = dispatchOptimizer()
    standard.timelimit = 1
    standard.maximumCapacity= 2450
    standard.maximumChargePower=   161
    standard.maximumDischargePower=   50
    standard.stitchTime()
    standard.presentResults()
