import collections
import datetime

import matplotlib.pyplot as plt
from entsoe.exceptions import NoMatchingDataError
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('viridis', 8)

from entsoe import EntsoePandasClient

import pandas as pd
import numpy as np


class CoalPowerPlant:
    def __init__(self):
        self.closingCoal = pd.read_csv('Coal.csv', quotechar="'")
        f = open("MyAPI.txt", "r")
        self.myapi = f.read()
        f.close()
        self.length = 8760
        # Running for the full year overflows so it is divided into weekly periods
        self.lengthBreakdown = 168

        self.timezone = 'Europe/Stockholm'
        self.year = 2021
        self.startDateTimestr = f'{self.year}-01-01T00'
        self.startDateTime = pd.Timestamp(f'{self.startDateTimestr}', tz=f'{self.timezone}')
        self.biddingCode = 'RO'
        self.nameCode = {'CzechRepublic': 'CZ', 'Germany': 'DE_LU', 'Denmark': 'DK', 'Greece': 'GR', 'Spain': 'ES',
                         'Finland': 'FI', 'France': 'FR', 'Hungary': 'HU', 'Italy': 'IT', 'Netherlands': 'NL',
                         'Poland': 'PL',
                         'Slovenia': 'SI', 'Slovakia': 'SK', 'UnitedKingdom': 'UK', 'Bulgaria': 'BG', 'Croatia': 'HR',
                         'Ireland': 'IE', 'Montenegro': 'ME', 'NorthMacedonia': 'MK', 'Romania': 'RO',
                         'Bosnia&Herzegovina': 'BA', 'Kosovo': 'XK', 'Serbia': 'RS', 'Turkey': 'TR'}

    def dropNoRetirementsAndPlanned(self):
        self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['RetirementYear'] == 9999].index, inplace=True)
        self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['RetirementYear'].isna()].index, inplace=True)

    def dropNoRetirementsAndAnnounced(self):
            self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['RetirementYear'] == 9999].index, inplace=True)
            self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['RetirementPlan'].isna()].index, inplace=True)
    def dropNoRetirements(self):
        self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['RetirementYear'] == 9999].index, inplace=True)
        self.closingCoal.fillna(value=0, inplace=True)
        self.closingCoal.RetirementYear.replace(to_replace=8888, value=2030, inplace=True)
        self.closingCoal['AnnounceOrPlanned']=self.closingCoal.RetirementYear+self.closingCoal.RetirementPlan
        self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['AnnounceOrPlanned'] == 0].index, inplace=True)

    def CountryPiePlot(self):
        self.dropNoRetirementsAndPlanned()
        print(len(self.closingCoal))
        countrylist = self.closingCoal.Country.unique().tolist()
        countrycount = {}
        for country in countrylist:
            counter = self.closingCoal.Country.str.count(f'{country}')
            counter = counter.sum()
            countrycount[f'{country}'] = counter

        countrycapacity = {}
        for country in countrylist:
            tempdropothers = self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['Country'] != country].index)
            power = tempdropothers.Capacity.sum()
            countrycapacity[f'{country}'] = power
        cutofflast=6
        cutoffcount = sorted(countrycount.values(), reverse=False)[cutofflast]
        countrycountleft = {k: v for k, v in countrycount.items() if v >= cutoffcount}
        others = sum(sorted(countrycount.values(), reverse=False)[:cutofflast])
        countrycountleft['Others'] = others

        cutoffcap = sorted(countrycapacity.values(), reverse=False)[cutofflast]
        countrycapleft = {k: v for k, v in countrycapacity.items() if v >= cutoffcap}
        others = sum(sorted(countrycapacity.values(), reverse=False)[:cutofflast])
        countrycapleft['Others'] = others

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{v:d}'.format(p=pct, v=val)

            return my_autopct

        fig, ax = plt.subplots(2,2)
        cmap = plt.get_cmap("tab20c")
        outercolors = cmap(np.arange(len(countrycapleft)))
        ax[0,0].pie(countrycountleft.values(), labels=countrycountleft.keys(),
                  wedgeprops=dict(width=1, edgecolor='w'), colors=outercolors,
                    autopct=make_autopct(countrycountleft.values()), pctdistance=0.8, labeldistance=1.1)
        ax[0,0].title.set_text('Announced Coal-Fired Retirement \n By Number of Units')
        ax[0,1].pie(countrycapleft.values(), labels=countrycapleft.keys(),
                  wedgeprops=dict(width=1, edgecolor='w'), colors=outercolors,
                    autopct=make_autopct(countrycapleft.values()), pctdistance=0.75, labeldistance=1.1)
        ax[0,1].title.set_text('Announced Coal-Fired Retirement \n By Capacity (MW)')
        self.closingCoal = pd.read_csv('Coal.csv', quotechar="'")
        self.dropNoRetirementsAndAnnounced()
        print(len(self.closingCoal))
        countrylist = self.closingCoal.Country.unique().tolist()
        countrycount = {}
        for country in countrylist:
            counter = self.closingCoal.Country.str.count(f'{country}')
            counter = counter.sum()
            countrycount[f'{country}'] = counter

        countrycapacity = {}
        for country in countrylist:
            tempdropothers = self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['Country'] != country].index)
            power = tempdropothers.Capacity.sum()
            countrycapacity[f'{country}'] = power
        cutofflast=6
        cutoffcount = sorted(countrycount.values(), reverse=False)[cutofflast]
        countrycountleft = {k: v for k, v in countrycount.items() if v >= cutoffcount}
        others = sum(sorted(countrycount.values(), reverse=False)[:cutofflast])
        countrycountleft['Others'] = others

        cutoffcap = sorted(countrycapacity.values(), reverse=False)[cutofflast]
        countrycapleft = {k: v for k, v in countrycapacity.items() if v >= cutoffcap}
        others = sum(sorted(countrycapacity.values(), reverse=False)[:cutofflast])
        countrycapleft['Others'] = others

        ax[1,0].pie(countrycountleft.values(), labels=countrycountleft.keys(),
                  wedgeprops=dict(width=1, edgecolor='w'), colors=outercolors,
                    autopct=make_autopct(countrycountleft.values()), pctdistance=0.8, labeldistance=1.1)
        ax[1,0].title.set_text('Planned Coal-Fired Retirement \n By Number of Units')
        ax[1,1].pie(countrycapleft.values(), labels=countrycapleft.keys(),
                  wedgeprops=dict(width=1, edgecolor='w'), colors=outercolors,
                    autopct=make_autopct(countrycapleft.values()), pctdistance=0.75, labeldistance=1.1)
        ax[1,1].title.set_text('Planned Coal-Fired Retirement \n By Capacity  (MW)')



        plt.show()

    def PlotTimeline(self):
        fig, ax =plt.subplots()
        self.dropNoRetirements()
        retirementyearlist = self.closingCoal.AnnounceOrPlanned.unique().tolist()
        retirementAmount = {}
        for year in retirementyearlist:
            temp = self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['AnnounceOrPlanned'] != year].index)
            power = temp.Capacity.sum()
            retirementAmount[f'{round(year)}'] = power
        for year in range(2043,2048):
            retirementAmount[f'{year}']=0
        retirementAmount[f'{2037}'] = 0
        retirementAmount[f'{2039}'] = 0
        retirementAmount = collections.OrderedDict(sorted(retirementAmount.items()))
        ax.bar(retirementAmount.keys(), retirementAmount.values(),label='Planned')

        self.closingCoal = pd.read_csv('Coal.csv', quotechar="'")
        self.dropNoRetirementsAndPlanned()
        self.closingCoal.RetirementYear.replace(to_replace=8888, value=2030, inplace=True)
        retirementyearlist = self.closingCoal.RetirementYear.unique().tolist()
        retirementAmount = {}
        for year in retirementyearlist:
            temp = self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['RetirementYear'] != year].index)
            power = temp.Capacity.sum()
            retirementAmount[f'{round(year)}'] = power
        for year in range(2043, 2048):
            retirementAmount[f'{year}'] = 0


        retirementAmount = collections.OrderedDict(sorted(retirementAmount.items()))
        ax.bar(retirementAmount.keys(), retirementAmount.values(), label='Announced')
        ax.legend()
        ax.set_xlabel('Year')
        ax.set_ylabel('Capacity Retired (MW)')
        ax.set_title('Coal-Fired Power Plants Capacity Announced or Planned to Retire')


        plt.show()

    def getOneCountryList(self, countrytarget='Poland'):

        if not countrytarget in self.nameCode.keys() or self.nameCode.values(): #todo: logic gate is incorrect
            print('Please check the spelling. The country is not in the list of countries or country codes.')

        onlyTarget = self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['Country'] != countrytarget].index)
        onlyTarget.reset_index(inplace=True)

        oneCountry = onlyTarget.Capacity.unique().tolist()

        if not oneCountry:
            newtarget= list(self.nameCode.keys())[list(self.nameCode.values()).index(f'{countrytarget}')]
            onlyTarget = self.closingCoal.drop(self.closingCoal.loc[self.closingCoal['Country'] != newtarget].index)
            onlyTarget.reset_index(inplace=True)
            oneCountry = onlyTarget.Capacity.unique().tolist()

        if not oneCountry:
            print('No Power Plants found, but it could be a spelling error.')

        return oneCountry

    def ageAtRetirement(self):
        fig, ax = plt.subplots()
        self.dropNoRetirements()
        self.closingCoal.RetirementYear.replace(to_replace=8888, value=2030, inplace=True)
        ageRetirementPlanned = self.closingCoal.AnnounceOrPlanned - self.closingCoal.Commissioningyear


        self.closingCoal = pd.read_csv('Coal.csv', quotechar="'")
        self.dropNoRetirementsAndPlanned()
        self.closingCoal.RetirementYear.replace(to_replace=8888, value=2030, inplace=True)
        ageRetirementAnnounced = self.closingCoal.RetirementYear - self.closingCoal.Commissioningyear
        ax.hist([ageRetirementPlanned, ageRetirementAnnounced],
                bins=int(max(ageRetirementPlanned)),
                label=['Planned','Announced'],
                 stacked=True, density=True)
        ax.legend()

        ax.set_ylabel('Frequency')
        ax.set_xlabel('Age at Announced or Planned Retirement (years)')
        ax.set_title('Age at Announced or Planned Retirement by Units Retiring')
        plt.show()

    def barCountryPlot(self):
        fig, ax = plt.subplots()

        retiring = {}
        capacityNow = pd.read_csv('Generation.csv',index_col=0)
        capacityNow['Sum']=capacityNow.sum(axis=1)
        capacityNow.sort_values('Sum', inplace=True,ascending=False)
        capacityNow=capacityNow.head(18)
        self.dropNoRetirementsAndPlanned()
        for country in self.closingCoal.Country.unique().tolist():
            retiring[f'{country}'] = sum(self.getOneCountryList(countrytarget=country))
        retiringAmount = {k: v for k, v in sorted(retiring.items(), key=lambda item: item[1], reverse=True)}

        ax.bar(capacityNow.index, capacityNow.sum(axis=1))
        ax.bar(retiringAmount.keys(), retiringAmount.values())

        plt.show()

    def getGenerationCapacity(self, biddingCode):

        client = EntsoePandasClient(api_key=self.myapi)
        start = self.startDateTime #+ datetime.timedelta(hours=100)
        Timespan = pd.date_range(start=self.startDateTime,
                                 periods=self.length +1,
                                 freq="H", tz='Europe/Stockholm')
        end = Timespan.max()
        dummy = {'BiddingCode':''}
        try:
            generation = client.query_installed_generation_capacity(biddingCode, start=start, end=end, psr_type=None)
            generation.reset_index(inplace=True)

            return generation
        except NoMatchingDataError:
            print(biddingCode)
            return pd.DataFrame([dummy])


if __name__ == '__main__':
    x = CoalPowerPlant()
    print(x.getOneCountryList('Poland'))
