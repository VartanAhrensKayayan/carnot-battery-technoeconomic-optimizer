import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from financial import sizingOptimizer
class PlotBuilder():
    def __int__(self):
        self.results=pd.DataFrame()
        self.resultsLocation = 'DE_LU_opex1_powerout100.csv'
    def getResults(self):
        self.resultsLocation = 'NewResults/DE_LU_opex1_powerout100.csv'
        self.results = pd.read_csv(f'{self.resultsLocation}',header=None, index_col=False)
        columnnames=sizingOptimizer()
        columnnames=columnnames.output.keys()
        print(columnnames)

        print(self.results)

        self.results.columns= columnnames#(['Storage', 'Power', 'PowerOut', 'CAPEX', 'Profit', 'Costs', 'Yearly Cashflow', 'LCOE', 'IRR', 'NPV', 'Capacity Factor', 'Utilization Factor', 'CO2 Avoided','Dunno'])
        self.results.drop(self.results.loc[self.results['LCOE']<=-1000].index, inplace=True)
        self.results.drop(self.results.loc[self.results['LCOE'] == 0].index, inplace=True)
        self.results.drop(self.results.loc[self.results['CAPEX'] <= -20_000_000].index, inplace=True)

        self.results.drop(self.results.loc[self.results['NPV']== -999999999999999].index, inplace=True)
        #self.results.drop(self.results.loc[self.results['NPV'] <= 0].index, inplace=True)
        #self.results.drop(self.results.loc[self.results['IRR']<= 0].index, inplace=True)
        #self.results.drop(self.results.loc[self.results['CO2 Avoided']== 0].index, inplace=True)

        self.results["CAPEX"]=self.results["CAPEX"]*-1/1_000_000
        self.results["Yearly Cashflow"]=self.results["Yearly Cashflow"]*-1/1_000_000
        self.results["NPV"]=self.results["NPV"]/1_000_000
        self.results["LCOE"]=self.results["LCOE"]*-1
    def dualPlot(self):
        self.getResults()
        fig, axs= plt.subplots(2)
        colorCoding='LCOE'
        firstplot=axs[0].scatter(self.results["CAPEX"],self.results['IRR']*-1,c=self.results[f'{colorCoding}'],cmap='viridis_r')
        axs[0].set_ylabel('IRR (%)')
        axs[0].title.set_text('Sizing Results for Germany in 2021')
        secondplot=axs[1].scatter(self.results["CAPEX"],self.results["NPV"],c=self.results[f'{colorCoding}'],cmap='viridis_r')
        axs[1].set_ylabel('NPV (M€)')
        axs[1].set_xlabel('CAPEX (M€)')

        cbar=plt.colorbar(secondplot,ax=[axs[1],axs[0]])
        cbar.ax.set_ylabel('LCOS (€/MWh)', rotation=270,  labelpad=40)

        BestLCOEindex=self.results.LCOE.idxmin()
        BestCAPEX= self.results.CAPEX.idxmin()
        BestIRRindex = self.results.IRR.idxmax()
        BestNPVindex = self.results.NPV.idxmax()
        Best={'LCOS':BestLCOEindex,'IRR':BestIRRindex,'NPV':BestNPVindex,'CAPEX':BestCAPEX}
        labeling=False
        if labeling:
            for factor in Best:
                print(f'The point with the best {factor} is '
                      f'with a storage capacity {self.results.Storage.at[Best[factor]]} MWh, '
                      f' {self.results.Power.at[Best[factor]]} MW Electric Heater '
                      f'and a {self.results.PowerOut.at[Best[factor]]} MW Power Block '
                      f'leading to an NPV of {self.results.NPV.at[Best[factor]]} M€ '
                      f'a IRR of {self.results.IRR.at[Best[factor]]} %'
                      f' and a LCOS of {self.results.LCOE.at[Best[factor]]} €/MWh.')

            annotatex=0.15
            axs[1].annotate('IRR', (self.results["CAPEX"].at[BestIRRindex],self.results["NPV"].at[BestIRRindex]),
                            xycoords='data',
                            xytext=(annotatex, 0.7), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='-',facecolor='black',linewidth=.1),
                            horizontalalignment='right', verticalalignment='top'
                            )
            axs[1].annotate('CAPEX', (self.results["CAPEX"].at[BestCAPEX], self.results["NPV"].at[BestCAPEX]),
                            xycoords='data',
                            xytext=(annotatex, 0.6), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='-', facecolor='black', linewidth=.1),
                            horizontalalignment='right', verticalalignment='top'
                            )

            axs[1].annotate('LCOS', (self.results["CAPEX"].at[BestLCOEindex],self.results["NPV"].at[BestLCOEindex]),
                            xycoords='data',
                            xytext=(annotatex, 0.8), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='-',facecolor='black',linewidth=.1),
                            horizontalalignment='right', verticalalignment='top'
                            )
            axs[1].annotate('NPV', (self.results["CAPEX"].at[BestNPVindex],self.results["NPV"].at[BestNPVindex]),
                            xycoords='data',
                            xytext=(annotatex, 0.9), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='-',facecolor='black',linewidth=.1),
                            horizontalalignment='right', verticalalignment='top'
                            )
        plt.show()
    def NinewayPlot(self):
       fig,axs= plt.subplots(3,3,sharex=True,sharey=True)
       firstrow = 0
       secondrow = 0
       opex=[5,2.5,1]
       powerout=[25,50,100]
       location=['DE_LU','Germany']
       for each in range(9):

           self.getResults(filename=f'NewResults/{location[0]}_opex{opex[firstrow]}_powerout{powerout[secondrow]}_IRR')



           aplot=axs[firstrow,secondrow].scatter(self.results["CAPEX"], self.results["IRR"],
                                                 c=self.results.LCOE, cmap='viridis')

           cbar = plt.colorbar(aplot, ax=axs[firstrow,secondrow])
           if firstrow==1 and secondrow==2:
               cbar.ax.set_ylabel('LCOS (€/MWh)', rotation=270, labelpad=40)

           if secondrow==0:
               axs[firstrow, secondrow].set_ylabel(f'OPEX {opex[firstrow]} %\n IRR (%)')
           firstrow += 1
           if firstrow == 3:
               secondrow += 1
               axs[firstrow-1, secondrow - 1].set_xlabel(f'CAPEX (M€)\n Power Block {powerout[secondrow-1]} %')
               firstrow = 0

       fig.suptitle(f'Summary for {location[1]}', fontsize=16)
       """BestLCOEindex = self.results.LCOE.idxmin()
       BestIRRindex = self.results.IRR.idxmax()
       BestNPVindex = self.results.NPV.idxmax()
       axs[1,1].annotate('IRR', (self.results["CAPEX"].at[BestIRRindex], self.results["NPV"].at[BestIRRindex]),
                       xycoords='data',
                       xytext=(0.8, 0.7), textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='-', facecolor='black', linewidth=.1),
                       horizontalalignment='right', verticalalignment='top'
                       )

       axs[1,1].annotate('LCOS', (self.results["CAPEX"].at[BestLCOEindex], self.results["NPV"].at[BestLCOEindex]),
                       xycoords='data',
                       xytext=(0.8, 0.8), textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='-', facecolor='black', linewidth=.1),
                       horizontalalignment='right', verticalalignment='top'
                       )
       axs[1,1].annotate('NPV', (self.results["CAPEX"].at[BestNPVindex], self.results["NPV"].at[BestNPVindex]),
                       xycoords='data',
                       xytext=(0.8, 0.9), textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='-', facecolor='black', linewidth=.1),
                       horizontalalignment='right', verticalalignment='top'
                       )"""
       plt.show()
    def technicalSix(self):
        fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
        firstrow = 0
        secondrow = 0
        location = ['RO', 'Romania']#['DE_LU','Germany']
        opex=1
        powerout=50
        self.getResults(filename=f'NewResults/{location[0]}_opex{opex}_powerout{powerout}_IRR_Lantz')
        self.results['Capacity Factor'].where(self.results['Capacity Factor'] <= 100, self.results['Capacity Factor']/8760, inplace=True)
        print(self.results['Capacity Factor'])

        factor=['Power','Storage','PowerOut','Capacity Factor','Utilization Factor','RoundTrip']
        factorPrettyname= ['Resistance heater (MW)','Storage (MWh)','Power Block (MW)','Capacity Factor (%)',
                           'Utilization Factor (%)','Round Trip Efficiency (%)']
        for each in range(6):



            aplot = axs[firstrow, secondrow].scatter(self.results["CAPEX"], self.results["IRR"],
                                                     c=self.results[f'{factor[each]}'], cmap='viridis')

            cbar = plt.colorbar(aplot, ax=axs[firstrow, secondrow])
            cbar.ax.set_ylabel(f'{factorPrettyname[each]}', rotation=270, labelpad=10)

            if secondrow == 0:
                axs[firstrow, secondrow].set_ylabel('IRR (%)')
            firstrow += 1
            if firstrow == 3:
                axs[firstrow - 1, secondrow - 1].set_xlabel('CAPEX (M€)')
                secondrow += 1
                firstrow = 0
            fig.suptitle(f'Technical Parameters for {location[1]}', fontsize=16)
        plt.show()

if __name__ == '__main__':
    x= PlotBuilder()

    x.dualPlot()

