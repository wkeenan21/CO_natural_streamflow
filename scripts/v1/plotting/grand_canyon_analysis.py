#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as md
from matplotlib.pyplot import figure
import matplotlib.patches as patches

#Data wrangling (the boring stuff)

#load in data
AllData = pd.read_csv(r"C:\Users\willy\Desktop\LotteryStats.csv", parse_dates=['Date'])

#Delete useless column
AllData.drop('Number', axis=1, inplace=True)

#Add column for day of week
AllData['DOW'] = AllData['Date'].dt.day_name()

#find dates where two trips were launched
Duplicates = AllData.duplicated()

#Add column to denote days with two trips
Trips = [1] * len(Duplicates)
AllData = AllData.assign(Trips = Trips)

#Delete double rows but add 1 to the trip column. Divide by two to represent relative number of trips
for i in range(len(Duplicates)):
    if Duplicates[i] == True:
        AllData.drop(index = i)
        AllData.loc[i-1, 'Trips'] = AllData['Trips'][i-1] + 1
        AllData.loc[i-1, 'Total'] = AllData['Total'][i-1]/2

#Select just the good weather dates because I'm not interested in freezing my ass off on the river.
# Summer = pd.DataFrame(columns=list(AllData.columns.values))
# for i in range(len(AllData['Date'])):
#     if AllData['Date'][i].month > 3 and AllData['Date'][i].month < 11:
#         Summer = Summer.append(AllData.loc[i])
# Summer = Summer.reset_index(drop=True)

Summer = AllData

#Add year as a new column
Year = []
for i in range(len(Summer['Date'])): Year.append(Summer['Date'].loc[i].year)
Summer = Summer.assign(Year = Year)

#Create column for dates without years
DM = []
for i in range(len(Summer['Date'])): DM.append('2020-' + str(Summer['Date'].loc[i].month) + '-' + str(Summer['Date'].loc[i].day))
Summer = Summer.assign(Date2 = DM)
Summer['Date2'] = pd.to_datetime(Summer['Date2'])

#Add column converting Total to Probability
myPoints = 3
prob = []
for i in Summer['Total']:
    print(i)
    try:
        prob.append(myPoints/(i))
    except:
        prob.append(np.nan)
Summer = Summer.assign(Probability = prob)

#Add column converting Probability to Years Until Success
Years_Until_Success = []
for i in Summer['Probability']:
    Years_Until_Success.append(100/i)
Summer = Summer.assign(Years_Until_Success = Years_Until_Success)

#create small trips
SmallTrips = Summer[Summer['Size'] == 'Small']
SmallTrips = SmallTrips.reset_index(drop=True)

#create standard trips list
StandardTrips = Summer[Summer['Size'] == 'Standard']
StandardTrips = StandardTrips.reset_index(drop=True)

#Viewing the Data
#Create functions for plotting data:
def plotLine(df, xaxis, yaxis, filt = None, yaxis2 = None):
    fig, ax = plt.subplots()

    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.plot(df[filter][xaxis], df[filter][yaxis], label = year)
        ax.legend(frameon = True)
    else:
        ax.plot(df[xaxis], df[yaxis])

    if yaxis2 != None:
        ax2 = ax.twinx()
        ax2.plot(xaxis, yaxis2, color='red')
        ax2.set_ylabel(yaxis2)

    ax.xaxis.set_major_locator(md.DayLocator(interval = 15))
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.show()

def plotScatter(df, xaxis, yaxis, filt = None, yaxis2 = None):
    fig, ax = plt.subplots()

    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.scatter(df[filter][xaxis], df[filter][yaxis], label = year)
        ax.legend(frameon = True)
    else:
        ax.scatter(df[xaxis], df[yaxis])

    if yaxis2 != None:
        ax2 = ax.twinx()
        ax2.scatter(xaxis, yaxis2, color='red')
        ax2.set_ylabel(yaxis2)

    ax.xaxis.set_major_locator(md.DayLocator(interval = 15))
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.show()

#This function averages the probability's by date and creates a new data frame we can use to plot the data.
def FindProbability(df):
    Days = df['Date2'].unique()
    Days = list(Days)
    Days.sort()
    DOY = []
    avgChances = []
    for i in range(len(Days)):
        name = Days[i]
        day = df[df['Date2'] == Days[i]]
        var = sum(day['Total'])
        count = len(day.index)
        average = var/count
        DOY.append(name)
        avgChances.append(average)

    proba = np.array(avgChances)
    proba = 100/(proba/5)
    proba = proba.ravel()
    lists = [DOY, proba, avgChances]

    Probability = pd.DataFrame(lists).transpose()
    Probability.columns = ['Date', 'Probability', 'Average Chances']
    return Probability

print('Standard Trips, Total vs Date')
plotLine(StandardTrips, 'Date2', 'Total', 'Year')
print('Small Trips, Total vs Date')
plotLine(SmallTrips, 'Date2', 'Total', 'Year')

#Call the FindProbability function
standardProba = FindProbability(StandardTrips)
smallProba = FindProbability(SmallTrips)

print('Standard Trips')
plotLine(standardProba, 'Date', 'Probability')
print('Small Trips')
plotLine(smallProba, 'Date', 'Probability')

#Create dataframe to show yearly average (for trips from April - October)
YearlyAverageProb = []
for year in Summer['Year'].unique():
    filter = Summer['Year'] == year
    avg = sum(Summer[filter]['Probability'])/len(Summer[filter]['Probability'])
    YearlyAverageProb.append((year, avg))
YearlyAvg = pd.DataFrame(YearlyAverageProb, columns = ('Year', 'Average Probability'))
print(YearlyAvg)

#This function finds the average number of chances per day of the week
def dayOfweek(df):
    Days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    names = []
    averages = []
    for i in range(len(Days)):
        name = Days[i]
        day = df[df['DOW'] == Days[i]]
        var = sum(day['Total'])
        count = len(day.index)
        average = var/count
        names.append(name)
        averages.append(average)
        lists = [names, averages]
    Weekdays = pd.DataFrame(lists).transpose()
    Weekdays.columns = ['Day', 'Average']
    return Weekdays

DOWsmallTrips = dayOfweek(SmallTrips)
DOWstandardTrips = dayOfweek(StandardTrips)


def findDiffs(df):
    mean = sum(df['Average']) / len(df['Average'])
    diff = []
    for i in range(len(df['Average'])):
        diff.append(str(int((df['Average'][i] / mean - 1)*100)) + '%')
    return diff

smallDiffs = findDiffs(DOWsmallTrips)
standardDiffs = findDiffs(DOWstandardTrips)

DOWstandardTrips = DOWstandardTrips.assign(Diff = standardDiffs)
DOWsmallTrips = DOWsmallTrips.assign(Diff = smallDiffs)

def addlabels(x,y,s):
    for i in range(len(x)):
        plt.text(i,y[i],s[i])

def plotBar(df, x, y, s):
    mean = sum(df[y]) / len(df[y])
    fig = plt.gcf()
    plt.bar(df[x], df[y])
    addlabels(df[x], df[y], df[s])
    plt.axhline(mean, color = 'r')
    plt.text(0,mean + mean*0.04, 'Mean', color = 'r')
    fig.set_size_inches(12, 7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

plotBar(DOWstandardTrips, 'Day', 'Average', 'Diff')
plotBar(DOWsmallTrips, 'Day', 'Average', 'Diff')