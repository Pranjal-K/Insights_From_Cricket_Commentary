#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pranjalk

"""
# importing some of the main libraries
import pandas as pd
import re


#Read File        
dataCSV = pd.read_csv("Data.txt", delimiter = '\n', names = ["value"])


# generating the data dataframe (It is the main dataframe)
data1 = pd.DataFrame(columns = ["ball"])
data2 = pd.DataFrame(columns = ["value"])
col = 'value'
data1['ball'] = dataCSV.loc[dataCSV[col].str.match('\d\d[.]\d|\d[.]\d', na=False), col]
data2['value'] = dataCSV.loc[dataCSV[col].shift(1).str.match('\d\d[.]\d|\d[.]\d', na=False), col]
data1.reset_index(inplace=True)
del data1['index']
data2.reset_index(inplace=True)
del data2['index']
data = pd.concat([data1, data2], axis=1, sort=False)

  
# data dataframe with split value columns 
new = data["value"].str.split(",", n = 2, expand = True) 
data["Bowler to Batsman"]= new[0] 
data["Run"]= new[1]  
data["Extra"] = new[2]
data.drop(columns =["value"], inplace = True) 


# data dataframe with two new columns and rearrangement
new = data["Bowler to Batsman"].str.split("to", n = 1, expand = True) 
data.drop(columns =["Bowler to Batsman"], inplace = True)
data["Bowler"] = new[0]
data["Batsman"] = new[1]
data = data[['ball', 'Bowler', 'Batsman', 'Run', 'Extra']]


# tidying the data dataframe
data['ball'] = data['ball'].str.strip()
data['Bowler'] = data['Bowler'].str.strip()
data['Batsman'] = data['Batsman'].str.strip()
data['Run'] = data['Run'].str.strip()
data['Extra'] = data['Extra'].str.strip()


# reversing the data dataframe
data = data.iloc[::-1]


# seperating no ball, although it's not present in this dataset
def valuation_formula_noball(x):
    if x.lower() == "no ball":
        return "noball"
    else:
        return x
data['Run'] = data.apply(lambda row: valuation_formula_noball(row['Run']), axis=1)


# seperating extra information in data dataframe
def removeAfterSpace(string):
    """
    input is a string 
    output is a string with everything after comma removed
    """
    return string.split(' ')[0].strip()
data.Run = data.Run.apply(removeAfterSpace)


# making extras(dataframe to store extra runs scored) dataframe and calculating total Extras Score
columns = ['b', 'lb', 'w', 'nb', 'p']
extras = pd.DataFrame(columns = columns)
s2 = pd.Series([0,0,0,0, 0], index=['b', 'lb', 'w', 'nb', 'p'])
extras = extras.append(s2, ignore_index = True)

totalExtrasScore = 0
for index, row in extras.iteritems():
    totalExtrasScore += row[0]


# converting number words to numbers and calculate extras and finalizing the data dataframe
def valuation_formula(x, extra):
    global extras
    if x.lower() == "no":
        return '0'
    elif re.search("\d", x):
        return str(x)
    elif x.lower() == "four":
        return 'four'
    elif x.lower() == "six":
        return 'six'
    elif x.lower() == "out":
        return "out"
    elif x.lower() == "leg":
        string1 = re.findall('\d run', extra)
        string2 = re.findall('\d', string1[0])
        extras.loc[0, 'lb'] += int(string2[0])
        return "leg"
    elif x.lower() == "bye":
        string1 = re.findall('\d run', extra)
        string2 = re.findall('\d', string1[0])
        extras.loc[0, 'b'] += int(string2[0])
        return "bye"
    elif x.lower() == "noball":
        extras.loc[0, 'nb'] += 1
        return "noball"
    elif x.lower() == "penalty":
        extras.loc[0, 'p'] += 5
        return "penalty"
    else:
        extras.loc[0, 'w'] += 1
        return "wide"
data['Run'] = data.apply(lambda row: valuation_formula(row['Run'], row['Extra']), axis=1)


# making a dataframe for bowlers
columns = ['BowlerName', 'Over', 'Maiden', 'Run', 'Wide', 'NoBall','Wicket', 'Eco']
bowlers = pd.DataFrame(columns = columns)


# populating the bowlers dataframe
tempData = data
bowlersIndex = 0
for index, row1 in tempData.iterrows():
    
    if not tempData.empty:
        s2 = pd.Series([0,0,0,0, 0, 0, 0, 0], index=['BowlerName', 'Over', 'Maiden', 'Run', 'Wide', 'NoBall', 'Wicket', 'Eco'])
        bowlers = bowlers.append(s2, ignore_index = True)
        
        bowlers.iloc[bowlersIndex, 0] = tempData.iloc[0, 1]
        bowlerTemp = tempData[tempData.Bowler == tempData.iloc[0, 1]]
        tempData = tempData[tempData.Bowler != tempData.iloc[0, 1]]
        numberOfNoWideBalls = 0
        maidenCount = 0
        
        temp = 0
        

        for i, row2 in bowlerTemp.iterrows():
            
            if row2['Run'] == "wide":
                bowlers.iloc[bowlersIndex, 4] += 1
                numberOfNoWideBalls -= 1
                bowlers.iloc[bowlersIndex, 3] += 1
            elif row2['Run'] == "noball":
                bowlers.iloc[bowlersIndex, 5] += 1
                bowlers.iloc[bowlersIndex, 3] += 1
            elif row2['Run'] == "leg" or row2['Run'] == "bye":
                string1 = re.findall('\d run', row2['Extra'])
                string2 = re.findall('\d', string1[0])
                bowlers.iloc[bowlersIndex, 3] += int(string2[0])
            elif row2['Run'] == "penalty":
                bowlers.iloc[bowlersIndex, 3] += 5
            elif row2['Run'] == 'out':
                bowlers.iloc[bowlersIndex, 6] += 1
            elif row2['Run'] == '0':
                maidenCount += 1
            elif re.search('four', row2['Run']):
                bowlers.iloc[bowlersIndex, 3] += 4
            elif re.search('six', row2['Run']):
                bowlers.iloc[bowlersIndex, 3] += 6
            elif re.search('\d', row2['Run']):
                bowlers.iloc[bowlersIndex, 3] += int(row2['Run'])
            temp += 1
            if temp == 6:
                if maidenCount == 6:
                    bowlers.iloc[bowlersIndex, 2] += 1
                maidenCount = 0
                temp = 0
            
            numberOfNoWideBalls += 1
            
        
        bowlers.iloc[bowlersIndex, 1] = int(numberOfNoWideBalls/6)
        bowlerTemp = bowlerTemp.iloc[0:0]
        bowlersIndex += 1

# calculate ECO
for index, row in bowlers.iterrows():
    row['Eco'] = row['Run']/row['Over']
    
    
# making a dataframe for batsmen
columns = ['Batsman', 'Run', 'Balls', '4s', '6s', 'SR', 'Wicket']
batsmen = pd.DataFrame(columns = columns)


# populating the batsmen dataframe
tempData = data
batsmenIndex = 0

for index, row1 in tempData.iterrows():
    
    if not tempData.empty:
        s2 = pd.Series([0,0,0,0,0,0,0], index=['Batsman', 'Run', 'Balls', '4s', '6s', 'SR', 'Wicket'])
        batsmen = batsmen.append(s2, ignore_index = True)
        
        batsmen.iloc[batsmenIndex, 0] = tempData.iloc[0, 2]
        batsmenTemp = tempData[tempData.Batsman == tempData.iloc[0, 2]]
        tempData = tempData[tempData.Batsman != tempData.iloc[0, 2]]
        
        
        for i, row2 in batsmenTemp.iterrows():                
            if row2['Run'] == "out":
                string = re.findall('[A-Za-z]+[ ]*[A-Za-z]*[ ]c[ ][A-Za-z]+[ ]*[A-Za-z]*[ ]b[ ][A-Za-z]+[ ]*[A-Za-z]*|[A-Za-z]+[ ]*[A-Za-z]*[ ]b[ ][A-Za-z]+[ ]*[A-Za-z]*', row2['Extra'])
                batsmen.iloc[batsmenIndex, 6] = string[0]
            elif row2['Run'] == 'four':
                batsmen.iloc[batsmenIndex, 3] += 1
                batsmen.iloc[batsmenIndex, 1] += 4
            elif row2['Run'] == 'six':
                batsmen.iloc[batsmenIndex, 4] += 1                        
                batsmen.iloc[batsmenIndex, 1] += 6
            elif re.search('\d', row2['Run']):
                batsmen.iloc[batsmenIndex, 1] += int(row2['Run'])
            if row2['Run'] != 'bye' and row2['Run'] != 'noball' and row2['Run'] != 'penalty' and row2['Run'] != 'wide': 
                batsmen.iloc[batsmenIndex, 2] += 1
                
        batsmenTemp = batsmenTemp.iloc[0:0]
        batsmenIndex += 1


# see who are not out
for index, row in batsmen.iterrows():
    if row['Wicket'] == 0:
        row['Wicket'] = 'not out'
        
        
# calculate SR
for index, row in batsmen.iterrows():
    row['SR'] = row['Run']/row['Balls']*100


# Fall of wickets
fallOfWicketsIndex = 0
teamRun = 0
wicket = 0
columns = ['Run', 'Wicket', 'Batsman', 'Over']
fallOfWickets = pd.DataFrame(columns = columns)
for index, row in data.iterrows():
    if row['Run'] == "out":
        wicket += 1
        s2 = pd.Series([0,0,0,0], index=['Run', 'Wicket', 'Batsman', 'Over'])
        fallOfWickets = fallOfWickets.append(s2, ignore_index = True)
        fallOfWickets.iloc[fallOfWicketsIndex, 0] = teamRun
        fallOfWickets.iloc[fallOfWicketsIndex, 1] = wicket
        fallOfWickets.iloc[fallOfWicketsIndex, 2] = row['Batsman']
        fallOfWickets.iloc[fallOfWicketsIndex, 3] = row['ball']
        fallOfWicketsIndex += 1
    elif row['Run'] == 'four':
        teamRun += 4
    elif row['Run'] == 'six':
        teamRun += 6
    elif re.search("\d", row['Run']):
        teamRun += int(row['Run'])
    elif row['Run'].lower() == "leg":
        string1 = re.findall('\d run', row['Extra'])
        string2 = re.findall('\d', string1[0])
        teamRun += int(string2[0])
    elif row['Run'].lower() == "bye":
        string1 = re.findall('\d run', row['Extra'])
        string2 = re.findall('\d', string1[0])
        teamRun += int(string2[0])
    elif row['Run'].lower() == "noball":
        teamRun += 1
    elif row['Run'].lower() == "penalty":
        teamRun += 5
    else:
        teamRun += 1


# calculating total
columns = ['Run', 'Wicket', 'Over']
total  = pd.DataFrame(columns = columns)
s2 = pd.Series([0,0,0], index=['Run', 'Wicket', 'Over'])
total = total.append(s2, ignore_index = True)

for index, row in batsmen.iterrows():
    total.iloc[0, 0] += row['Run']

for index, row in extras.iteritems():
    total.iloc[0, 0] += row[0]

for index, row in bowlers.iterrows():
    total.iloc[0, 1] += row['Wicket']
    total.iloc[0, 2] += row['Over']
    

# fastest Ball
columns = ['Bowler', 'speed']    
fastestBallDataFrame = pd.DataFrame(columns = columns)
fastestBallDataFrameIndex = 0

for index, row in data.iterrows():
    string = re.findall('\d\d\d.\dkm/h|\d\dkph', row['Extra'])
    s2 = pd.Series([0,0], index=['Bowler', 'speed'])
    fastestBallDataFrame = fastestBallDataFrame.append(s2, ignore_index = True)
    fastestBallDataFrame.iloc[fastestBallDataFrameIndex, 0] = row['Bowler']
    if len(string) != 0:
        fastestBallDataFrame.iloc[fastestBallDataFrameIndex, 1] = string[0]
    else:
        fastestBallDataFrame.iloc[fastestBallDataFrameIndex, 1] = None
    fastestBallDataFrameIndex += 1
    
fastestBallDataFrame = fastestBallDataFrame.dropna()

# cleaning the speed column of fastestBallDataFrame dataframe
fastestBallDataFrame['speed'] = fastestBallDataFrame['speed'].map(lambda x: x.rstrip('km/h|kph'))

for index, row in fastestBallDataFrame.iterrows():
    row['speed'] = float(row['speed'])
    
# calculating the fastest Ball    
fastestBall = fastestBallDataFrame[fastestBallDataFrame['speed']==fastestBallDataFrame['speed'].max()]

fastestBall.reset_index(inplace=True)
del fastestBall['index']


# biggest six using sentiment analysis, might not be the most correct one
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
install('textblob')
    
from textblob import TextBlob

columns = ['Batsman', 'Sentiment', 'Subjectivity']    
sentiment = pd.DataFrame(columns = columns)
sentimentIndex = 0

for index, row in data.iterrows():
    if row['Run'] == 'six':    
        s2 = pd.Series([0,0,0], index=['Batsman', 'Sentiment', 'Subjectivity'])
        sentiment = sentiment.append(s2, ignore_index = True)
        analysis = TextBlob(row['Extra'])
        sentiment.iloc[sentimentIndex, 0] = data.iloc[index, 2]
        sentiment.iloc[sentimentIndex, 1] = analysis.sentiment[0]
        sentiment.iloc[sentimentIndex, 2] = analysis.sentiment[1]
        sentimentIndex += 1
    
biggestSix = sentiment[sentiment['Sentiment']==sentiment['Sentiment'].max()]

biggestSix.reset_index(inplace=True)
del biggestSix['index']

    
# display Scorecard
pd.set_option('display.max_columns', 30)
print(total.iloc[0, 0], '-', total.iloc[0, 1], '(', total.iloc[0, 2], ')')
print('Batsman\n',batsmen, '\n')
print('Extras        ',totalExtrasScore,'(b',extras.iloc[0, 0] ,'lb',extras.iloc[0, 1] ,'w',extras.iloc[0, 2] ,'nb' ,extras.iloc[0, 3] ,'p',extras.iloc[0, 4], ')')
print('Total        ',total.iloc[0, 0],'(',total.iloc[0, 1] ,'wkts, ',total.iloc[0, 2] ,'Overs',')')
print('\nFall of wickets')
for index, row in fallOfWickets.iterrows():
    print(fallOfWickets.iloc[index, 0],'-',fallOfWickets.iloc[index, 1] ,'(',fallOfWickets.iloc[index, 2] ,',',fallOfWickets.iloc[index, 3] ,')', end=', ')
print('\n')
print('Bowler')
print(bowlers)
print('\nFastest Ball\n', fastestBall.iloc[0, 0], fastestBall.iloc[0, 1])
print('\nBiggest Six\n', biggestSix.iloc[0, 0])    
    
    
# plotting average runs scored in each over and average runs scored overall
# Note: This does not include runs scored by noball, penalty or wide
# Reason: To provide a more accurate insight in the performance of the batting team
totalRunsScoredByBatsmen = 0

for index, row in data.iterrows():
    if row['Run'].lower() == "no":
        None
    elif re.search("\d", row['Run']):
        totalRunsScoredByBatsmen += int(row['Run'])        
    elif row['Run'].lower() == "four":
        totalRunsScoredByBatsmen += 4        
    elif row['Run'].lower() == "six":
        totalRunsScoredByBatsmen += 6
    elif row['Run'].lower() == "out":
        None
    elif row['Run'].lower() == "leg":
        string1 = re.findall('\d run', row['Extra'])
        string2 = re.findall('\d', string1[0])
        totalRunsScoredByBatsmen += int(string2[0])
    elif row['Run'].lower() == "bye":
        string1 = re.findall('\d run', row['Extra'])
        string2 = re.findall('\d', string1[0])
        totalRunsScoredByBatsmen += int(string2[0])
    elif row['Run'].lower() == "noball":
        None
    elif row['Run'].lower() == "penalty":
        None
    else:
        None


averageTotalRunsScoredByBatsmen = totalRunsScoredByBatsmen/20

columns = ['Over', 'TotalRun']
averageRunByBatsmenPerOver = pd.DataFrame(columns = columns)
currentOver = 0
s2 = pd.Series([0,0], index=['Over', 'TotalRun'])
averageRunByBatsmenPerOver = averageRunByBatsmenPerOver.append(s2, ignore_index = True)

for index, row in data.iterrows():
    if row['Run'].lower() == "no":
        None
    elif re.search("\d", row['Run']):
        if int(float(row['ball'])) == currentOver:
            None
        else:
            s2 = pd.Series([0,0], index=['Over', 'TotalRun'])
            averageRunByBatsmenPerOver = averageRunByBatsmenPerOver.append(s2, ignore_index = True)
            currentOver += 1
            averageRunByBatsmenPerOver.iloc[currentOver, 0] = currentOver
        averageRunByBatsmenPerOver.iloc[currentOver, 1] += int(row['Run'])
        
    elif row['Run'].lower() == "four":
        if int(float(row['ball'])) == currentOver:
            None
        else:
            s2 = pd.Series([0,0], index=['Over', 'TotalRun'])
            averageRunByBatsmenPerOver = averageRunByBatsmenPerOver.append(s2, ignore_index = True)
            currentOver += 1
            averageRunByBatsmenPerOver.iloc[currentOver, 0] = currentOver
        averageRunByBatsmenPerOver.iloc[currentOver, 1] += 4
    elif row['Run'].lower() == "six":
        if int(float(row['ball'])) == currentOver:
            None
        else:
            s2 = pd.Series([0,0], index=['Over', 'TotalRun'])
            averageRunByBatsmenPerOver = averageRunByBatsmenPerOver.append(s2, ignore_index = True)
            currentOver += 1
            averageRunByBatsmenPerOver.iloc[currentOver, 0] = currentOver
        averageRunByBatsmenPerOver.iloc[currentOver, 1] += 6
    
    elif row['Run'].lower() == "out":
        None
    elif row['Run'].lower() == "leg":
        string1 = re.findall('\d run', row['Extra'])
        string2 = re.findall('\d', string1[0])
        
        if int(float(row['ball'])) == currentOver:
            None
        else:
            s2 = pd.Series([0,0], index=['Over', 'TotalRun'])
            averageRunByBatsmenPerOver = averageRunByBatsmenPerOver.append(s2, ignore_index = True)
            currentOver += 1
            averageRunByBatsmenPerOver.iloc[currentOver, 0] = currentOver
        averageRunByBatsmenPerOver.iloc[currentOver, 1] += int(string2[0])
    
    elif row['Run'].lower() == "bye":
        string1 = re.findall('\d run', row['Extra'])
        string2 = re.findall('\d', string1[0])
        
        if int(float(row['ball'])) == currentOver:
            None
        else:
            s2 = pd.Series([0,0], index=['Over', 'TotalRun'])
            averageRunByBatsmenPerOver = averageRunByBatsmenPerOver.append(s2, ignore_index = True)
            currentOver += 1
            averageRunByBatsmenPerOver.iloc[currentOver, 0] = currentOver
        averageRunByBatsmenPerOver.iloc[currentOver, 1] += int(string2[0])
    
    elif row['Run'].lower() == "noball":
        None
    elif row['Run'].lower() == "penalty":
        None
    else:
        None


# plotting process
import matplotlib.pyplot as plt
import numpy as np


plt.xticks(np.arange(min(averageRunByBatsmenPerOver['Over']), max(averageRunByBatsmenPerOver['Over'])+1, 1.0))
plt.yticks(np.arange(min(averageRunByBatsmenPerOver['TotalRun']), max(averageRunByBatsmenPerOver['TotalRun'])+1, 1.0))
plt.xlabel('Overs')
plt.ylabel('Average Runs')
plt.plot(averageRunByBatsmenPerOver['Over'], averageRunByBatsmenPerOver['TotalRun'])
plt.plot(averageRunByBatsmenPerOver['Over'], [averageTotalRunsScoredByBatsmen]*20)

# average runs scored during power play, middle overs and end overs
# Note: This does not include runs scored by noball, penalty or wide
totalPowerPlay = 0
totalMiddle = 0
totalEnd = 0

for index, row in data.iterrows():
    
    if float(row['ball']) < 6:
        if row['Run'].lower() == "no":
            None
        elif re.search("\d", row['Run']):
            totalPowerPlay += int(row['Run'])        
        elif row['Run'].lower() == "four":
            totalPowerPlay += 4        
        elif row['Run'].lower() == "six":
            totalPowerPlay += 6
        elif row['Run'].lower() == "out":
            None
        elif row['Run'].lower() == "leg":
            string1 = re.findall('\d run', row['Extra'])
            string2 = re.findall('\d', string1[0])
            totalPowerPlay += int(string2[0])
        elif row['Run'].lower() == "bye":
            string1 = re.findall('\d run', row['Extra'])
            string2 = re.findall('\d', string1[0])
            totalPowerPlay += int(string2[0])
        elif row['Run'].lower() == "noball":
            None
        elif row['Run'].lower() == "penalty":
            None
        else:
            None
        
    elif float(row['ball']) >=6 and float(row['ball']) <= 15:
        if row['Run'].lower() == "no":
            None
        elif re.search("\d", row['Run']):
            totalMiddle += int(row['Run'])        
        elif row['Run'].lower() == "four":
            totalMiddle += 4        
        elif row['Run'].lower() == "six":
            totalMiddle += 6
        elif row['Run'].lower() == "out":
            None
        elif row['Run'].lower() == "leg":
            string1 = re.findall('\d run', row['Extra'])
            string2 = re.findall('\d', string1[0])
            totalMiddle += int(string2[0])
        elif row['Run'].lower() == "bye":
            string1 = re.findall('\d run', row['Extra'])
            string2 = re.findall('\d', string1[0])
            totalMiddle += int(string2[0])
        elif row['Run'].lower() == "noball":
            None
        elif row['Run'].lower() == "penalty":
            None
        else:
            None
    else:
        if row['Run'].lower() == "no":
            None
        elif re.search("\d", row['Run']):
            totalEnd += int(row['Run'])        
        elif row['Run'].lower() == "four":
            totalEnd += 4        
        elif row['Run'].lower() == "six":
            totalEnd += 6
        elif row['Run'].lower() == "out":
            None
        elif row['Run'].lower() == "leg":
            string1 = re.findall('\d run', row['Extra'])
            string2 = re.findall('\d', string1[0])
            totalEnd += int(string2[0])
        elif row['Run'].lower() == "bye":
            string1 = re.findall('\d run', row['Extra'])
            string2 = re.findall('\d', string1[0])
            totalEnd += int(string2[0])
        elif row['Run'].lower() == "noball":
            None
        elif row['Run'].lower() == "penalty":
            None
        else:
            None


# plotting the score of power play, middle overs and end overs
plt.figure()
plt.yticks([totalPowerPlay, totalMiddle, totalEnd])
plt.xlabel('Match Parts')
plt.ylabel('Runs Scored')
plt.scatter(['PowerPlay', 'MiddleOvers', 'EndOvers'], [totalPowerPlay, totalMiddle, totalEnd])

