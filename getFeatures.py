import time; import pickle; import datetime
import numpy as np; import pandas as pd; import pylab as p
from sklearn import *
import getData

""" Author : Brandon Veber
    Date   : 1/6/2014
    email  : veber001@umn.edu

This python file contains all the modules required to extract features
that can be used for 3P% prediction

This program makes use of the getData.py file

These programs have been tested and verified on Ubuntu 14.04
using Python 3.4

Library dependencies:
    urllib, time, pickle, datetime
    numpy       version >= 1.9.0
    pandas      version >= 0.13.1
    matplotlib  version >= 1.5.0
    sklearn     version >= 0.16.0
        
Bug List:   No known bugs
"""

def main(careerData=None,seasonStats=None):
    """
    This is the main module.  This can be run without any inputs, in which case
    the careerData and seasonStats variables will be creating using modules
    in the getData.py file.  Once all the features are created they are saved
    as pickle files.
    
    Inputs:
        careerData - dictionary or pickle file, optional (default=None)
        
            This is the dictionary containing all relevant career data.
            If the data has not been collected previously, then it will be
            collected with the default input.  Either a dictionary or a saved
            pickle file are acceptable inputs

        seasonStats - dictionary of pickle file, optional (default=None)
        
            This is the dictionary containing end of season stats for all 
            relevant years. If the data has not been collected previously, 
            then it will be collected with the default input.  Either a 
            dictionary or a saved pickle file are acceptable inputs

    Outputs: 
        nonRookieData - dictionary
        
            This is the dictionary containing all relevant feature data for
            Non-Rookies. The keys are the seasons and the values are 
            the features
            
        rookieData - dictionary

            This is the dictionary containing all relevant feature data for
            Rookies. The keys are the seasons and the values are 
            the features

        careerData - dictionary
        
            If a new player is found during feature extraction that was not
            previously in the saved careerData variable, it will be updated.
    """
    if careerData:
        try: careerData = pickle.load(careerData)
        except: pass
    if seasonStats:
        try: seasonStats = pickle.load(seasonStats)
        except: pass  
    if not (careerData and seasonStats):
        careerData,seasonStats=getData.main()
    nonRookieData,rookieData,careerData=getTrainData(careerData,seasonStats)
    pickle.dump(nonRookieData,open('nonRookieData.p','wb'))
    pickle.dump(rookieData,open('rookieData.p','wb'))
    pickle.dump(careerData,open('careerData.p','wb'))
    return(nonRookieData,rookieData,careerData)

def getTrainData(careerData=None,seasonStats=None):
    """
    This module returns a dictionary of all machine learning formatted
    data for each year.

    Inputs:
        careerData - dictionary, optional (default = None)
            
            The saved career data

    Outputs:
        nonRookieData - dictionary
        
            All machine learning formatted data saved by year. The keys
            are the year.  The values is the data dictionary from the
            getMLDataYear module
            
        careerData - dictionary
        
            The updated career data logs
    """
    nonRookieData = {}
    rookieData = {}
    career3PA={}
    if not seasonStats: seasonStats = {2015:getData.seasonStatsOnline(2015)}
    years = list(range(2000,2015)); years.reverse()
    if not careerData:
        careerData = getData.getCareerStats(seasonStats[2015])
    for year in years:  
        nonRookieData[year],rookieData[year],careerData,nc = getMLDataYear(year,careerData,seasonStats[year])
    return(nonRookieData,rookieData,careerData)

def getMLDataYear(year,careerData,seasonStatsYear):
    """
    This module returns data in a machine learning format
    for a given year, with 7 input features. 
    Career 3P%, Last Year 3P%, 
    Career 3PA, Last Year 3PA,
    Career FT%, Last Year FT% 
    and Years of NBA Experience.

    It also determines whether a player is in their rookie season, and
    finds their college data and creates 4 input features:
    Career 3P%, Career FT%, Career 3PA, Career Games Played
    
    Inputs:
        years - int, required
            
            The year to gather data from

        careerData - dictionary, required
            
            The saved career data
            
        seasonStats - dictionary, required
        
            The saved season-total statistics
            
    Outputs:
        data - dictionary 
            
            A dictionary with keys of X and y. 
            -X is a numpy array of shape n x 7 (n=number of players) 
            containing all descriptor data
            -y is a numpy array of shape n x 1
            containing the target data (actual 3P%)

        rookieData - dictionary

            A dictionary with keys of X and y. 
            -X is a numpy array of shape n x 4 (n=number of players) 
            containing all descriptor data
            -y is a numpy array of shape n x 1
            containing the target data (actual 3P%)

    """
    last2digits=str(year)[-2:]
    season = str((datetime.datetime(year,1,1)-datetime.timedelta(days=365)).year)+'-'+last2digits
    print(season)
    yearFeatures = []
    rookieFeatures = []
    year3P = []
    rookieYear3P = []
    notCalc=[]
    for index,row in seasonStatsYear.iterrows():
        if row['URL'] not in careerData:
            print(row['Player'],' not in saved career data')
            careerData[row['URL']] = getData.getPlayerCareerStats(row['URL'])
            seasonIndex = len(careerData[row['URL']])-1
        else: 
            try: seasonIndex = careerData[row['URL']][careerData[row['URL']]['Season']==season].index[0]
            except: careerData[row['URL']]=getData.getPlayerCareerStats(row['URL'])
        if seasonIndex <= 1:
            playerFeatures = getRookieFeatures(row['URL'])
            if not np.isnan(careerData[row['URL']].ix[seasonIndex]['3P%']) and not isinstance(playerFeatures,int):
                rookieFeatures.append(playerFeatures)
                rookieYear3P.append(careerData[row['URL']].ix[seasonIndex]['3P%'])
            else: notCalc.append(row['URL']); #print('Rookie ',row['Player'],' not caclulated')
        else:
            if np.sum(careerData[row['URL']].ix[:seasonIndex-1]['3PA'])/(seasonIndex-1) > 10 and not np.isnan(careerData[row['URL']].ix[seasonIndex]['3P%']):
                rowData = getNonRookieFeatures(careerData,row,seasonIndex)
                yearFeatures.append(rowData)
                year3P.append(careerData[row['URL']].ix[seasonIndex]['3P%'])
            else: 
                notCalc.append(row['URL'])
    imputer = preprocessing.Imputer(axis=1)
    yearFeatures=imputer.fit_transform(yearFeatures)
    data = {'X':np.array(yearFeatures),'y':year3P}
    rookieData = {'X':np.array(rookieFeatures),'y':rookieYear3P}
    return(data,rookieData,careerData,notCalc)

def getRookieFeatures(url):
    """
    This module collects college data for a player.  If possible it returns
    the following 4 features:
    Career College 3P%, Career College 3PA
    Career College FT%, Career College Games Played
    
    Inputs:
        url - string, required
        
            This is the specific url tag for any player of interest
            
    Outputs:
        features - list or int
        
            If an int of 1 is returned it means there is no available NCAA data.
            If an int of 2 is returned it means the player is a low-volume 
            3-point shooter.  Otherwise the 4 rookie features are returned.
    """
    url = 'http://www.basketball-reference.com'+url
    features=[] 
    table = getData.extractHTML(url,subType='college')
    if len(table)==0: return(1)
    headerNode = table[1].xpath('th')
    categories = [node.text for node in headerNode]
    rowsData=[]
    for i in range(1,len(table)):
            rowsData.append([node.text if node.text else node.xpath('a/@href')+node.xpath('a/text()') for node in table[i].xpath('td')])    
    careerStats = pd.DataFrame(columns=categories)
    for row in rowsData:
        if row and row[0] == 'Career': 
            #3P%,FT%,3PA,GP
            try: features = [float(row[20]),float(row[21]),float(row[8]),float(row[3])]
            except: return(2)
    return(np.array(features))
    
def getNonRookieFeatures(careerData,row,seasonIndex):
    """
    This module returns features in a machine learning format
    for a given year, with 7 input features. 
    Career 3P%, Last Year 3P%, 
    Career 3PA, Last Year 3PA,
    Career FT%, Last Year FT% 
    and Years of NBA Experience.

    Inputs:
        careerData - dictionary, required
        
        row - row of Pandas Dataframe, required
        
        seasonIndex - int, required
        
    Output:
        rowData - list
            
            The desired features      
    """            
    career3Pperc = np.mean(np.ma.masked_array(careerData[row['URL']].ix[:seasonIndex-1]['3P%'],
                            np.isnan(careerData[row['URL']].ix[:seasonIndex-1]['3P%'])))
    lastYear3Pperc = careerData[row['URL']].ix[seasonIndex-1]['3P%']
    yearI = seasonIndex
    while np.isnan(lastYear3Pperc):
        yearI -= 1
        if yearI == 0:
            lastYear3Pperc = career3Pperc
        else: lastYear3Pperc = careerData[row['URL']].ix[yearI-1]['3P%']
    careerThreePA = np.mean(careerData[row['URL']].ix[:seasonIndex-1]['3PA'])
    lastYear3PA = careerData[row['URL']].ix[seasonIndex-1]['3PA']
    mask = np.ma.masked_array(careerData[row['URL']].ix[:seasonIndex-1]['FT%'],
                            np.isnan(careerData[row['URL']].ix[:seasonIndex-1]['FT%']))
    careerFTperc = np.mean(np.ma.masked_array(careerData[row['URL']].ix[:seasonIndex-1]['FT%'],
                            np.isnan(careerData[row['URL']].ix[:seasonIndex-1]['FT%'])))
    lastYearFTperc = careerData[row['URL']].ix[seasonIndex-1]['FT%']
    yearI = seasonIndex
    while np.isnan(lastYearFTperc):
        yearI -= 1
        if yearI == 0:
            lastYearFTperc = careerFTperc
        else: lastYearFTperc = careerData[row['URL']].ix[yearI-1]['FT%']
    exp = seasonIndex
    rowData = [career3Pperc,lastYear3Pperc,careerThreePA,
            lastYear3PA,careerFTperc,lastYearFTperc,exp]
    return(np.array(rowData))
