import urllib.request as urllib; from lxml import etree
import numpy as np; import pandas as pd
import time; import pickle; import datetime

""" Author : Brandon Veber
    Date   : 1/3/2014
    email  : veber001@umn.edu

This python file contains all the modules required to obtain statistics 
for all players in a given season, and find the career statistics for all 
players who were active in the given season.

All data collected from www.basketball-reference.com

These programs have been tested and verified on Ubuntu 14.04
using Python 3.4

Library dependencies:
    urllib, time, pickle, datetime
    lxml version >= 3.3.3
    numpy version >= 1.9.0
    pandas version >= 0.13.1
    
Bug List:  1 - It is possible to skip a player if his webpage won't load
            
            """

def main(startSeason=2000,endSeason=2015,verbose=0):
    """
    """
    seasonStats = {}
    careerData,lookUp = {},None

    for year in np.arange(endSeason,startSeason-1,-1):
        seasonStats[year]=seasonStatsOnline(year)
        for index,row in seasonStats[year].iterrows():
            careerData,lookUp = updateCareerStats(row['URL'],row['Player'],careerData,lookUp,verbose)    

    pickle.dump(seasonStats,open('seasonStats.p','wb'))
    pickle.dump(careerData,open('careerData.p','wb'))
    pickle.dump(lookUp,open('careerKeyLookup.p','wb'))
    return(seasonStats,careerData,lookUp)

def seasonStatsOnline(year=datetime.datetime.now().year,statType='totals'):
    """
    This module opens the basketball-reference site to get
    the seasons stats and player URLs.  

    Inputs:
        year - int, optional (default = current year)
        
            Specifies the desired year for the basketball-reference url.
            
        statType - string, optional (Default = 'totals')
        
            The type of season statistics to retrieve.  The default
            is the season total ('totals') for basic stats.  
            Alternatives are per game ('per_game'), per 36 minutes ('per_minute'),
            per 100 possessions ('per_poss') and advanced stats like
            PER, TS%, etc. ('advanced').
    
    Output:
        seasonStats - Pandas DataFrame

            The data for all players based on the desired statistic type.
            The data is saved as a dataframe so it can be easily searched
            and sorted.
            
    """
    print('Gathering season statistics for ',year)
    url = 'http://www.basketball-reference.com/leagues/NBA_'+str(year)+'_'+statType+'.html'
    try: table = extractHTML(url,'season',statType)
    except: print("Didn't read properly! Waiting 15s, then trying again"); time.sleep(15)
    headerNode = table[0].xpath('th')
    categories = ['URL']+[node.text for node in headerNode][1:]
    categories[11] = '3PM'; categories[13] = '3P%'
    rowsData = []
    for i in range(1,len(table)):
        rowsData.append([node.text if node.text else node.xpath('a/@href')+node.xpath('a/text()') for node in table[i].xpath('td')][1:])

    seasonStats = pd.DataFrame(columns=categories)

    for row in rowsData:
        if row:
            url = row[0][0]
            convertedRow = [url]
            for i in range(len(row)):
                if categories[i+1] in ['Player','Tm']:
                    convertedRow.append(row[i][1])
                elif categories[i+1] in ['Age','Pos']:
                    convertedRow.append(row[i])
                else:
                    if not row[i]: convertedRow.append(np.nan)
                    else: convertedRow.append(float(row[i]))
            if url in list(seasonStats['URL']):
                index = np.where(seasonStats['URL'] == url)[0][0]
                if seasonStats.loc[index]['Tm']=='O':
                    seasonStats.loc[index]['Tm'] = [convertedRow[4]]
                else: seasonStats.ix[index]['Tm'] = seasonStats.ix[index]['Tm']+[convertedRow[4]]
            else: seasonStats.loc[len(seasonStats)] = convertedRow

    print(statType+' stats for all players found')
    return(seasonStats)

def getCareerStats(seasonStats):
    """
    This module finds ands saves all active players career data by season
    
    Inputs:
        seasonStats - Pandas DataFrame, required
        
            The dataframe containing the yearly data for a given season 
            created by the seasonStatsOnline module
            
    Outputs:
        careerStats - dictionary
            
            This dictionary contains the career statistics of all active
            players during the season of interest.  The keys are the player URL.
            
        lookUp - Pandas DataFrame
        
            This dataframe has a reference of the player name assosciated 
            with the URL.  This makes it possible to search the careerStats
            dictionary by player name rather than the slightly cryptic URL.
    """
    careerStats={}
    lookUp=pd.DataFrame(columns=['url','Player'])
    for index in seasonStats.index:
        url = seasonStats.loc[index,'URL']
        print(seasonStats.loc[index]['Player'])
        lookUp.loc[len(lookUp)]=[url,seasonStats.loc[index]['Player']]
        try:
            careerStats[url]=getPlayerCareerStats(url)
        except:
            print("Didn't read properly! Waiting 15s, then trying again")
            time.sleep(15)
            try:
                careerStats[url]=getPlayerCareerStats(url)
            except: 
                print('Did not find data for ',seasonStats.loc[index]['Player'])
                continue
    return(careerStats,lookUp)

def updateCareerStats(url,name,careerData={},lookUp=None,verbose=0):
    """
    This module can create and update saved career data one player at a time
    
    Inputs:
        url - string, required
        
        name - string, required
        
        careerData - dictionary, optional (default = {})
        
        lookUp - Pandas DataFrame, optional (default = None)
        
        verbose - int, optional (default=0)
        
    Outputs:
        careerStats - dictionary
            
            This dictionary contains the career statistics of all active
            players during the season of interest.  The keys are the player URL.
            
        lookUp - Pandas DataFrame
        
            This dataframe has a reference of the player name assosciated 
            with the URL.  This makes it possible to search the careerStats
            dictionary by player name rather than the slightly cryptic URL.
    
    """
    if not careerData:
        if verbose >=1: print('Getting career data for ',name)
        careerData[url] = getPlayerCareerStats(url)
        lookUp = pd.DataFrame(np.reshape([url,name],(1,2)),columns=['url','Player'])
    else:
        if url not in careerData:
            if verbose >=1: print('Getting career data for ',name)
            careerData[url] = getPlayerCareerStats(url)
            lookUp.loc[len(lookUp)] = [url,name]
    return(careerData,lookUp)
        

def getPlayerCareerStats(url,type='totals'):
    """
    This module reads all players career webpages and extracts the 
    desired statistics by season
    
    Inputs:
        url - string, required
            
            This is the unique URL derived from the seasonStats dataframe
            that is generated in seasonStatsOnline module.  It contains
            the last part of a url for the basketball-reference website.
            For example: Stephen Curry's URL is /players/c/curryst01.html

        type - string, optional (default = 'totals')
            The type of statistics to retrieve.  The default
            is the season total ('totals') for basic stats.  
            Alternatives are per game ('per_game'), per 36 minutes ('per_minute'),
            per 100 possessions ('per_poss') and advanced stats like
            PER, TS%, etc. ('advanced').

    Outputs:
        careerStats - Pandas DataFrame
        
            This is the dataframe containing the players career statistics
            with each row corresponding to a particular season
    """

    url = 'http://www.basketball-reference.com/'+url
    table=extractHTML(url,'career',type)
    headerNode = table[0].xpath('th')
    categories = [node.text for node in headerNode]
    rowsData=[]
    for i in range(1,len(table)):
            rowsData.append([node.text if node.text else node.xpath('a/@href')+node.xpath('a/text()') for node in table[i].xpath('td')])    
    careerStats = pd.DataFrame(columns=categories)
    for row in rowsData:
        if row and row[0] != 'Career' and row[2][:12] != 'Did Not Play': 
            convertedRow = []
            for i in range(len(row)):
                if categories[i] in ['Season','Tm','Lg']:
                    convertedRow.append(row[i][1])
                elif categories[i] in ['Age','Pos']:
                    convertedRow.append(row[i])
                else:
                    if not row[i]: convertedRow.append(np.nan)
                    else: convertedRow.append(float(row[i]))
            if convertedRow[0] in list(careerStats['Season']):
                index = np.where(careerStats['Season'] == convertedRow[0])[0][0]
                if careerStats.loc[index]['Tm']=='O':
                    careerStats.loc[index,('Tm')] = [convertedRow[2]]
                else: careerStats.loc[index]['Tm'] = careerStats.loc[index]['Tm']+[convertedRow[2]]
            else: careerStats.loc[len(careerStats)] = convertedRow
        elif row[2][:12] == 'Did Not Play': continue
        else: break
    return(careerStats)

def extractHTML(url,type='season',subType='totals'):
    """ 
    This module returns specific HTML tables in a raw lxml format

    Inputs:
        url - string, required
        
            The full pro-basketball-reference URL
            
        type - string, optional (default = 'season')
        
            This is the main type of data that is being collected. 
            The options are a season's worth of data for all players
            active during the given season ('season'), gamelog data that
            extracts information from every game ('gamelogs') and data
            for an entire player's career ('career')
            
        subType - string, optional (default = 'totals')
        
            This is the specific type of data that is being collected.
            if type == 'season': 
                the options are 'totals','per_game','per_minutes','per_poss' and
                'advanced'
            if type == 'gamelogs':
                the options are 'player' and 'team'. This returns a list of
                two tables, the first table is basic stats, and the second is
                advanced stats
            if type == 'career':
                the options are 'totals','per_game','per_minutes','per_poss',
                'advanced' and 'shooting'

    Outputs:
        table - list of etree instances, or etree instance
    """
    page = urllib.urlopen(url)
    s = page.read()
    html = etree.HTML(s)
    if type == 'season': 
        return(html.xpath('//table[@id="'+subType+'"]//tr'))
    elif type == 'gamelogs': 
        if subType == 'player': return([html.xpath('//table[@id="pgl_basic"]//tr'),html.xpath('//table[@id="pgl_advanced"]//tr')])
        elif subType == 'team': return([html.xpath('//table[@id="tgl_basic"]//tr'),html.xpath('//table[@id="tgl_advanced"]//tr')])
    elif type == 'career': return(html.xpath('//table[@id="'+subType+'"]//tr'))
