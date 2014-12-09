"""
******
NBAMakeTrees
******
Tested in Python3.4 on Windows8 and Ubuntu 14

Collects data from stats.nba.com and basketball-reference.com

This module finds the top scoring player on each team and uses uses
5 parameters from their shotlog data
 -Shot Clock
 -Dribbles
 -Touch Time
 -Shot Distance
 -Closest Defender Distance

From these parameters, a decision tree is used to determine the players
most frequent shooting situations and their success in each one.
"""

__author__ = """Brandon Veber (veber.brandon@gmail.com)"""
__all__ = ['main', 'mainTest', 'makeTrees', 'getShotlogs',
           'extractShotlogs', 'getCurrentPlayers','getAllPlayers',
           'getTopScorers','getTeamURLs','getAllTeams','readHTML']

import requests, pydot, os
import pandas as pd
from lxml import etree
import urllib.request as urllib
import numpy as np
from sklearn import *
import subprocess

def main():
    """Retrieves all relevant data and saves each teams top scorer's
    tree to a png file"""
    allPlayers = getAllPlayers()
    currentPlayers = getCurrentPlayers(allPlayers)
    topScorers = getTopScorers()
    topScorersDataFrame = {}
    for team in topScorers.keys():
        topScorers[team] = currentPlayers.loc[currentPlayers['DISPLAY_LAST_COMMA_FIRST']==topScorers[team].split(' ')[1]+', '+topScorers[team].split(' ')[0]]
    shotlogs = getShotlogs(topScorers)
    data = extractShotlogs(shotlogs)
    try:
        makeTrees(data)
    except: return(data)
    return([])
def mainTest(currentPlayers,topScorers):
    """Testing version of main module that requires pre-retrieval of data"""
##    allPlayers = getAllPlayers()
##    currentPlayers = getCurrentPlayers(allPlayers)
##    topScorers = getTopScorers()
    for team in topScorers.keys():
        topScorers[team] = currentPlayers.loc[currentPlayers['DISPLAY_LAST_COMMA_FIRST']==topScorers[team].split(' ')[1]+', '+topScorers[team].split(' ')[0]]
    return(topScorers)

def makeTrees(data):
    """Takes preprocessed shotlog data to create and save tree graphic"""
    for team in sorted(list(data.keys())):
        print(data[team]['Player'])
        nShots = len(data[team]['y'])
        print('NSplits',[nShots/10,nShots/7.5,nShots/5])
        grid = grid_search.GridSearchCV(tree.DecisionTreeClassifier(),{'min_samples_leaf':[nShots/10,nShots/7.5,nShots/5]})
        grid.fit(data[team]['X'],data[team]['y'])
        print(grid.best_params_)
        clf = grid.best_estimator_
        clf.fit(data[team]['X'],data[team]['y'])
        fname = list(data[team]['Player'])[0]
        with open('C://Users/Brandon/Documents/GitHub/Analyze_NBA/'+fname+'.dot','w') as f:
            f=tree.export_graphviz(clf,out_file=f,feature_names = ['Shot Clock','Dribbles','Touch Time',
                                                                     'Shot Dist','Closest Defender Distance'])
        program = 'C://Program Files (x86)/Graphviz2.38/bin/dot.exe'
        subprocess.call([program,'-Tpng',fname+'.dot','-o',fname+'.png'])
        

def getShotlogs(topScorers):
    """Makes pandas dataframe from players shotlogs"""
    url0 = 'http://stats.nba.com/stats/playerdashptshotlog?'
    params = {'DateFrom':'',
                'DateTo':'',
                'GameSegment':'',
                'LastNGames':'0',
                'LeagueID':'00',
                'Location':'',
                'Month':'0',
                'OpponentTeamID':'0',
                'Outcome':'',
                'Period':'0',
                'PlayerID':'',
                'Season':'2014-15',
                'SeasonSegment':'Pre All-Star',
                'SeasonType':'Regular Season',
                'TeamID':'0',
                'VsConference':'',
                'VsDivision':''}
    shotlogs = {}
    for team in topScorers.keys():
        shotlogs[team] = {'Player':topScorers[team]['PLAYERCODE']}
        params['PlayerID'] = topScorers[team]['PERSON_ID']
        page = requests.get(url0,params=params)
        page=page.json()
        shotlogs[team]['log'] = pd.DataFrame(page['resultSets'][0]['rowSet'],columns=page['resultSets'][0]['headers'])
    return(shotlogs)

def extractShotlogs(shotlogs):
    """Preprocesses raw dataframe to give usable data for classification"""
    data = {}
    for team in shotlogs.keys():
        tempLog = shotlogs[team]['log']
        data[team] = {'Player':shotlogs[team]['Player'],'X':np.transpose(np.vstack((tempLog['SHOT_CLOCK'],tempLog['DRIBBLES'],
                                                                                    tempLog['TOUCH_TIME'],tempLog['SHOT_DIST'],
                                                                                    tempLog['CLOSE_DEF_DIST']))),
                      'y':tempLog['SHOT_RESULT']}
        data[team]['X'][:,0][np.where(np.isnan(data[team]['X'][:,0]))[0]] = [float(time[-1]) for time in tempLog['GAME_CLOCK'][np.where(np.isnan(data[team]['X'][:,0]))[0]]]
    return(data)

def getCurrentPlayers(allPlayers):
    """Returns all active players from NBA players list"""
    return(allPlayers.loc[allPlayers['TO_YEAR'] == '2014'].sort_index(by=['DISPLAY_LAST_COMMA_FIRST'],ascending=[True]))
    
def getAllPlayers():
    """Returns all players from NBA history"""
    url = 'http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2014-15'
    page = requests.get(url)
    page=page.json()
    allPlayers = pd.DataFrame(page['resultSets'][0]['rowSet'],columns=page['resultSets'][0]['headers'])
    currentPlayers = allPlayers.loc[allPlayers['TO_YEAR'] == '2014'].sort_index(by=['DISPLAY_LAST_COMMA_FIRST'],ascending=[True])
    return(allPlayers)

def getTopScorers():
    """Finds the top scorers in terms of Points Per Game for each team"""
    topScorer={}
    teamURLs = getTeamURLs()
    for teamURL in teamURLs:
        url0 = 'http://www.basketball-reference.com'+teamURL
        href0 = readHTML(url0).xpath('//li[@class="narrow small_text"]//a//@href')[0]
        url = 'http://www.basketball-reference.com'+href0
        if href0.split('.')[0].split('/')[-1]=='2015':
            print(url)
            html = readHTML(url)
            tables = pd.io.html.read_html(url)
            if len(tables) == 10:
                topScorer[href0.split('.')[0].split('/')[-2]] = list(tables[4].sort_index(by=['PTS'],ascending=[False])['Player'])[0]
            elif len(tables) == 11:
                topScorer[href0.split('.')[0].split('/')[-2]] = list(tables[5].sort_index(by=['PTS'],ascending=[False])['Player'])[0]
            print(href0.split('.')[0].split('/')[-2],topScorer[href0.split('.')[0].split('/')[-2]])
    return(topScorer)

def getTeamURLs():
    """Finds the urls for each team on basketball-reference.com"""
    url = 'http://www.basketball-reference.com/teams/'
    html = readHTML(url)
    hrefs = html.xpath('//tbody//tr[@class="full_table"]//td//a//@href')
    return(hrefs)

def getAllTeams():
    """Returns list of all team names"""
    return(['celtics','nets','knicks','sixers','raptors',
             'bulls','cavaliers','pistons','pacers','bucks',
             'hawks','hornets','heat','magic','wizards',
             'mavericks','rockets','grizzlies','pelicans','spurs',
             'nuggets','timberwolves','thunder','blazers','jazz',
             'warriors','clippers','lakers','suns','kings'])
def readHTML(url):
    """opens urls and parses it as HTML"""
    return(etree.HTML(urllib.urlopen(url).read()))

if __name__ == '__main__':
    main()
