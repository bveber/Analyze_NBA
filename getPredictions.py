import time; import pickle; import datetime; import itertools; import sys
import numpy as np; import pandas as pd; import pylab as p
from sklearn import *
import getData,getFeatures

""" Author : Brandon Veber
    Date   : 1/8/2014
    email  : veber001@umn.edu

This python file contains all the modules required to predict
3P% for all players in the 2014-15 season

This program makes use of the getData.py file and the getFeatures.py

These programs have been tested and verified on Ubuntu 14.04 and Windows 8
using Python 3.4

Library dependencies:
    urllib, time, pickle, datetime
    numpy       version >= 1.9.0
    pandas      version >= 0.13.1
    matplotlib  version >= 1.5.0
    sklearn     version >= 0.16.0
        
Bug List:   1 - European rookies not predicted
            2 - basketball-reference doesn't have college table for some players
                even if they had a college career
            3 - Career% is average of season %, not total_made/total_attempts
            4 - Unable to predict players with newly expanded 3P shooting role
                who previouly had fewer than 10 3PA/season
            """

def predictNextSeason(year=2015,nonRookieData='nonRookieData.p',rookieData='rookieData.p',careerData='careerData.p',seasonStats='seasonStats.p'):
    """This module uses previously selected algorithms:
            -Veterans: Random Forest; n_estimators=500, min_samples_split=125
            -Novices: SVM; C=.15, gamma = .015, epsilon= .05
        It then trains the models and generates predictions in csv format

        Inputs:
            year - string, optional (default=2015)

            nonRookieData,rookieData,careerData,seasonStats - dictionary, optional (default = None)

                If None, then the variable is generated using the getFeatures.py file
            
        Outputs:
            predictionNonRookies,predicitonRookies - Pandas Dataframe
                
                The dataframes containing the predictions for both groups
    """
    t0=time.time()
    last2digits=str(year)[-2:]
    season = str((datetime.datetime(year,1,1)-datetime.timedelta(days=365)).year)+'-'+last2digits
    if not (careerData and seasonStats):
        seasonStats,careerData,lookUp = getData.main()
    if not (nonRookieData and rookieData):
        nonRookieData,rookieData,careerData=getFeatures.main(careerData)
    nonRookieData,rookieData,careerData,seasonStats=tryPickle(nonRookieData,rookieData,careerData,seasonStats)
    print('All past data found! Now fitting models ',time.time()-t0)
    nonRookiesModel,nonRookiesTrain,nonRookiesScaler = getModel(nonRookieData,'nonRookies')
    rookiesModel,rookiesTrain,rookiesScaler = getModel(rookieData,'rookies')
    print('Models fitted! Now getting all current players features ',time.time()-t0)
    nonRookies,rookies = findPlayerFeatures(year,seasonStats[year],careerData,
        nonRookiesTrain,rookiesTrain,nonRookiesScaler,rookiesScaler)
    print('Features found! Now making predictions ',time.time()-t0)
    predictionsNonRookies = getPredictions(nonRookies,nonRookiesModel,'nonRookies')
    print('Non-Rookie Predictions made! Now predicting Rookies ',time.time()-t0)
##    predictionsNonRookies.to_csv(season+'_Veteran_Predictions.csv',index=False)
    predictionsRookies = getPredictions(rookies,rookiesModel,'rookies')
##    predictionsRookies.to_csv(season+'_Novice_Predictions.csv',index=False)
    predictionsNonRookies.append(predictionsRookies).to_csv(season+'_Predictions.csv',index=False)
    print('Total Runtime is ',time.time()-t0,'s')
    return(predictionsNonRookies,predictionsRookies)

def tryPickle(nonRookieData,rookieData,careerData,seasonStats):
    try: nonRookieData = pickle.load(open(nonRookieData,'rb'))
    except: pass  
    try: rookieData = pickle.load(open(rookieData,'rb'))
    except: pass  
    try: careerData = pickle.load(open(careerData,'rb'))
    except: pass
    try: seasonStats = pickle.load(open(seasonStats,'rb'))
    except: pass  
    return(nonRookieData,rookieData,careerData,seasonStats)

def getPredictions(data,model,group):
    """This model generates all the predictions
    
    Inputs:
        data - Pandas Dataframe, required
        
            Features for every player in the desired year
            
        model - sklearn model, required
        
            The supervised learning algorithm that will make predictions
            
        group - string, required
        
            'nonRookies' for veterans
            'rookies' for novices
    Outputs:
        predictions - Pandas DataFrame
    """
    predictions = pd.DataFrame(columns=['Player','3P% Prediction','+/-'])
    if group=='nonRookies':
        numFeatures=7
    elif group=='rookies':
        numFeatures=4
    for index,row in data.iterrows():
        if len(row['Features'])>numFeatures:predictions.loc[len(predictions)]=[row['Player'],row['Features'],np.nan]
        else:
            plusMinus = getPlusMinus(row['3PA'],group)
            print(row['Player'],plusMinus)
            predictions.loc[len(predictions)]=[row['Player'],model.predict(row['Features'])[0]*100,plusMinus]
    return(predictions)

def getPlusMinus(ThreePA,group):
    if group=='nonRookies':
        bins=np.array([10,100,250,500,1750])
        pm=np.array([8,7,6,5,4])
        return(pm[np.where(bins<=ThreePA)[0][-1]])
    elif group=='rookies':
        bins=np.array([1,50,100,200,300])
        pm=np.array([13,11,10,9,8])
        return(pm[np.where(bins<=ThreePA)[0][-1]])        
        
    
def getModel(data,group):
    """
    This module returns the model, unscaled training data and scaler for a desired group.
    
    Inputs:
        data - Pandas Dataframe, required
        
            Features for every player in the desired year

        group - string, required
        
            'nonRookies' for veterans
            'rookies' for novices
    Outputs:
        clf - SKlearn model
        unScaledTrain - numpy array
        scaler - sklearn scaler
    """
    train,unScaledTrain,scaler = getAllTrainData(data)
    if group=='nonRookies':
        clf = ensemble.RandomForestRegressor(min_samples_split=125,random_state=1)
    elif group == 'rookies':
        clf = svm.SVR(C=.15,gamma=.015,epsilon=.05,random_state=1)
    clf.fit(train['X'],train['y'])
    return(clf,unScaledTrain,scaler)

def getAllTrainData(data):
    """
    This module turns all past data into training data
    """
    for year in range(2000,2015):
        X = data[2000]['X']
        y = data[2000]['y']
        for i in range(2001,year):
            X = np.vstack((X,data[i]['X']))
            y= y+data[i]['y']
        train = {'X': X,'y':np.array(y)}
        scaler = preprocessing.StandardScaler()
        scaledTrain = {'X':scaler.fit_transform(train['X']),'y':train['y']}
    return(scaledTrain,train,scaler)

def findPlayerFeatures(year,seasonStatsYear,careerData,nonRookiesTrain,
        rookiesTrain,nonRookiesScaler,rookiesScaler):
    """
    This module finds all player features.  
        -nonRookies have >1 year of NBA experience
        -rookies have <= 1 year of NBA experience
    """
    nonRookies = pd.DataFrame(columns=['URL','Player','Features','3PA'])
    rookies = pd.DataFrame(columns=['URL','Player','Features','3PA'])
    for index,row in seasonStatsYear.iterrows():
        last2digits=str(year)[-2:]
        season = str((datetime.datetime(year,1,1)-datetime.timedelta(days=365)).year)+'-'+last2digits
        seasonIndex = careerData[row['URL']][careerData[row['URL']]['Season']==season].index[0]
        if seasonIndex <= 1:
            rookieFeatures = getFeatures.getRookieFeatures(row['URL'])
            if isinstance(rookieFeatures,int):
                if rookieFeatures==1: rookies.loc[len(rookies)] = [row['URL'],row['Player'],'No data available',np.nan]
                elif rookieFeatures==2: rookies.loc[len(rookies)] = [row['URL'],row['Player'],'Low-Volume 3-Point Shooter',np.nan]
            else:
                scaledRookieFeats = rookiesScaler.transform(rookieFeatures)
                rookies.loc[len(rookies)] = [row['URL'],row['Player'],scaledRookieFeats,rookieFeatures[2]]
        else:
            if np.sum(careerData[row['URL']].ix[:seasonIndex-1]['3PA'])/(seasonIndex-1) > 10:
                feat = getFeatures.getNonRookieFeatures(careerData,row,seasonIndex)
                for i in range(len(feat)):
                    if np.isnan(feat[i]): feat[i] = np.mean(nonRookiesTrain['X'][:,i])
                scaledFeat = nonRookiesScaler.transform(feat)
                nonRookies.loc[len(nonRookies)] = [row['URL'],row['Player'],scaledFeat,np.sum(careerData[row['URL']].ix[:seasonIndex-1]['3PA'])]
##                print(nonRookies.loc[len(nonRookies)])
            else: nonRookies.loc[len(nonRookies)] = [row['URL'],row['Player'],'Low-Volume 3-Point Shooter',np.nan]
    return(nonRookies,rookies)

def testMethods(nonRookieData=None,rookieData=None,careerData=None):
    """
    The test suite for deciding the best model
    """
    if not careerData:
        seasonStats,careerData,lookUp = getData.main()
    if not (nonRookieData and rookieData):
        nonRookieData,rookieData,careerData=getFeatures.main(careerData)
    resultsNonRookies,predsNonRookies = getCrossVal(nonRookieData,careerData)
    resultsRookies,predsRookies = getRookieCrossVal(rookieData,careerData)
    resultsNonRookies = writeResToPandas(resultsNonRookies,'nonRookies')
    resultsRookies = writeResToPandas(resultsRookies,'rookies')
    return(resultsNonRookies,resultsRookies,predsNonRookies,predsRookies)
def writeResToPandas(results,group):
    if group=='nonRookies':
        pairs=[key[0]+'_'+key[1] for key in itertools.product(['lastYear','career','rf','knn','svm'],['MAE','MSE'])]
    elif group=='rookies':
        pairs=[key[0]+'_'+key[1] for key in itertools.product(['career','rf','knn','svm'],['MAE','MSE'])]
    res=pd.DataFrame(columns=['Season']+pairs)
    for year in results:
        temp=[]
        for pair in pairs:
            temp.append(results[year][pair.split('_')[0]][pair.split('_')[1]])
        res.loc[len(res)]=[year]+temp
    return(res)
    
def getRookieCrossVal(rookieData,careerData):
    """This module validates different techniques for 3P% prediction for novices
    using 2010-2014 seasons as test sets.
    """
    res = {}
    preds = {}
    for year in range(2010,2015):
        print(year)
        X = rookieData[2000]['X']
        y = rookieData[2000]['y']
        for i in range(2001,year):
            X = np.vstack((X,rookieData[i]['X']))
            y= y+rookieData[i]['y']
        train = {'X': X,'y':np.array(y)}
        test = {'X':rookieData[year]['X'],'y':np.array(rookieData[year]['y'])}
        scaler = preprocessing.StandardScaler()
        scaledTrain = {'X':scaler.fit_transform(train['X']),'y':train['y']}
        scaledTest = {'X':scaler.transform(test['X']),'y':test['y']}
        knnGrid = grid_search.GridSearchCV(neighbors.KNeighborsRegressor(),
            param_grid={'n_neighbors':[35],'leaf_size':[1]},
            scoring='mean_squared_error')
        svmGrid = grid_search.GridSearchCV(svm.SVR(),
            param_grid={'C':[.25],'gamma':[.015],'epsilon':[.075]},
            scoring='mean_squared_error')
        rfGrid = grid_search.GridSearchCV(ensemble.RandomForestRegressor(),
            param_grid={'n_estimators':[500],'min_samples_split':[80]},
            scoring='mean_squared_error')
        knnGrid.fit(scaledTrain['X'],scaledTrain['y'])
        svmGrid.fit(scaledTrain['X'],scaledTrain['y'])
        rfGrid.fit(scaledTrain['X'],scaledTrain['y'])
        print(knnGrid.best_estimator_)
        print(svmGrid.best_estimator_)
        print(rfGrid.best_estimator_)
        knnPreds = knnGrid.predict(scaledTest['X'])
        svmPreds = svmGrid.predict(scaledTest['X'])
        rfPreds = rfGrid.predict(scaledTest['X'])
        career = test['X'][:,0]                    
        career = np.array(career)
        preds[year] = {'knn':knnPreds,'svm':svmPreds,'rf':rfPreds,
                        'career':career,'actual':scaledTest['y'],'3PA':test['X'][:,2]}
        res[year] = {'knn':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,knnPreds*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,knnPreds*100)},
                     'svm':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,svmPreds*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,svmPreds*100)},
                     'rf':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,rfPreds*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,rfPreds*100)},
                     'career':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,career*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,career*100)},
                    }
        print(writeResToPandas({year:res[year]},'rookies'))
    return(res,preds)

def getCrossVal(allData,careerData):
    """This module validates different techniques for 3P% prediction for veterans
    using 2010-2014 seasons as test sets.
    """
    res = {}
    preds = {}
    for year in range(2010,2015):
        print(year)
        try:
            seasonStats = getData.seasonStatsOnline(year)
        except: 
            print("Webpage didn't read properly, waiting 30 seconds then trying again")
            time.sleep(30)
        X = allData[2000]['X']
        y = allData[2000]['y']
        for i in range(2001,year):
            X = np.vstack((X,allData[i]['X']))
            y= y+allData[i]['y']
        train = {'X': X,'y':np.array(y)}
        test = {'X':allData[year]['X'],'y':np.array(allData[year]['y'])}
        scaler = preprocessing.StandardScaler()
        scaledTrain = {'X':scaler.fit_transform(train['X']),'y':train['y']}
        scaledTest = {'X':scaler.transform(test['X']),'y':test['y']}
        knnGrid = grid_search.GridSearchCV(neighbors.KNeighborsRegressor(),
            param_grid={'n_neighbors':[100],'leaf_size':[1]},
            scoring='mean_squared_error')
        svmGrid = grid_search.GridSearchCV(svm.SVR(),
            param_grid={'C':[.15],'gamma':[.015],'epsilon':[.05]},
            scoring='mean_squared_error')
        rfGrid = grid_search.GridSearchCV(ensemble.RandomForestRegressor(),
            param_grid={'n_estimators':[500],'min_samples_split':[125]},
            scoring='mean_squared_error')
        knnGrid.fit(scaledTrain['X'],scaledTrain['y'])
        svmGrid.fit(scaledTrain['X'],scaledTrain['y'])
        rfGrid.fit(scaledTrain['X'],scaledTrain['y'])
        print(knnGrid.best_estimator_)
        print(svmGrid.best_estimator_)
        print(rfGrid.best_estimator_)
        knnPreds = knnGrid.predict(scaledTest['X'])
        svmPreds = svmGrid.predict(scaledTest['X'])
        rfPreds = rfGrid.predict(scaledTest['X'])
        career = []; lastYear = []; ThreePA = []; leagueAverage=np.array([np.mean(train['X'][:,0])]*len(test['y']))
        
        for index,row in seasonStats.iterrows():
            last2digits=str(year)[-2:]
            season = str((datetime.datetime(year,1,1)-datetime.timedelta(days=365)).year)+'-'+last2digits
            seasonIndex = careerData[row['URL']][careerData[row['URL']]['Season']==season].index[0]
            if seasonIndex <= 1:
                continue#yearFeatures.append([np.nan for i in range(7)])
            else:
                rowData = getFeatures.getNonRookieFeatures(careerData,row,seasonIndex)
            if np.sum(careerData[row['URL']].ix[:seasonIndex-1]['3PA'])/(seasonIndex-1) > 10 and not np.isnan(careerData[row['URL']].ix[seasonIndex]['3P%']):
                    career.append(rowData[0])
                    lastYear.append(rowData[1])
                    
                    ThreePA.append(np.sum(careerData[row['URL']].ix[:seasonIndex-1]['3PA']))
                    
        career = np.array(career); lastYear=np.array(lastYear);ThreePA=np.array(ThreePA)
        preds[year] = {'knn':knnPreds,'svm':svmPreds,'rf':rfPreds,
                        'career':career,'lastYear':lastYear,'leagueAverage':leagueAverage,
                        'actual':scaledTest['y'],'3PA':ThreePA}
        res[year] = {'knn':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,knnPreds*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,knnPreds*100)},
                     'svm':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,svmPreds*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,svmPreds*100)},
                     'rf':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,rfPreds*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,rfPreds*100)},
                     'career':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,career*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,career*100)},
                     'leagueAverage': {'MSE':metrics.mean_squared_error(scaledTest['y']*100,leagueAverage*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,leagueAverage*100)},
                     'lastYear':{'MSE':metrics.mean_squared_error(scaledTest['y']*100,lastYear*100),
                            'MAE':metrics.mean_absolute_error(scaledTest['y']*100,lastYear*100)}
                    }
        print(writeResToPandas({year:res[year]},'nonRookies'))
    return(res,preds)

def plotResiduals(preds,year,model,scoring='residual_error',titleEnd=''):
    """
    This module plots the error for a give year
    """
    predsModel=preds[year][model]*100
    predsCareer=preds[year]['career']*100
    actual=preds[year]['actual']*100
    p.figure()
    last2digits=str(year)[-2:]
    season = str((datetime.datetime(year,1,1)-datetime.timedelta(days=365)).year)+'-'+last2digits
    if scoring=='residual_error':
        testResiduals = (predsModel-actual)
        careerResiduals = (predsCareer-actual)
        xlabel = 'Residual Error'
        titleStart=''
        if titleEnd=='Novices':
            range = np.arange(np.min([np.min(testResiduals),np.min(careerResiduals)]),
                np.max([np.max(testResiduals),np.max(careerResiduals)]),5)
        else:
            range = np.arange(np.min([np.min(testResiduals),np.min(careerResiduals)]),
                            np.max([np.max(testResiduals),np.max(careerResiduals)]),2.5)
    if scoring=='squared_error':
        testResiduals = (predsModel-actual)**2
        careerResiduals = (predsCareer-actual)**2
        xlabel = 'Squared Error'
        titleStart = 'Log '
        if titleEnd=='Novices':
            range = [10,50,100,250,500,1000,1500]#np.logspace(0,np.log10(np.max([np.max(careerResiduals),np.max(testResiduals)])),10)
        else:
            range=[5,25,50,100,500,1000]
    nBins=range
    yModel,binCentersModel = getHist(testResiduals,nBins)
    yCareer,binCentersCareer = getHist(careerResiduals,nBins)
    p.plot(binCentersModel,yModel,'-',color='r',label='Model Errors')
    p.plot(binCentersCareer,yCareer,'-',color='b',label='Career 3P% Errors')
    if scoring=='residual_error':
        p.axvline(x=0,color='k')
        x1,x2,y1,y2 = p.axis()
        p.axis([-50,50,0,y2])
    if scoring=='squared_error':
        x1,x2,y1,y2 = p.axis()
        if titleEnd=='Novices':
            p.axis([0,1250,0,y2])
        else: p.axis([0,400,0,y2])
    p.xlabel(xlabel)
    p.ylabel('Occurrences')
    p.title(titleEnd+' '+titleStart+'Histogram of '+xlabel+' for '+season+' Season')
    p.legend(loc='best')
    p.show(block=False)

def getHist(residuals,nBins):
    """
    This module creates the histogram data
    """
    y,binEdges = np.histogram(residuals,bins=nBins)
    binCenters = 0.5*(binEdges[1:]+binEdges[:-1])
    return(y,binCenters)

if __name__ == "__main__":
    r=predictNextSeason()
