# Assignment 4
# Description
# In this assignment you must read in a file of metropolitan regions and associated sports teams from assets/wikipedia_data.html and answer some questions about each metropolitan region. Each of these regions may have one or more teams from the "Big 4": NFL (football, in assets/nfl.csv), MLB (baseball, in assets/mlb.csv), NBA (basketball, in assets/nba.csv or NHL (hockey, in assets/nhl.csv). Please keep in mind that all questions are from the perspective of the metropolitan region, and that this file is the "source of authority" for the location of a given sports team. Thus teams which are commonly known by a different area (e.g. "Oakland Raiders") need to be mapped into the metropolitan region given (e.g. San Francisco Bay Area). This will require some human data understanding outside of the data you've been given (e.g. you will have to hand-code some names, and might need to google to find out where teams are)!

# For each sport I would like you to answer the question: what is the win/loss ratio's correlation with the population of the city it is in? Win/Loss ratio refers to the number of wins over the number of wins plus the number of losses. Remember that to calculate the correlation with pearsonr, so you are going to send in two ordered lists of values, the populations from the wikipedia_data.html file and the win/loss ratio for a given sport in the same order. Average the win/loss ratios for those cities which have multiple teams of a single sport. Each sport is worth an equal amount in this assignment (20%*4=80%) of the grade for this assignment. You should only use data from year 2018 for your analysis -- this is important!

# Notes
# Do not include data about the MLS or CFL in any of the work you are doing, we're only interested in the Big 4 in this assignment.
# I highly suggest that you first tackle the four correlation questions in order, as they are all similar and worth the majority of grades for this assignment. This is by design!
# It's fair game to talk with peers about high level strategy as well as the relationship between metropolitan areas and sports teams. However, do not post code solving aspects of the assignment (including such as dictionaries mapping areas to teams, or regexes which will clean up names).
# There may be more teams than the assert statements test, remember to collapse multiple teams in one city into a single value!
# Question 1
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the NHL using 2018 data.

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
cities['NHL']=cities['NHL'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
nhl_df['team']=nhl_df['team'].str.replace('*','',regex=True)
nhl_df=nhl_df[nhl_df['year']==2018]
def nhl_correlation(): 

    NHL_MetroArea=cities.drop(['NFL','MLB','NBA'],axis=1)
    NHL_MetroArea=NHL_MetroArea[NHL_MetroArea['NHL']!='']
    NHL_MetroArea['NHL']=NHL_MetroArea['NHL'].str.split(r'([A-Z][^A-Z]*\S)([A-Z][^A-Z]*)')
    NHL_MetroArea=NHL_MetroArea.explode('NHL')
    NHL_MetroArea=NHL_MetroArea[NHL_MetroArea['NHL']!='']
    NHL_MetroArea=NHL_MetroArea.rename(columns={'NHL':'team'})
    
    
    w=nhl_df['W'].str.findall('[0-9]{2}')
    w=w.str[0]
    w=pd.to_numeric(w)
    l=nhl_df['L'].str.findall('[0-9]{2}')
    l=l.str[0]
    l=pd.to_numeric(l)
    nhl_df['win_loss_by_region']=w/(w+l)
    nhl_team=nhl_df.drop(['GP','W','L','OL','PTS','PTS%','GF','GA','SRS','SOS','RPt%','ROW','year','League'],axis=1)
    nhl_team=nhl_team.dropna(axis=0)
    nhl_teamMean=nhl_team.groupby('team').mean()
    nhl_teamMean=nhl_teamMean.reset_index()
    NHL_MetroArea['join']=1
    nhl_teamMean['join']=1
    DataFrameFull=NHL_MetroArea.merge(nhl_teamMean,on='join').drop('join',axis=1)
    DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
    NHL_df=DataFrameFull[DataFrameFull['match']]
    
    NHL_df=pd.DataFrame(NHL_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())

    NHL_df=NHL_df.reset_index()
    NHL_df['population_by_region']=pd.to_numeric(NHL_df['population_by_region'])
    
    population_by_region = NHL_df['population_by_region'].iloc[:] # pass in metropolitan area population from cities
    win_loss_by_region = NHL_df['win_loss_by_region'].iloc[:] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]
    pearson_coef,p_value=stats.pearsonr(population_by_region, win_loss_by_region)
    
    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"
    

    return pearson_coef
  nhl_correlation()
0.012486162921209907
# Question 2
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the NBA using 2018 data.

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nba_df=pd.read_csv("assets/nba.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
cities['NBA']=cities['NBA'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
nba_df['team']=nba_df['team'].str.findall(r'\w*\s\w*\s\w*|\w*\s\w*')
nba_df=nba_df[nba_df['year']==2018]
W=nba_df['W'].str.findall('[0-9]{1,2}')
W=W.str[0]
W=pd.to_numeric(W)
L=nba_df['L'].str.findall('[0-9]{1,2}')
L=L.str[0]
L=pd.to_numeric(L)
nba_df['win_loss_by_region']=W/(W+L)
nba_df=nba_df.drop(['W','L','W/L%','GB','PS/G','PA/G','SRS','year','League'],axis=1)
nba_df=nba_df.dropna(axis=0)
nba_df['team']=nba_df['team'].str[0]
nba_df['team']=nba_df['team'].str.replace('\xa0','',regex=True)

def nba_correlation():
    
    NBA_MetroArea=cities.drop(['NFL','MLB','NHL'],axis=1)
    NBA_MetroArea=NBA_MetroArea[NBA_MetroArea['NBA']!='']
    NBA_MetroArea['NBA']=NBA_MetroArea['NBA'].str.split(r'([A-Z][^A-Z]*\S)([A-Z][^A-Z]*)')
    NBA_MetroArea=NBA_MetroArea.explode('NBA')
    NBA_MetroArea=NBA_MetroArea[NBA_MetroArea['NBA']!='']
    NBA_MetroArea=NBA_MetroArea.rename(columns={'NBA':'team'})
    
    nbaMean=nba_df.groupby('team').mean()
    nbaMean=nbaMean.reset_index()
    
    NBA_MetroArea['join']=1
    nbaMean['join']=1
    
    DataFrameFull=NBA_MetroArea.merge(nbaMean,on='join').drop('join',axis=1)
    DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
    NBA_df=DataFrameFull[DataFrameFull['match']]
    NBA_df=pd.DataFrame(NBA_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())
    NBA_df=NBA_df.reset_index()
    NBA_df['population_by_region']=pd.to_numeric(NBA_df['population_by_region'])

    NBA_df=NBA_df.reset_index(drop=True)
    
   
    population_by_region = NBA_df['population_by_region'].iloc[:] # pass in metropolitan area population from cities
    win_loss_by_region =  NBA_df['win_loss_by_region'].iloc[:] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]
    pearson_coef,p_value=stats.pearsonr(population_by_region, win_loss_by_region)
    
    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return pearson_coef
 
nba_correlation()
-0.17657160252844617
# Question 3
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the MLB using 2018 data.

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
cities['MLB']=cities['MLB'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
mlb_df=mlb_df[mlb_df['year']==2018]
mlb_df['W']=pd.to_numeric(mlb_df['W'])
mlb_df['L']=pd.to_numeric(mlb_df['L'])
mlb_df['win_loss_by_region']=mlb_df['W']/(mlb_df['W']+mlb_df['L'])
mlb_df=mlb_df.drop(['W','L','W-L%','GB','year','League'],axis=1)
def mlb_correlation(): 
    
    MLB_MetroArea=cities.drop(['NFL','NBA','NHL'],axis=1)
    MLB_MetroArea=MLB_MetroArea[MLB_MetroArea['MLB']!='']
    MLB_MetroArea['MLB']=MLB_MetroArea['MLB'].str.findall(r'[A-Z][a-z]*\s[A-Z][a-z]*|[A-Z][a-z]*')
    MLB_MetroArea=MLB_MetroArea.explode('MLB')
    MLB_MetroArea=MLB_MetroArea.rename(columns={'MLB':'team'})
    
    mlbMean=mlb_df.groupby('team').mean()
    mlbMean=mlbMean.reset_index()
    MLB_MetroArea['join']=1
    mlbMean['join']=1
    DataFrameFull=MLB_MetroArea.merge(mlbMean,on='join').drop('join',axis=1)
    DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
    MLB_df=DataFrameFull[DataFrameFull['match']]
    MLB_df=pd.DataFrame(MLB_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())
    MLB_df=MLB_df.reset_index()
    MLB_df['population_by_region']=pd.to_numeric(MLB_df['population_by_region'])
    MLB_df=MLB_df.sort_values('population_by_region')
    MLB_df=MLB_df.reset_index(drop=True)
    
    
    population_by_region = MLB_df['population_by_region'] # pass in metropolitan area population from cities
    win_loss_by_region = MLB_df['win_loss_by_region'] # pass in win/loss ratio from mlb_df in the same order as cities["Metropolitan area"]
    pearson_coef,p_value=stats.pearsonr(population_by_region, win_loss_by_region)
    
    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return pearson_coef
 
mlb_correlation()
0.1502769830266931
# Question 4
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the NFL using 2018 data.

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
cities['NFL']=cities['NFL'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
nfl_df=nfl_df[nfl_df['year']==2018]
cols=nfl_df.columns.tolist()
cols=cols[-2:-1]+cols[-4:-3]+cols[-14:-13]
nfl_df=nfl_df[cols]
nfl_df['team']=nfl_df['team'].str.findall(r'[A-Z][^A-Z]+\s[A-Z][a-z]*\s[A-Z][a-z]*|[A-Z][^A-Z]+\s[A-Z][a-z]*')
nfl_df['team']=nfl_df['team'].str[0]
nfl_df=nfl_df.dropna()
W=pd.to_numeric(nfl_df['W'])
L=pd.to_numeric(nfl_df['L'])
win_loss_by_region=W/(W+L)
nfl_df.loc[:,'win_loss_by_region']=win_loss_by_region
def nfl_correlation(): 
    
    NFL_MetroArea=cities.drop(['MLB','NBA','NHL'],axis=1)
    NFL_MetroArea=NFL_MetroArea[NFL_MetroArea['NFL']!='']
    NFL_MetroArea=NFL_MetroArea[NFL_MetroArea['NFL']!=' ']
    NFL_MetroArea['NFL']=NFL_MetroArea['NFL'].str.findall(r'\d*[a-z]*|[A-Z][a-z]*')
    NFL_MetroArea=NFL_MetroArea.explode('NFL')
    NFL_MetroArea=NFL_MetroArea[NFL_MetroArea['NFL']!='']
    NFL_MetroArea=NFL_MetroArea.rename(columns={'NFL':'team'})
    
    nflMean=nfl_df.groupby('team').mean()
    nflMean=nflMean.reset_index()
    NFL_MetroArea['join']=1
    nflMean['join']=1
    DataFrameFull=NFL_MetroArea.merge(nflMean,on='join').drop('join',axis=1)
    DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
    NFL_df=DataFrameFull[DataFrameFull['match']]
    NFL_df=pd.DataFrame(NFL_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())
    NFL_df=NFL_df.reset_index()
    NFL_df['population_by_region']=pd.to_numeric(NFL_df['population_by_region'])
    NFL_df=NFL_df.sort_values('population_by_region')
      
    population_by_region = NFL_df['population_by_region'] # pass in metropolitan area population from cities
    win_loss_by_region = NFL_df['win_loss_by_region'] # pass in win/loss ratio from nfl_df in the same order as cities["Metropolitan area"]
    pearson_coef,p_value=stats.pearsonr(population_by_region, win_loss_by_region)
    
    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return pearson_coef
 
nfl_correlation()
0.004922112149349437
# Question 5
# In this question I would like you to explore the hypothesis that given that an area has two sports teams in different sports, those teams will perform the same within their respective sports. How I would like to see this explored is with a series of paired t-tests (so use ttest_rel) between all pairs of sports. Are there any sports where we can reject the null hypothesis? Again, average values where a sport has multiple teams in one region. Remember, you will only be including, for each sport, cities which have teams engaged in that sport, drop others as appropriate. This question is worth 20% of the grade for this assignment.

import pandas as pd
import numpy as np
import scipy.stats as stats
import re
from functools import reduce
from scipy.stats import ttest_rel

mlb_df=pd.read_csv("assets/mlb.csv")
nhl_df=pd.read_csv("assets/nhl.csv")
nba_df=pd.read_csv("assets/nba.csv")
nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
    
cities['MLB']=cities['MLB'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
mlb_df=mlb_df[mlb_df['year']==2018]
MLB_MetroArea=cities.drop(['NFL','NBA','NHL'],axis=1)
MLB_MetroArea=MLB_MetroArea[MLB_MetroArea['MLB']!='']
MLB_MetroArea['MLB']=MLB_MetroArea['MLB'].str.findall(r'[A-Z][a-z]*\s[A-Z][a-z]*|[A-Z][a-z]*')
MLB_MetroArea=MLB_MetroArea.explode('MLB')
MLB_MetroArea=MLB_MetroArea.rename(columns={'MLB':'team'})
mlb_df['W']=pd.to_numeric(mlb_df['W'])
mlb_df['L']=pd.to_numeric(mlb_df['L'])
mlb_df['win_loss_by_region']=mlb_df['W']/(mlb_df['W']+mlb_df['L'])
mlb_df=mlb_df.drop(['W','L','W-L%','GB','year','League'],axis=1)
mlbMean=mlb_df.groupby('team').mean()
mlbMean=mlbMean.reset_index()
MLB_MetroArea['join']=1
mlbMean['join']=1
DataFrameFull=MLB_MetroArea.merge(mlbMean,on='join').drop('join',axis=1)
DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
MLB_df=DataFrameFull[DataFrameFull['match']]
MLB_df=pd.DataFrame(MLB_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())

MLB_df=MLB_df.reset_index()
MLB_df['population_by_region']=pd.to_numeric(MLB_df['population_by_region'])
MLB_df=MLB_df.sort_values('population_by_region')
MLB_df=MLB_df.reset_index(drop=True)    
    
cities['NHL']=cities['NHL'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
nhl_df=nhl_df[nhl_df['year']==2018]
NHL_MetroArea=cities.drop(['NFL','MLB','NBA'],axis=1)
NHL_MetroArea=NHL_MetroArea[NHL_MetroArea['NHL']!='']
NHL_MetroArea['NHL']=NHL_MetroArea['NHL'].str.split(r'([A-Z][^A-Z]*\S)([A-Z][^A-Z]*)')
NHL_MetroArea=NHL_MetroArea.explode('NHL')
NHL_MetroArea=NHL_MetroArea[NHL_MetroArea['NHL']!='']
NHL_MetroArea=NHL_MetroArea.rename(columns={'NHL':'team'})
nhl_df['team']=nhl_df['team'].str.replace('*','',regex=True)
w=nhl_df['W'].str.findall('[0-9]{2}')
w=w.str[0]
w=pd.to_numeric(w)
l=nhl_df['L'].str.findall('[0-9]{2}')
l=l.str[0]
l=pd.to_numeric(l)
nhl_df['win_loss_by_region']=w/(w+l)
nhl_team=nhl_df.drop(['GP','W','L','OL','PTS','PTS%','GF','GA','SRS','SOS','RPt%','ROW','year','League'],axis=1)
nhl_team=nhl_team.dropna(axis=0)
nhl_teamMean=nhl_team.groupby('team').mean()

nhl_teamMean=nhl_teamMean.reset_index()

NHL_MetroArea['join']=1
nhl_teamMean['join']=1
DataFrameFull=NHL_MetroArea.merge(nhl_teamMean,on='join').drop('join',axis=1)
DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
NHL_df=DataFrameFull[DataFrameFull['match']]
NHL_df=pd.DataFrame(NHL_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())

NHL_df=NHL_df.reset_index()
NHL_df['population_by_region']=pd.to_numeric(NHL_df['population_by_region'])
NHL_df=NHL_df.sort_values('population_by_region')
NHL_df=NHL_df.reset_index(drop=True)

cities['NBA']=cities['NBA'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
NBA_MetroArea=cities.drop(['NFL','MLB','NHL'],axis=1)
NBA_MetroArea=NBA_MetroArea[NBA_MetroArea['NBA']!='']
NBA_MetroArea['NBA']=NBA_MetroArea['NBA'].str.split(r'([A-Z][^A-Z]*\S)([A-Z][^A-Z]*)')
NBA_MetroArea=NBA_MetroArea.explode('NBA')
NBA_MetroArea=NBA_MetroArea[NBA_MetroArea['NBA']!='']
NBA_MetroArea=NBA_MetroArea.rename(columns={'NBA':'team'})
nba_df['team']=nba_df['team'].str.findall(r'\w*\s\w*\s\w*|\w*\s\w*')
nba_df=nba_df[nba_df['year']==2018]
W=nba_df['W'].str.findall('[0-9]{1,2}')
W=W.str[0]
W=pd.to_numeric(W)
L=nba_df['L'].str.findall('[0-9]{1,2}')
L=L.str[0]
L=pd.to_numeric(L)
nba_df['win_loss_by_region']=W/(W+L)
nba_df=nba_df.drop(['W','L','W/L%','GB','PS/G','PA/G','SRS','year','League'],axis=1)
nba_df=nba_df.dropna(axis=0)
nba_df['team']=nba_df['team'].str[0]
nba_df['team']=nba_df['team'].str.replace('\xa0','',regex=True)
nbaMean=nba_df.groupby('team').mean()

nbaMean=nbaMean.reset_index()

NBA_MetroArea['join']=1
nbaMean['join']=1
DataFrameFull=NBA_MetroArea.merge(nbaMean,on='join').drop('join',axis=1)
DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
NBA_df=DataFrameFull[DataFrameFull['match']]
NBA_df=pd.DataFrame(NBA_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())
NBA_df=NBA_df.reset_index()
NBA_df['population_by_region']=pd.to_numeric(NBA_df['population_by_region'])

NBA_df=NBA_df.sort_values('population_by_region')
NBA_df=NBA_df.reset_index(drop=True)

cities['NFL']=cities['NFL'].replace({r'\[.*\]':'','—':''},regex=True)
cities=cities.rename(columns={'Population (2016 est.)[8]':'population_by_region'})
nfl_df=nfl_df[nfl_df['year']==2018]
NFL_MetroArea=cities.drop(['MLB','NBA','NHL'],axis=1)
NFL_MetroArea=NFL_MetroArea[NFL_MetroArea['NFL']!='']
NFL_MetroArea=NFL_MetroArea[NFL_MetroArea['NFL']!=' ']
NFL_MetroArea['NFL']=NFL_MetroArea['NFL'].str.findall(r'\d*[a-z]*|[A-Z][a-z]*')
NFL_MetroArea=NFL_MetroArea.explode('NFL')
NFL_MetroArea=NFL_MetroArea[NFL_MetroArea['NFL']!='']
NFL_MetroArea=NFL_MetroArea.rename(columns={'NFL':'team'})
cols=nfl_df.columns.tolist()
cols=cols[-2:-1]+cols[-4:-3]+cols[-14:-13]
nfl_df=nfl_df[cols]
nfl_df['team']=nfl_df['team'].str.findall(r'[A-Z][^A-Z]+\s[A-Z][a-z]*\s[A-Z][a-z]*|[A-Z][^A-Z]+\s[A-Z][a-z]*')
nfl_df['team']=nfl_df['team'].str[0]
nfl_df=nfl_df.dropna()
W=pd.to_numeric(nfl_df['W'])
L=pd.to_numeric(nfl_df['L'])
win_loss_by_region=W/(W+L)
nfl_df.loc[:,'win_loss_by_region']=win_loss_by_region
nflMean=nfl_df.groupby('team').mean()
nflMean=nflMean.reset_index()
NFL_MetroArea['join']=1
nflMean['join']=1
DataFrameFull=NFL_MetroArea.merge(nflMean,on='join').drop('join',axis=1)
DataFrameFull['match']=DataFrameFull.apply(lambda x:x.team_y.find(x.team_x),axis=1).ge(0)
NFL_df=DataFrameFull[DataFrameFull['match']]
NFL_df=pd.DataFrame(NFL_df.groupby(['Metropolitan area','population_by_region'])['win_loss_by_region'].mean())
NFL_df=NFL_df.reset_index()
NFL_df['population_by_region']=pd.to_numeric(NFL_df['population_by_region'])
NFL_df=NFL_df.sort_values('population_by_region')

NFL_df=NFL_df.reset_index(drop=True)
        
MLB_df=MLB_df.rename(columns={'win_loss_by_region':'MLB','population_by_region':'population_by_region MLB'})
NHL_df=NHL_df.rename(columns={'win_loss_by_region':'NHL','population_by_region':'population_by_region NHL'})
NBA_df=NBA_df.rename(columns={'win_loss_by_region':'NBA','population_by_region':'population_by_region NBA'})
NFL_df=NFL_df.rename(columns={'win_loss_by_region':'NFL','population_by_region':'population_by_region NFL'})

def sports_team_performance(): 
    MLB=MLB_df 
    NHL=NHL_df
    NBA=NBA_df
    NFL=NFL_df
    sports=[NFL,NBA,NHL,MLB]
   
    
    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    
    df_merged=reduce(lambda left,right:pd.merge(left,right,on='Metropolitan area',how='outer'),sports)
    df_merged=df_merged.drop(['Metropolitan area','population_by_region NFL','population_by_region NBA','population_by_region NHL','population_by_region MLB'],axis=1)
    dct={x:{y:ttest_rel(df_merged[x],df_merged[y],nan_policy = 'omit').pvalue for y in df_merged} for x in df_merged}
    p_values=pd.DataFrame(dct)
    p_values=p_values.applymap(lambda x:np.nan if x=='--' else x)
    p_values=p_values.replace(0.000000,np.nan)
    
    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values
