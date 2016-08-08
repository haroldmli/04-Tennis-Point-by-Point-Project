#--------------------------------------------------------------------------------------------
# Project: Bayesian_Elements_in_PBP.py
#
# Description: This script receives real-time tennis data and exports images that show
# real-time probabilities of each player winning the match.
#--------------------------------------------------------------------------------------------

# Import Packages

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import re
import matplotlib
from scipy.stats import beta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import math
import glob
import pickle
import time
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib import gridspec

#--------------------------------------------------------------------------------------------
# Part 0: Mappings
#
# Description: Mappings from Tennis States to Structures that our code can use.
#--------------------------------------------------------------------------------------------

game_states = {
    (0,0): 'Start of Game',
    (15,0): '15-0',
    (0,15): '0-15',
    (15,15): '15-15',
    (30,0): '30-0',
    (0,30): '0-30',
    (30,15): '30-15',
    (15,30): '15-30',
    (30,30): '30-30',
    (40,0): '40-0',
    (0,40): '0-40',
    (40,15): '40-15',
    (15,40): '15-40',
    (40,30): '40-30',
    (30,40): '30-40',
    (40,40): '40-40',
    (45,40): 'AD-40',
    (40,45): '40-AD',
}

coordinate_states = {
    'Start of Game': (0,0),
    'Server 15-0': (1,1),
    'Server 0-15': (1,-1),
    'Server 15-15': (2,0),
    'Server 30-0': (2,2),
    'Server 0-30': (2,-2),
    'Server 30-15': (3,1),
    'Server 15-30': (3,-1),
    'Server 30-30': (4,0),
    'Server 40-0': (3,3),
    'Server 40-15': (4,2),
    'Server 40-30': (5,1),
    'Server 0-40': (3,-3),
    'Server 15-40': (4,-2),
    'Server 30-40': (5,-1),
    'Server 40-40': (6,0),
    'Server AD-40': (7,1),
    'Server 40-AD': (7,-1)
}

#--------------------------------------------------------------------------------------------
# Base Probability Model Data that dictates chance of winning given chance of winning
# point while serving/returning
#--------------------------------------------------------------------------------------------

game_data, set_data, tiebreak_data, match_data = dict(), dict(), dict(), dict()


#--------------------------------------------------------------------------------------------
# Part 1: Data Retrieval
#
# Description: Imports CSV files and creates data tables needed to make real-time
# predictions.
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Get Ranking, Players and Point Data
#--------------------------------------------------------------------------------------------

def get_rankings_players_point_data():
    rankings = pd.read_csv('tennis_atp-master/atp_rankings_10s.csv',header=None)
    rankings.columns = ['Week','Ranking','Player ID','Points']
    rankings.index = rankings['Week']
    players = pd.read_csv('tennis_atp-master/atp_players.csv',header=None)
    players.columns = ['Player ID','First','Last','L/R','DOB','Country']
    all_point_data = pd.read_csv('All Point Data.csv',index_col=0)
    point_data = all_point_data[['Challenger' not in x for x in all_point_data['Tourney']]]
    
    return rankings, players, point_data

#--------------------------------------------------------------------------------------------
# Get Relevant Match Data
#--------------------------------------------------------------------------------------------

def get_match_data():
    match_10 = pd.read_csv('tennis_atp-master/atp_matches_2010.csv')
    match_11 = pd.read_csv('tennis_atp-master/atp_matches_2011.csv')
    match_12 = pd.read_csv('tennis_atp-master/atp_matches_2012.csv')
    match_13 = pd.read_csv('tennis_atp-master/atp_matches_2013.csv')
    match_14 = pd.read_csv('tennis_atp-master/atp_matches_2014.csv')
    match_15 = pd.read_csv('tennis_atp-master/atp_matches_2015.csv')

    all_matches = pd.concat([match_10, match_11, match_12, match_13, match_14, match_15], axis = 0)   
    return all_matches

#--------------------------------------------------------------------------------------------
# To Tie Matches with Point Data, we had to map the match 
# dates to the Monday of that particular week
#--------------------------------------------------------------------------------------------

def getMondayOfDate(d):
    date = datetime.strptime(str(d),'%Y%m%d')
    wd = date.weekday()
    if wd >= 5:
        date = date + timedelta(days = 7 - wd)
    elif wd == 0:
        date = date + timedelta(days = 0 - wd)
    else:
        date = date + timedelta(days = 0 - wd)
    return int(date.strftime('%Y%m%d'))


#--------------------------------------------------------------------------------------------
# Get Relevant Tournament Data, which includes:
# Tournament Year, Name, Date and Surface
#--------------------------------------------------------------------------------------------

def tourney_info(all_matches):
    tourney_results = all_matches.groupby(['tourney_name','tourney_date','surface']).count().index
    unique_tourney = []
    
    for t,d,s in tourney_results.values:
        if t[-3:] == ' CH':
            t = t[:-3]
        elif t[-2:] == ' Q':
            t = t[:-2]
        elif t[-7:] == 'Masters':
            t = t[:-7]
        t = t.replace(" ", "")
        t = t.replace(".","")
        
        # Certain Tournament Began Before the End of the Previous Year - Adjusted accordingly
        if d == 20131229 or d == 20131230:
            unique_tourney.append(('14',t,getMondayOfDate(d),s))
        elif d == 20121230 or d == 20121231:
            unique_tourney.append(('13',t,getMondayOfDate(d),s))
        else:
            unique_tourney.append((str(d)[2:4],t,getMondayOfDate(d),s))
            
    tourney_dict = defaultdict(dict)
    for a, b, c, d in unique_tourney: 
        tourney_dict[a][b] = (c,d)
        
    return tourney_dict


#--------------------------------------------------------------------------------------------
# Initial Pass of Getting City from Tournament Name
# See below for more manual edits
#--------------------------------------------------------------------------------------------

def getCityFromTourney(tourney):
    try:
        return re.split('[- . \d]',re.split('ATP',tourney)[1])[0] 
    except IndexError:
        return ""

#--------------------------------------------------------------------------------------------
# Manual Cleaning of City Data to ensure that match data and 
# point data joined successfully
#--------------------------------------------------------------------------------------------

def changeCityName(year, city):
    if city == 'Qatar':
        return 'Doha'
    elif city == 'QatarLive':
        return 'Doha'
    elif city == 'RioDeJaneiro':
        return 'RiodeJaneiro'
    elif city == 'Monaco':
        return 'MonteCarlo'
    elif city == 's':
        return 's-Hertogenbosch'
    elif city == 'WashingtonD':
        return 'Washington'
    elif city == 'Montreal' or city == 'Toronto':
        return 'Canada'
    elif city == 'Winston' or city == 'WinstonSalem':
        return 'Winston-Salem'
    elif city == 'St':
        return 'StPetersburg'
    elif city == 'StPetersburgOpen':
        return 'StPetersburg'
    elif city == 'StudenaCroatiaOpen':
        return 'Umag'
    elif city == 'WorldTourFinals' or city == 'London':
        return 'TourFinals'
    elif int(year) <= 13 and (city == 'VinadelMar' or city == 'VinaDelMar'):
        return 'Santiago'
    elif int(year) == 14 and city == 'Santiago':
        return 'VinadelMar'
    elif city == 'HoustonClayCourtChampionship':
        return 'Houston'
    elif city == 'QueensClubLondon':
        return 'Queen\'sClub'
    elif city == 'Oeiras':
        return 'Estoril'
    elif city == 'Dusseldorf' and int(year) == 13:
        return 'PowerHorseCup'
    elif 'ChallengeTour' in city or 'ChallenegerTour' in city:
        return ''
    else:
        return city

#--------------------------------------------------------------------------------------------
# Add Tournament Date and Surface to Point Data using Cleaning Code
# From Above
#--------------------------------------------------------------------------------------------

def add_tourney_surface_to_point_data(point_data, tourney_dict):
    city_map = [getCityFromTourney(x) for x in point_data['Tourney'].values]
    tourney_data = [tourney_dict[str(x)][changeCityName(x,y)][0] if changeCityName(x,y) != '' else ''
                    for x,y in zip(point_data['Year'].values, city_map)] 
    surface_data = [tourney_dict[str(x)][changeCityName(x,y)][1] if changeCityName(x,y) != '' else ''
                    for x,y in zip(point_data['Year'].values, city_map)] 
    
    point_data['Tourney Week'] = tourney_data
    point_data['Surface'] = surface_data
    return point_data

#--------------------------------------------------------------------------------------------
# Clean Player Names in Point Data in order to join cleanly with the 
# Player Dataset
#--------------------------------------------------------------------------------------------

def cleanPlayerNames(name):
    if name in ['Albert Ramos Vinolas','Albert Ramos  Vinolas']:
        return 'Albert Ramos'
    elif name == 'Aleksandr Nedovesov':
        return 'Aleksandr Nedovyesov'
    elif name in ['Alex Bogomolov  Jr','Alex Jr. Bogomolov']:
        return 'Alex Bogomolov Jr'
    elif name == 'Aljax Bedene':
        return 'Aljaz Bedene'
    elif name in ['Andrei Kuznetsov', 'Andrey kumantsov']:
        return 'Andrey Kuznetsov'
    elif name == 'Blav Kavcic':
        return 'Blaz Kavcic'
    elif name == 'Cedric Marcel Stebe':
        return 'Cedrik Marcel Stebe'
    elif name == 'Dennis Kudla':
        return 'Denis Kudla'
    elif name in ['Diego Schwartzman','Diego Sebastian Schwartman']:
        return 'Diego Sebastian Schwartzman'
    elif name == 'Eilas Ymer':
        return 'Elias Ymer'
    elif name == 'Ernest Gulbis':
        return 'Ernests Gulbis'
    elif name == 'Federico Del Bonis':
        return 'Federico Delbonis'
    elif name == 'Frances Tiafoe':
        return 'Francis Tiafoe'
    elif name in ['Izak Van der Merwe', 'Izak van der Merwe']:
        return 'Izak Van Der Merwe'
    elif name == 'Jan Herynch':
        return 'Jan Hernych'
    elif name == 'Joao Olavo Souza':
        return 'Joao Souza'
    elif name == 'Juan Martin del Potro':
        return 'Juan Martin Del Potro'
    elif name == 'Kei Nishkori':
        return 'Kei Nishikori'
    elif name in ['Kenny de Schepper', 'Kenny De Scheper']:
        return 'Kenny De Scheper'
    elif name == 'Marco Trugelliti':
        return 'Marco Trungelliti'
    elif name == 'Mathew Ebden':
        return 'Matthew Ebden'
    elif name == 'Michael Russel':
        return 'Michael Russell'
    elif name in ['Mikail Kukushkin', 'Mikhael Kukushkin']:
        return 'Mikhail Kukushkin'
    elif name in ['Ricardas Barankis','Richard Berankis','Ricardas Bernakis']:
        return 'Ricardas Berankis'
    elif name in ['Rogerio Dutra DA Silva', 'Rogerio Dutra Da Silva']:
        return 'Rogerio Dutra Silva'
    elif name == 'Stan Wawrinka':
        return 'Stanislas Wawrinka'
    elif name in ['Teimuraz Gabashvili','Teymuraz Gabashvilli']:
        return 'Teymuraz Gabashvili'
    elif name == 'Thiemo de Bakker':
        return 'Thiemo De Bakker'
    elif name =='Thomasz Bellucci':
        return 'Thomaz Bellucci'
    elif name == 'Victor Estrella Burgos':
        return 'Victor Estrella'
    elif name in ['Victor Troicki','Vikor Troicki']:
        return 'Viktor Troicki'
    elif name == 'Dmitry Tursonov':
        return 'Dmitry Tursunov'
    elif name == 'Julian Benneteau':
        return 'Julien Benneteau'
    elif name == 'Mikhail Youhzny':
        return 'Mikhail Youzhny'
    elif name == 'Nick Krygios':
        return 'Nick Kyrgios'
    elif name == 'Philipp Kohlschrieber':
        return 'Philipp Kohlschreiber'
    elif name == 'Ranier Schuettler':
        return 'Rainer Schuettler'
    elif name in ['Roberta Bautista Agut', 'Roberto Batista Agut']:
        return 'Roberto Bautista Agut'
    elif name == 'Sam Groth':
        return 'Samuel Groth'
    elif name == 'Sergei Stakhovsky':
        return 'Sergiy Stakhovsky'
    elif name == 'Tatsumo Ito':
        return 'Tatsuma Ito'
    elif name == 'Tim Smyzek':
        return 'Tim Smyczek'
    elif name == 'Yen Hsun LU':
        return 'Yen Hsun Lu'
    elif name in ['Jarkko Niemenen', 'Jarko Nieminen']:
        return 'Jarkko Nieminen'
    else:
        return name.replace('-',' ')


#--------------------------------------------------------------------------------------------
# Clean Point Data Player Namesusing Function Above
#--------------------------------------------------------------------------------------------
  
def clean_player_info(point_data):
    point_data_tourney = pd.DataFrame(point_data[point_data['Tourney Week'] != ''])
    point_data_tourney['Player 1'] = [cleanPlayerNames(x) for x in point_data_tourney['Player 1']]
    point_data_tourney['Player 2'] = [cleanPlayerNames(x) for x in point_data_tourney['Player 2']]
    
    return point_data_tourney


#--------------------------------------------------------------------------------------------
# Combine First and Last Names in Player Dataset to Join with Point Data
#--------------------------------------------------------------------------------------------

def combine_first_last(players):
    players['Name'] = [str(x) + ' ' + str(y) for x,y in zip(players['First'],players['Last'])]
    return players


#--------------------------------------------------------------------------------------------
# Merge Point Data with Players and Rankings to get Player ID and Ranking at
# point in time
#--------------------------------------------------------------------------------------------

def merge_point_ranking_players(point_data_tourney, players, rankings):
    all_data_plus_1 = pd.merge(left=point_data_tourney,right=players, how='left', 
                               left_on='Player 1', right_on='Name', suffixes = ('_pt','_1'))
    all_data_plus_2 = pd.merge(left=all_data_plus_1,right=players, how='left', 
                               left_on='Player 2', right_on='Name', suffixes = ('_1','_2'))
    all_data_plus_3 = pd.merge(left=all_data_plus_2, right=rankings,how='left',
                               left_on=['Player ID_1','Tourney Week'],
                               right_on=['Player ID','Week'], suffixes = ('_1','_1R'))
    all_data_plus_4 = pd.merge(left=all_data_plus_3, right=rankings,how='left',
                               left_on=['Player ID_2','Tourney Week'],
                               right_on=['Player ID','Week'], suffixes = ('_1R','_2R'))
    
    return all_data_plus_4


#--------------------------------------------------------------------------------------------
# Classify Ranking Group for Each Player for Probability Predictions
#--------------------------------------------------------------------------------------------

def classifyRank(num):
    if num <= 4:
        return str(num)
    elif num <= 10:
        return "5-10"
    elif num <= 20:
        return "11-20"
    elif num <= 50:
        return '21-50'
    elif num <= 100:
        return '51-100'
    else:
        return 'Outside Top 100'

def rankingGroup(rank1, rank2, server):
    if server == 1:
        x1, x2 = rank1, rank2
    elif server == 2:
        x1, x2 = rank2, rank1
    return classifyRank(x1) + ' vs. ' + classifyRank(x2)


#--------------------------------------------------------------------------------------------
# Adds Ranking Matchup and Removes Missing Data
#--------------------------------------------------------------------------------------------

def final_clean_up(all_data_plus_4):
    all_data_plus_4['Ranking Matchup'] = [rankingGroup(x,y,z) for x,y,z in 
                                          zip(all_data_plus_4['Ranking_1R'],
                                              all_data_plus_4['Ranking_2R'],
                                              all_data_plus_4['Server'])]

    full_data = all_data_plus_4[~(np.isnan(all_data_plus_4['Ranking_2R']) | np.isnan(all_data_plus_4['Ranking_1R']) |
        np.isnan(all_data_plus_4['Player ID_1']) | np.isnan(all_data_plus_4['Player ID_2']))]


    missing_data = all_data_plus_4[(np.isnan(all_data_plus_4['Ranking_2R']) | np.isnan(all_data_plus_4['Ranking_1R']) |
        np.isnan(all_data_plus_4['Player ID_1']) | np.isnan(all_data_plus_4['Player ID_2']))]
    
    print str(len(full_data)) + " Point Observations were Valid."
    print str(len(missing_data)) + " Point Observations were Removed."
    
    return full_data

#--------------------------------------------------------------------------------------------
# Main Function that Cleans All Data
#--------------------------------------------------------------------------------------------

def create_clean_data():
    rankings, players, point_data = get_rankings_players_point_data()
    all_matches = get_match_data()
    tourney_dict = tourney_info(all_matches)
    point_data = add_tourney_surface_to_point_data(point_data, tourney_dict)
    point_data_tourney = clean_player_info(point_data)
    players = combine_first_last(players)
    all_data_plus_4 = merge_point_ranking_players(point_data_tourney, players, rankings)
    full_data = final_clean_up(all_data_plus_4)
    
    return full_data, all_matches, rankings

#--------------------------------------------------------------------------------------------
# Part 2: Grouping Data to Use in Predictive Model
#
# Description: Using the Cleaned "full data", this part of the code groups the full data
# by various dimensions, so that the model can adjust match scenarios by these dimensions
# (e.g. surface, ranking differences, recent form, head-2-head, etc.)
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Maps Tiebreak States (They are different from the traditional game scoring system)
#--------------------------------------------------------------------------------------------

def tiebreak_state(state):
    p1,p2, serving = state

    if p1 >= 6 and p2 >= 6:
        if p1 > p2:
            p1, p2 = 7,6
        elif p1 < p2:
            p1, p2 = 6,7
        else:
            p1, p2 = 6,6
    
    if ((p1+p2) % 4 == 1) or ((p1+p2) % 4 == 2):
        if serving == 'Serving':
            suffix = ' (R)'
        else:
            suffix = ' (S)'
    else:
        if serving == 'Serving':
            suffix = ' (S)'
        else:
            suffix = ' (R)'
        
    if p1 >= 6 and p2 >=6:
        if p1 > p2:
            return '7-6' + suffix
        elif p1 < p2:
            return '6-7' + suffix
        else:
            return '6-6' + suffix
    else:
        return str(p1)+'-'+str(p2) + suffix

    
def tiebreak_coordinates(game_state):
    if game_state not in coordinate_states.keys():
        #print game_state#[7:]
        g1, g2 = game_state[7:].split('-')
        #print g1, g2
        return (int(g1)+int(g2), int(g1)-int(g2))
    else:
        return coordinate_states[game_state]


#--------------------------------------------------------------------------------------------
# This function translates all the different game states into winning percentages.
# The score_state_all data frame splits the point data into the dimensions specified, while
# the score_state_sum data frame splits the data further into win/loss, in order to
# calculate the winning %.
#--------------------------------------------------------------------------------------------

def get_win_pct_data(score_state_all, score_state_sum, matchup_exists = True):
    all_results = []
    for matchup, set_score, game_score, game_state in score_state_all.index.values:   
        try:
            x, y = tiebreak_coordinates(game_state)
            if game_score == '6-6':
                if x % 4 == 1 or x % 4 == 2:
                    all_results.append((matchup, set_score,game_score,'Serving',game_state[7:]+ ' (R)', 
                                1.0*float(score_state_sum.loc[(matchup, set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                x, y))
                else:
                    all_results.append((matchup, set_score,game_score,'Serving',game_state[7:]+ ' (S)', 
                                1.0*float(score_state_sum.loc[(matchup, set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                x, y))
            elif 'Server' in game_state:
                all_results.append((matchup, set_score,game_score,'Serving',game_state[7:], 
                                1.0*float(score_state_sum.loc[(matchup, set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                x, y))
            else:
                all_results.append((matchup, set_score,game_score,'Serving',game_state, 
                                1.0*float(score_state_sum.loc[(matchup, set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                x, y))

            if 'Start of Game' not in game_state:
                score1, score2 = game_state[7:].split('-')
                new_score = score2+'-'+score1   
            else:
                new_score = game_state

            if game_score == 'Up 1 from 6-6 5th Set':
                newgscore = 'Down 1 from 6-6 5th Set'
            elif game_score == 'Down 1 from 6-6 5th Set':
                newgscore = 'Up 1 from 6-6 5th Set'
            elif game_score == 'All Square from 6-6 5th Set':
                newgscore = game_score
            else:
                gscore1, gscore2 = game_score.split('-')
                newgscore = gscore2+'-'+gscore1

            sscore1, sscore2 = set_score.split('-')
            newsscore = sscore2+'-'+sscore1

            if game_score == '6-6':
                if x % 4 == 1 or x % 4 == 2:
                    new_score = new_score + ' (S)'
                else:
                    new_score = new_score + ' (R)'
            
            if matchup_exists:
                m1, m2 = matchup.split(' vs. ')
                newmatchup = m2 + ' vs. ' + m1
            else:
                newmatchup = matchup

            all_results.append((newmatchup, newsscore,newgscore,'Returning',new_score, 
                                1.0 - 1.0*float(score_state_sum.loc[(matchup, set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]),
                                x, -1*y))
        except KeyError:
            #print 'Key Error at ' + str((matchup, set_score, game_score, game_state))
            x, y = tiebreak_coordinates(game_state)
            if game_score == '6-6':
                if x % 4 == 1 or x % 4 == 2:
                    all_results.append((matchup, set_score, game_score, 'Serving',game_state[7:]+ ' (R)', 0.,
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]), x, y))
                else:
                    all_results.append((matchup, set_score, game_score, 'Serving',game_state[7:]+ ' (S)', 0.,
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]), x, y))
            elif 'Server' in game_state:
                all_results.append((matchup, set_score, game_score, 'Serving',game_state[7:], 0.,
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]), x, y))
            else:
                all_results.append((matchup, set_score, game_score, 'Serving',game_state, 0.,
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]), x, y))

            #print game_state
            if 'Start of Game' not in game_state:
                score1, score2 = game_state[7:].split('-')
                new_score = score2+'-'+score1   
            else:
                new_score = game_state

            if game_score == 'Up 1 from 6-6 5th Set':
                newgscore = 'Down 1 from 6-6 5th Set'
            elif game_score == 'Down 1 from 6-6 5th Set':
                newgscore = 'Up 1 from 6-6 5th Set'
            elif game_score == 'All Square from 6-6 5th Set':
                newgscore = game_score
            else:
                gscore1, gscore2 = game_score.split('-')
                newgscore = gscore2+'-'+gscore1
            sscore1, sscore2 = set_score.split('-')
            newsscore = sscore2+'-'+sscore1

            if game_score == '6-6':
                if x % 4 == 1 or x % 4 == 2:
                    new_score = new_score + ' (S)'
                else:
                    new_score = new_score + ' (R)'
            
            if matchup_exists:
                m1, m2 = matchup.split(' vs. ')
                newmatchup = m2 + ' vs. ' + m1
            else:
                newmatchup = matchup

            all_results.append((newmatchup, newsscore, newgscore, 'Returning',new_score, 1.,
                                int(score_state_all.loc[(matchup, set_score,game_score,game_state)]), x, -1*y))
    return all_results


#--------------------------------------------------------------------------------------------
# Sets up the "Real Dataset" which splits point data by ranking group, and game situation
#--------------------------------------------------------------------------------------------

def setup_real_data(full_data):
    score_state_all = pd.DataFrame(full_data.groupby(['Ranking Matchup',
                                      'Set Score Server View',
                                      'Game Score Server View', 
                                      'Game State New']).count()['Player 1'])
    score_state_sum = pd.DataFrame(full_data.groupby(['Ranking Matchup',
                                     'Set Score Server View',
                                     'Game Score Server View', 
                                     'Game State New', 
                                     'Server Winner']).count()['Player 1'])
    all_results = get_win_pct_data(score_state_all, score_state_sum, matchup_exists = True)
    sum_data = pd.DataFrame(all_results,columns=['Matchup','Set Score','Game Score',
                                  'Serving at Start of Game?','Point Score','% Win',
                                  'Number of Instances','Points Elapsed','Y'])
    sum_data.index = sum_data['Set Score'] + sum_data['Game Score'] + sum_data['Serving at Start of Game?'] + \
                     sum_data['Point Score'] + sum_data['Matchup']
    return sum_data


#--------------------------------------------------------------------------------------------
# Sets up the "Prior Dataset" which splits point data by game situation only
#--------------------------------------------------------------------------------------------

def setup_prior_data():
    prior_data = pd.read_csv('Match_Probs_50_50.csv',index_col=0)
    prior_data.index = prior_data['Set Score'] + prior_data['Game Score'] + prior_data['Serving?'] + prior_data['Point Score']
    return prior_data


#--------------------------------------------------------------------------------------------
# Sets up the "Matchup Dataset" which splits point data by ranking group only
#--------------------------------------------------------------------------------------------

def setup_matchup_data(sum_data):
    matchup_data = pd.DataFrame(sum_data[(sum_data['Set Score'] == '0-0') & 
                                         (sum_data['Game Score'] == '0-0') &
                                         (sum_data['Point Score'] == 'Start of Game')])
    matchup_data['Win Instances'] = matchup_data['% Win'] * matchup_data['Number of Instances']
    matchup_sum = matchup_data.groupby('Matchup').sum()[['Win Instances','Number of Instances']]
    return matchup_sum


#--------------------------------------------------------------------------------------------
# Sets up the "Surface Dataset" which splits point data by surface and game situations
#--------------------------------------------------------------------------------------------

def setup_surface_data(full_data):
    surface_state_all = pd.DataFrame(full_data.groupby(['Surface',
                                      'Set Score Server View',
                                      'Game Score Server View', 
                                      'Game State New']).count()['Player 1'])
    surface_state_sum = pd.DataFrame(full_data.groupby(['Surface',
                                     'Set Score Server View',
                                     'Game Score Server View', 
                                     'Game State New', 
                                     'Server Winner']).count()['Player 1'])
    surface_results = get_win_pct_data(surface_state_all, surface_state_sum, matchup_exists = False)
    surface_data = pd.DataFrame(surface_results,columns=['Surface','Set Score','Game Score',
                                      'Serving at Start of Game?','Point Score','% Win',
                                      'Number of Instances','Points Elapsed','Y'])
    surface_data.index = surface_data['Set Score'] + surface_data['Game Score'] + \
                         surface_data['Serving at Start of Game?'] + surface_data['Point Score'] + \
                         surface_data['Surface']
    return surface_data


#--------------------------------------------------------------------------------------------
# Sets up the "H2H Dataset" which splits point data by head-to-head situations among players
#--------------------------------------------------------------------------------------------

def setup_h2h_data(all_matches):
    all_ids = np.unique(np.concatenate((np.unique(all_matches['winner_id']),
                                        np.unique(all_matches['loser_id'])),axis = 0))
    h2h_data = pd.DataFrame(0,index=all_ids, columns = all_ids)

    all_h2h = pd.DataFrame(all_matches.groupby(['winner_id','loser_id']).count()['tourney_id'])
    for key,row in all_h2h.iterrows():
        winner_id, loser_id = key
        #print row['tourney_id']
        h2h_data.ix[winner_id, loser_id] = row['tourney_id']
    return h2h_data


#--------------------------------------------------------------------------------------------
# Calls the functions above, and returns all "grouped" datasets
#--------------------------------------------------------------------------------------------

def setup_all_datasets(full_data, all_matches, datecutoff):
    
    full_data = full_data[(full_data['Week_1R'] < datecutoff) &
                          (full_data['Week_2R'] < datecutoff)]
    all_matches = all_matches[all_matches['tourney_date'] < datecutoff]
    
    prior_data = setup_prior_data()
    real_data = setup_real_data(full_data)
    matchup_data = setup_matchup_data(real_data)
    surface_data = setup_surface_data(full_data)
    h2h_data = setup_h2h_data(all_matches)
    return prior_data, real_data, matchup_data, surface_data, all_matches, h2h_data

#--------------------------------------------------------------------------------------------
# Returns the recent W/L record of a particular player to capture recent form
#--------------------------------------------------------------------------------------------

def recent_activity(player_id, d2, data, months, surface=None):
    enddate = datetime.strptime(str(d2),'%Y%m%d')
    begindate = enddate + relativedelta(months = -1*months)
    d1 = int(begindate.strftime('%Y%m%d'))
    #print d1, d2
    if surface == None:
        wins = len(data[(data['tourney_date'] >= d1) & 
                        (data['tourney_date'] < d2) &
                        (data['winner_id'] == player_id)])
        losses = len(data[(data['tourney_date'] >= d1) & 
                        (data['tourney_date'] < d2) &
                        (data['loser_id'] == player_id)])
    else:
        wins = len(data[(data['tourney_date'] >= d1) & 
                        (data['tourney_date'] < d2) &
                        (data['winner_id'] == player_id) &
                        (data['surface'] == surface)])
        losses = len(data[(data['tourney_date'] >= d1) & 
                        (data['tourney_date'] < d2) &
                        (data['loser_id'] == player_id) &
                        (data['surface'] == surface)])
    return wins, losses

#--------------------------------------------------------------------------------------------
# Returns the matchup grid that has ranking group win probabilities based on intuition
#--------------------------------------------------------------------------------------------

def setup_matchup_grid():
    matchup_groups = ['1','2','3','4','5-10','11-20','21-50','51-100','Outside Top 100']
    matchup_grid = [[0.50, 0.65, 0.70, 0.75, 0.80, 0.85, 0.93, 0.97, 0.98],
                    [0.35, 0.50, 0.60, 0.73, 0.77, 0.83, 0.90, 0.95, 0.97],
                    [0.30, 0.40, 0.50, 0.70, 0.73, 0.80, 0.85, 0.90, 0.96],
                    [0.25, 0.27, 0.30, 0.50, 0.70, 0.75, 0.80, 0.85, 0.95],
                    [0.20, 0.23, 0.27, 0.30, 0.50, 0.70, 0.75, 0.80, 0.90],
                    [0.15, 0.17, 0.20, 0.25, 0.30, 0.50, 0.60, 0.70, 0.80],
                    [0.07, 0.10, 0.15, 0.20, 0.25, 0.40, 0.50, 0.60, 0.70],
                    [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
                    [0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]]
    matchup_grid = pd.DataFrame(matchup_grid, columns = matchup_groups, index = matchup_groups)
    return matchup_grid

#--------------------------------------------------------------------------------------------
# Part 3: The Base Probability Model
#
# Description: This section constructs the base probability model that calculates a player's
# chance of winning a match at a given game situation and a pre-designated chance of winning
# a point while serving and while returning.
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Determines Chance of Winning a Game given chance of winning when serving/returning
#--------------------------------------------------------------------------------------------

def game_win_prob(x, y, p):
    if (x,y,p) in game_data:
        return game_data[(x,y,p)]
    w = p**2/(1.-2*p*(1-p))
    if (x == 4 and y <= 2):
        return 1.
    elif (x <= 2 and y == 4):
        return 0.
    elif x >= 3 and y >= 3:
        if x == y:
            results =  w
        elif x > y:
            results = p + (1.-p)*w
        elif x < y:
            results = p*w
        game_data[(x,y,p)] = w
        game_data[(x+1,y,p)] = p + (1.-p)*w
        game_data[(x,y+1,p)] = p*w
        return results
    else:
        results = game_win_prob(x+1, y, p) * p + game_win_prob(x, y+1, p) * (1-p)
        game_data[(x,y,p)] = results
        return results

#--------------------------------------------------------------------------------------------
# Determines Chance of Winning a Set given chance of winning when serving/returning
#--------------------------------------------------------------------------------------------

def set_win_prob(x, y, s_game, r_game, s_pt, r_pt, final_score = None, win = 1 ):
    if (x, y, s_pt, r_pt, final_score, win) in set_data:
        return set_data[(x, y, s_pt, r_pt, final_score, win)]
    
    if final_score == None:
        if (x == 6 and y <= 4) or (x == 7):
            return win
        elif (x <= 4 and y == 6) or (y == 7):
            return 1.-win
    elif final_score == 'odd' and win:
        if (x == 6 and y <= 4) or (x == 7):
            if (x+y) % 2 == 1:
                return 1.
            else:
                return 0.
        elif (x <= 4 and y == 6) or (y == 7):
            return 0.
    elif final_score == 'odd' and 1.-win:
        if (x <= 4 and y == 6) or (y == 7):
            if (x+y) % 2 == 1:
                return 1.
            else:
                return 0.
        elif (x == 6 and y <= 4) or (x == 7):
            return 0.
    elif final_score == 'even' and win:
        if (x == 6 and y <= 4) or (x == 7):
            if (x+y) % 2 == 0:
                return 1.
            else:
                return 0.
        elif (x <= 4 and y == 6) or (y == 7):
            return 0.
    elif final_score == 'even' and 1.-win:
        if (x <= 4 and y == 6) or (y == 7):
            if (x+y) % 2 == 0:
                return 1.
            else:
                return 0.
        elif (x == 6 and y <= 4) or (x == 7):
            return 0.
    
    if (x == 6 and y == 6):
        results = tiebreak_win_prob(0,0, s_pt, r_pt) * set_win_prob(x+1, y,s_game,r_game, 
                                                                 s_pt, r_pt, final_score = final_score, win=win) + \
               (1.-tiebreak_win_prob(0,0, s_pt, r_pt)) * set_win_prob(x, y+1, s_game,r_game, 
                                                                      s_pt, r_pt, final_score = final_score, win=win)
    else:
        results = set_win_prob(y+1, x, 1.-r_game, 1.-s_game, 1.-r_pt, 
                            1.-s_pt, final_score = final_score, win = 1.-win) * (1-s_game) + \
               set_win_prob(y, x+1, 1.-r_game, 1.-s_game, 1.-r_pt, 
                            1.-s_pt, final_score = final_score, win = 1.-win) * s_game
    
    set_data[(x, y, s_pt, r_pt, final_score, win)] = results
    return results


#--------------------------------------------------------------------------------------------
# Determines Chance of Winning a Tiebreak given chance of winning when serving/returning
#--------------------------------------------------------------------------------------------

def tiebreak_win_prob(x, y, s, r):
    if (x,y,s,r) in tiebreak_data:
        return tiebreak_data[(x,y,s,r)]
    
    w = (s*r)/(1-(s*(1.-r)+(1.-s)*r))   
    if (x == 7 and y <= 5):
        return 1.
    elif (x <= 5 and y == 7):
        return 0.
    elif x >= 6 and y >= 6:
        if x == y:
            results = w
        elif x > y:
            if (x+y) % 4 == 1:
                results = r + (1.-r)*w
            else:
                results = s + (1.-s)*w
        elif x < y:
            if (x+y) % 4 == 1:
                results = r*w
            else:
                results = s*w
        tiebreak_data[(x,y,s,r)] = w
        tiebreak_data[(x+1,y,s,r)] = r + (1.-r)*w
        tiebreak_data[(x,y+1,s,r)] = r*w
        tiebreak_data[(x+1,y+1,s,r)] = w
        tiebreak_data[(x+2,y+1,s,r)] = s + (1.-s)*w
        tiebreak_data[(x+1,y+2,s,r)] = s*w
            
    elif (x+y) % 4 == 0 or (x+y) % 4 == 3:
        results = tiebreak_win_prob(x, y+1,s,r) * (1.-s) + tiebreak_win_prob(x+1, y,s,r) * s
    elif (x+y) % 4 == 1 or (x+y) % 4 == 2:
        results = tiebreak_win_prob(x, y+1,s,r) * (1.-r) + tiebreak_win_prob(x+1, y,s,r) * r
    tiebreak_data[(x,y,s,r)] = results
    return results

#--------------------------------------------------------------------------------------------
# Uses the functions above to determine chance of winning in any game situation
#--------------------------------------------------------------------------------------------

def win_prob(s1, s2, g1, g2, p1, p2, s_pt, r_pt):
    if (s1, s2, g1, g2, p1, p2, s_pt, r_pt) in match_data:
        return match_data[(s1, s2, g1, g2, p1, p2, s_pt, r_pt)]
    
    if s1 == 2 and s2 < 2:
        return 1.
    elif s1 < 2 and s2 == 2:
        return 0.
    
    if g1 == 6 and g2 == 6:
        first_set_win_odd = tiebreak_win_prob(p1,p2,s_pt, r_pt)
        first_set_loss_odd = 1.-tiebreak_win_prob(p1,p2,s_pt,r_pt)
        first_set_win_even, first_set_loss_even = 0.,0.
    else:    
        s_game, r_game = game_win_prob(0,0,s_pt), game_win_prob(0,0,r_pt)

        first_game_win = game_win_prob(p1, p2, s_pt)

        first_set_loss_odd = first_game_win * set_win_prob(g2, g1+1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                           final_score = 'odd', win = 1) + \
                             (1.-first_game_win) * set_win_prob(g2+1, g1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                                final_score = 'odd', win = 1)

        first_set_loss_even = first_game_win * set_win_prob(g2, g1+1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                           final_score = 'even', win = 1) + \
                             (1.-first_game_win) * set_win_prob(g2+1, g1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                                final_score = 'even', win = 1)

        first_set_win_odd = first_game_win * set_win_prob(g2, g1+1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                           final_score = 'odd', win = 0) + \
                             (1.-first_game_win) * set_win_prob(g2+1, g1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                                final_score = 'odd', win = 0)

        first_set_win_even = first_game_win * set_win_prob(g2, g1+1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                           final_score = 'even', win = 0) + \
                             (1.-first_game_win) * set_win_prob(g2+1, g1, 1.-r_game, 1.-s_game, 1.-r_pt, 1.-s_pt,
                                                                final_score = 'even', win = 0)

    if (g1+g2) % 2 == 0:
        results = first_set_loss_odd * (1.-win_prob(s2+1, s1, 0, 0, 0, 0, 1.-r_pt, 1.-s_pt)) + \
                  first_set_loss_even * win_prob(s1, s2+1, 0, 0, 0, 0, s_pt, r_pt) + \
                  first_set_win_odd * (1.-win_prob(s2, s1+1, 0, 0, 0, 0, 1.-r_pt, 1.-s_pt)) + \
                  first_set_win_even * win_prob(s1+1, s2, 0, 0, 0, 0, s_pt, r_pt)
    else:
        results = first_set_loss_even * (1.-win_prob(s2+1, s1, 0, 0, 0, 0, 1.-r_pt, 1.-s_pt)) + \
                  first_set_loss_odd * win_prob(s1, s2+1, 0, 0, 0, 0, s_pt, r_pt) + \
                  first_set_win_even * (1.-win_prob(s2, s1+1, 0, 0, 0, 0, 1.-r_pt, 1.-s_pt)) + \
                  first_set_win_odd * win_prob(s1+1, s2, 0, 0, 0, 0, s_pt, r_pt)
        match_data[(s1, s2, g1, g2, p1, p2, s_pt, r_pt)] = results
        
    return results
    
#--------------------------------------------------------------------------------------------
# Test Match Win Percentage given different Serve/Return Point Win Percentages
#--------------------------------------------------------------------------------------------
    
def test_diff_serve_win_pct(s_base, r_base, inc, lim):
    grid_data = []
    # s_base, r_base, inc, lim = 0.65, 0.35, 0.005, 0.1
    for s in np.arange(s_base - lim, s_base + lim + inc, inc):
        print s
        for r in np.arange(r_base - 0.1, r_base + 0.11, 0.005):
            grid_data.append([s, r, win_prob(0,0,0,0,0,0,s,r)])
    grid_data = pd.DataFrame(grid_data, columns = ['Serve Win %', 'Return Win %', 'Win %'])
    
    return grid_data

#--------------------------------------------------------------------------------------------
# Plots the 3D Results from Function Above
#--------------------------------------------------------------------------------------------

def plot_diff_serve_win_pct_3d(grid_data):
    fig = plt.figure(figsize = (8,6))
    ax = fig.gca(projection='3d')
    ax.scatter(grid_data['Serve Win %'], grid_data['Return Win %'], grid_data['Win %'],
               c = grid_data['Win %'], cmap = 'RdYlGn', s = 40)
    ax.set_xlim(0.54, 0.76)
    ax.set_ylim(0.24, 0.46)
    ax.set_zlim(-0.01, 1.01)
    ax.set_xlabel('\nServe Win %')
    ax.set_ylabel('\nReturn Win %')
    ax.set_zlabel('\nWin %')
    ax.set_title('Match Win Odds Given Serve Win and Return Win Probability\n')


#--------------------------------------------------------------------------------------------
# Plots the 2D Version from Function Above
#--------------------------------------------------------------------------------------------

def plot_diff_serve_win_pct_2d(grid_data):
    rowidx = np.arange(0, 41*41, 43)
    subset_data = grid_data.loc[rowidx]
    fig, ax = plt.subplots(1,1)
    ax.scatter(subset_data['Serve Win %'], subset_data['Win %'])
    ax.set_xlim(0.54, 0.76)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('\n Serve Win %')
    ax.set_ylabel('Match Win %\n')
    ax.set_title('Match Win Odds given Serve Win %\n(Return Win % is 30% Lower)\n')


#--------------------------------------------------------------------------------------------
# Determines the Serve/Return Point Win % that resembled the desinated match win percentage
#--------------------------------------------------------------------------------------------

def get_win_prob(match_prob, grid_x, grid_y):
    grid_x = grid_x.values
    grid_y = grid_y.values
    idx = min([i for i,y in enumerate(grid_y) if y >= match_prob])
    s_win = grid_x[idx-1] + (grid_x[idx] - grid_x[idx-1]) * (match_prob - grid_y[idx-1])/(grid_y[idx] - grid_y[idx-1])
    return s_win, s_win - 0.3


#--------------------------------------------------------------------------------------------
# These 2 Functions Ensure Correct Point Transformations
#--------------------------------------------------------------------------------------------

def transform_pt(p1):
    pt_dict = {'0': 0, '15':1, '30':2, '40':3, 'AD':4}
    return pt_dict[p1]


def transform_pt_t(p1, p2):
    p1 = p1.split(' (')[0]
    p2 = p2.split(' (')[0]
    if int(p1) + int(p2) >= 12:
        if (int(p1) + int(p2)) % 4 == 0:
            return 6,6
        elif (int(p1) + int(p2)) % 4 == 2:
            return 7,7
        elif (int(p1) + int(p2)) % 4 == 1 and int(p1) > int(p2):
            return 7,6
        elif (int(p1) + int(p2)) % 4 == 1 and int(p1) < int(p2):
            return 6,7
        elif (int(p1) + int(p2)) % 4 == 3 and int(p1) > int(p2):
            return 8,7
        elif (int(p1) + int(p2)) % 4 == 3 and int(p1) < int(p2):
            return 7,8
    else:
        return int(p1), int(p2)

#--------------------------------------------------------------------------------------------
# The master function that calls functions above and determine game situation win probability
#--------------------------------------------------------------------------------------------

def master_win_prob(s1, s2, g1, g2, p1, p2, match_win):
    if g1 == 6 and g2 == 6:
        p1, p2 = transform_pt_t(p1,p2)
    else:
        p1, p2 = transform_pt(p1), transform_pt(p2)
    s, r = get_win_prob(match_win, subset_data['Serve Win %'], subset_data['Win %'])
    return win_prob(s1, s2, g1, g2, p1, p2,s,r)


#--------------------------------------------------------------------------------------------
# Part 4: The Predictive Model
#
# Description: Using the grouped datasets and base model, we make predictions on a
# matchup by matchup point by point basis.
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Returns the actual win probability based on factors described above. Also returns the
# beta distribution related to that prediction to showcase confidence of prediction.
#--------------------------------------------------------------------------------------------

def get_posterior_spec(prior_data, real_data, matchup_data, surface_data, all_matches, h2h_data, base_data,
                       matchup, setscore, gamescore, serving, pointscore, surface, 
                       players = None, date = None, months = None,
                       point_param = 10, matchup_param = 10, s_param = 10,
                       r_param = 10, h2h_param = 10, base_param = 100,
                       show=True, return_estimate=True):
    
    max_pt_data = np.log(prior_data['Number of Instances'].max()) + 2
    max_matchup_data = np.log(matchup_data['Number of Instances'].max()) + 2
    max_s_data = np.log(surface_data['Number of Instances'].max()) + 2
    max_h2h_data = h2h_data.max(skipna = True).max()
    
    real_data.index = real_data['Set Score'] + real_data['Game Score'] + \
                      real_data['Serving at Start of Game?'] + real_data['Point Score'] +  \
                      real_data['Matchup']
    
    # 0) Base Data based on Probability Model
    m1, m2 = matchup.split(' vs. ')
    bwin = base_data.loc[m1][m2]
    s1, s2 = setscore.split('-')
    g1, g2 = gamescore.split('-')
    if pointscore == 'Start of Game':
        p1, p2 = '0', '0'
    else:
        p1, p2 = pointscore.split('-')
    if serving == 'Serving':
        bprob = master_win_prob(int(s1), int(s2), int(g1), int(g2), p1, p2, bwin)
    else:
        bprob = 1. - master_win_prob(int(s2), int(s1), int(g2), int(g1), p2, p1, 1.-bwin)
    #print 'Base Prob: ' + str(bprob)
    prior_alpha_0 = base_param * bprob
    prior_beta_0 = base_param * (1.-bprob)
    
    # 1) Prior Data Containing Overall Point Matchup
    row = prior_data.loc[setscore + gamescore + serving + pointscore]
            
    tot = (np.log(row['Number of Instances']) + 2) / max_pt_data * point_param
    prior_alpha = tot * row['% Win']
    prior_beta = tot * (1. - row['% Win'])
    
    # 2) Prior Data Containing Overall Matchup Situation
    m_win, m_instance = matchup_data.loc[matchup].values
    mtot = (np.log(m_instance) + 2) / max_matchup_data * matchup_param
    prior_alpha_2 = mtot * m_win / m_instance
    prior_beta_2 = mtot * (1. - m_win / m_instance)
    
    # 3) Surface Data Containing Surface + Point Situation
    if setscore + gamescore + serving + pointscore + surface in surface_data.index:
        srow = surface_data.loc[setscore + gamescore + serving + pointscore + surface]
        stot = (np.log(srow['Number of Instances']) + 2) / max_s_data * s_param
        prior_alpha_3 = stot * srow['% Win']
        prior_beta_3 = stot * (1. - srow['% Win'])
    else:
        prior_alpha_3, prior_beta_3 = 0.,0.
    
    # 4) Recent Form Data
    if players != None:
        p1, p2 = players
        p1win, p1loss = recent_activity(p1, date, all_matches, months, surface = surface)
        p2win, p2loss = recent_activity(p2, date, all_matches, months, surface = surface)
        #print 'Player 1 Win Loss: ' + str((p1win, p1loss))
        #print 'Player 2 Win Loss: ' + str((p2win, p2loss))
        scale1 = 1.*(p1win+p1loss)/(100.*months/12.) * r_param
        scale2 = 1.*(p2win+p2loss)/(100.*months/12.) * r_param
        
        prior_alpha_4 = 1.*p1win/max(p1win+p1loss,1)*scale1 + 1.*p2loss/max(p2win+p2loss,1)*scale2
        prior_beta_4 = 1.*p1loss/max(p1win+p1loss,1)*scale1 + 1.*p2win/max(p2win+p2loss,1)*scale2
        
        #print prior_alpha_4, prior_beta_4
        
    else:
        prior_alpha_4, prior_beta_4 = 0., 0.
        
    # 5) H2H Data
    if players != None:
        p1, p2 = players
        if p1 in h2h_data.index and p2 in h2h_data.index:
            p1win = h2h_data.ix[p1,p2]
            p2win = h2h_data.ix[p2,p1]
        else:
            p1win, p2win = 0., 0.
        #print 'H2H: ' + str((p1win, p2win))
        prior_alpha_5 = 1.*p1win/max_h2h_data * h2h_param
        prior_beta_5 = 1.*p2win/max_h2h_data * h2h_param
        
        #print prior_alpha_5, prior_beta_5
    else:
        prior_alpha_5, prior_beta_5 = 0., 0.
    
    
    # 5) New Data Containing Point + Matchup
    idx = setscore + gamescore + serving + pointscore + matchup
    if idx in real_data.index.values:
        row_data = real_data.ix[idx,['% Win','Number of Instances']]
        pct_win, instances = row_data.values
        real_alpha = int(round(pct_win * instances,0))
        real_beta = instances - real_alpha
    else:
        real_alpha, real_beta = 0,0
        
    a = prior_alpha_0 + prior_alpha + prior_alpha_2 + prior_alpha_3 + prior_alpha_4 + prior_alpha_5 + real_alpha
    b = prior_beta_0 + prior_beta + prior_beta_2 + prior_beta_3 + prior_beta_4 + prior_beta_5 + real_beta

    if show:
        print (a, b, round(100.*a/(a+b),2),round(100*beta.ppf(0.5, a, b),2))
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0,1,0.001)
        ax.plot(x, beta.pdf(x, a, b),
                'r-', lw=3, label='beta pdf')
        ax.plot((beta.ppf(0.16, a, b),beta.ppf(0.16, a, b)),
                (0.,beta.pdf(beta.ppf(0.16, a, b), a, b)),
                'r-', lw=2)
        ax.plot((beta.ppf(0.84, a, b),beta.ppf(0.84, a, b)),
                (0.,beta.pdf(beta.ppf(0.84, a, b), a, b)),
                'r-', lw=2)  
        plt.show()
    
    if return_estimate:
        estimate, lo, hi = beta.ppf(0.5, a, b), beta.ppf(0.2, a, b), beta.ppf(0.8,a,b)
        return estimate, lo, hi, a, b


#--------------------------------------------------------------------------------------------
# Part 5: Testing the Predictive Model
#
# Description: This section is used to run the predictive model on various parameters and
# determine the parameters that produce the best results on the testing data (2015 matches)
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Returns the predicted values from get_posterior_spec for matches randomly selected
# beyond a cutoff date (which determines the test dataset)
#--------------------------------------------------------------------------------------------

def test_parameters(full_data, all_matches, datecutoff, testnum, params,
                    training_data = None):
    
    if training_data == None:
        prior_data, real_data, matchup_data,surface_data, all_matches_b, h2h_data = \
        setup_all_datasets(full_data, all_matches, datecutoff)
    else:
        prior_data, real_data, matchup_data,         surface_data, all_matches_b, h2h_data = training_data
    
    pbp_data_test = full_data[(full_data['Week_1R'] >= datecutoff) &
                              (full_data['Week_2R'] >= datecutoff)].sample\
                            (testnum)
    
    print 'Test Data Set Up: ' + str(len(pbp_data_test)) + ' Rows'
    point_param, matchup_param, s_param, r_param, h2h_param, base_param = params
    
    results = []

    for i, row in pbp_data_test.iterrows():
        if row['Server'] == 1:
            serving = 'Serving'
        else:
            serving = 'Returning'
            
        if str(row['p1Game']) + '-' + str(row['p2Game']) == '6-6':
            point_score = tiebreak_state((row['p1Score'],row['p2Score'],
                                          serving))
        else:
            point_score = game_states[(row['p1Score'],row['p2Score'])]
        
        prob = get_posterior_spec(prior_data, real_data, matchup_data, 
                                  surface_data, all_matches, h2h_data, matchup_grid,
                                  row['Ranking Matchup'], 
                                  str(row['p1Set'])+'-'+str(row['p2Set']),
                                  str(row['p1Game'])+'-'+str(row['p2Game']), 
                                  serving, 
                                  point_score, row['Surface'],
                                  players = (int(row['Player ID_1']),
                                             int(row['Player ID_2'])), 
                                  date = datecutoff, months = 3,
                                  point_param = point_param, 
                                  matchup_param = matchup_param, 
                                  s_param = s_param, r_param = r_param, 
                                  h2h_param = h2h_param,
                                  base_param = base_param,
                                  show=False, return_estimate=True)
        winner = row['Winner'] % 2
        results.append((prob, winner))
        
    return results

#--------------------------------------------------------------------------------------------
# Calls the Function Above and Stores the Predictions in CSVs
#--------------------------------------------------------------------------------------------

def store_analysis_data():
    datecutoff = 20150101
    count = 2000

    prior_data, real_data, matchup_data, surface_data, all_matches_b, h2h_data = \
    setup_all_datasets(full_data, all_matches, datecutoff)

    print 'Data Collected'

    param_diffs = (5,10)
    b_param_diffs = (100,300)

    idx = 0
    for p in param_diffs:
        for m in param_diffs:
            for s in param_diffs:
                for r in param_diffs:
                    for h in param_diffs:
                        for b in b_param_diffs:
                            print idx
                            results = test_parameters(full_data, all_matches, 
                                                          datecutoff, count, 
                                                          (p,m,s,r,h,b),
                                                          training_data = (prior_data, 
                                                                           real_data, 
                                                                           matchup_data,
                                                                           surface_data, 
                                                                           all_matches_b, 
                                                                           h2h_data))
                            pd.DataFrame(results, columns = ['Prob','Results']).to_csv(                                    
                                    'Parameter Testing Results/'+str(p)+'_'+str(m)+'_'+str(s)+'_'+str(r)+                       
                                    '_'+str(h)+'_'+str(b)+'.csv')
                            idx = idx + 1


#--------------------------------------------------------------------------------------------
# Retrieves the Analysis Results and Plots the Accuracy of the Predictions by Probability
# Bins - the closer the predicted point is in the bin, the better the model is
#--------------------------------------------------------------------------------------------

def analyze_results(results, bins, show_graph = True):
    bin_probs = np.zeros((bins,2))
    for prob, win in results:
        bin_probs[int(tuple(map(float, prob[1:-1].split(',')))[0]*bins)][0] =  \
        bin_probs[int(tuple(map(float, prob[1:-1].split(',')))[0]*bins)][0] + win
        bin_probs[int(tuple(map(float, prob[1:-1].split(',')))[0]*bins)][1] =  \
        bin_probs[int(tuple(map(float, prob[1:-1].split(',')))[0]*bins)][1] + 1
    
    x = list(np.arange(0.5/bins,1.,1./bins))
    all_x = list(np.arange(0.,1.,1./bins)) + [1.]

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize=(15,8))
        ax.plot(x, [y[0]/y[1] for y in bin_probs],
                    'b.', ms=12)
        for k,y in zip(x,bin_probs):
            ax.plot((k,k),(k-1.96*((k*(1-k))/y[1])**0.5,k+1.96*((k*(1-k))/y[1])**0.5),
                    'g-', lw=2)
    
        for i,val in enumerate(x):
            ax.plot((val,val),(all_x[i],all_x[i+1]),'r-', lw=2)  

        ax.set_ylim(-0.05,1.05)
        ax.set_ylabel('Win Probability\n', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(all_x[i]*100))+'-\n'+str(int(all_x[i+1]*100))+'%' 
                            for i,d in enumerate(all_x[:-1])])
        ax.set_xlabel('\nWin Prob Buckets', fontsize = 15)
        ax.set_title('\nTest Model Results\n', fontsize=17)
        zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
        zed = [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()]
    
    error = sum([(xidx-y[0]/y[1])**2 for xidx, y in zip(x, bin_probs)])
    out_of_bounds = sum([(y[0]/y[1] > all_x[i+1]) or (y[0]/y[1] < all_x[i])
                         for i,y in enumerate(bin_probs)])
    out_of_bounds_loose = sum([(y[0]/y[1] > k+1.96*((k*(1-k))/y[1])**0.5) 
                               or (y[0]/y[1] < k-1.96*((k*(1-k))/y[1])**0.5)
                         for k,y in zip(x,bin_probs)])
    decreasing = sum([p[0]/p[1] > q[0]/q[1] 
                      for p,q in zip(bin_probs[:-1],bin_probs[1:])])
    
    return error, out_of_bounds, out_of_bounds_loose,decreasing,\
           zip(np.round(np.arange(0.5/bins,1.,1./bins),3),
               [round(k[0]/k[1],4) for k in bin_probs],[int(k[1]) for k in bin_probs])


#--------------------------------------------------------------------------------------------
# Calls the Functions Above and Stores the Important Error Metrics
#--------------------------------------------------------------------------------------------

def retrieve_parameter_testing_results():
    error_data = []
    for p in param_diffs:
        for m in param_diffs:
            for s in param_diffs:
                for r in param_diffs:
                    for h in param_diffs:
                        for b in b_param_diffs:
                            results = pd.read_csv('Parameter Testing Results/' + str(p)
                                                  + '_' + str(m) + '_' + str(s) 
                                                  + '_' + str(r) + '_' + str(h) + '_' + str(b)
                                                  + '.csv',
                                                  index_col=0)
                            #print results.values
                            e, ob, obl, d, rd = analyze_results(results.values, 20, 
                                                               show_graph = False)
                            error_data.append(((p,m,s,r,h,b), e, ob, obl, d))

#--------------------------------------------------------------------------------------------
# Displays Error Results for a Particular Set of Parameters
#--------------------------------------------------------------------------------------------
                            
def show_parameter_testing_result(p,m,s,r,h,b):                          
    results = pd.read_csv('Parameter Testing Results/'+str(p)+'_'+str(m)+'_'+str(s)+'_'+str(r)+                       
                                    '_'+str(h)+'_'+str(b)+'.csv',
                                              index_col=0)
    error, ob, d, dl, results_data = analyze_results(results.values, 20)

#--------------------------------------------------------------------------------------------
# Part 6: Presenting the Results in Real Time
#
# Description: This section retrieves real time data and produces PNG that is shown on the
# Github Repository site as soon as it is updated
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Updates Existing Match Data with Live Data and Updates PNG file to display in
# live dashboard.
#--------------------------------------------------------------------------------------------

def plot_match(date, param, filename, full_data, all_matches, 
               rankings, surface, training_data, pickle_dict = dict(),
               timestamp = ''):
    
    hfont = {'fontname':'Helvetica'}
    
    data = pd.read_csv(filename,index_col = 0)
    
    if len(data) > 0:
        datecutoff = int(date.replace('-',''))

        all_weeks = np.unique(rankings.index)
        rank_week = max([i for i in all_weeks if i <= datecutoff])
        rankings = rankings.loc[rank_week]
        
        try:
            r1 = classifyRank(float(rankings[rankings['Player ID'] 
                                       == data['Player 1 ID'].values[0]]
                              ['Ranking'].values[0]))
        except IndexError:
            print 'Ranking Not Found for Player 1 - assume ranking 100+'
            r1 = 'Outside Top 100'
        
        try:
            r2 = classifyRank(float(rankings[rankings['Player ID']
                                   == data['Player 2 ID'].values[0]]
                         ['Ranking'].values[0]))
        except IndexError:
            print 'Ranking Not Found for Player 2 - assuming ranking 100+'
            
            r2 = 'Outside Top 100'
        matchup = r1 + ' vs. ' + r2

        if surface == None:
            surface = data['Surface'].values[0]

        if training_data == None:
            prior_data, real_data, matchup_data, surface_data, all_matches_b, h2h_data = \
            setup_all_datasets(full_data, all_matches, datecutoff)
        else:
            prior_data, real_data, matchup_data, surface_data, all_matches_b, h2h_data = training_data

        print 'Data Setup Complete'

        pointparam, matchparam, sparam, rparam, h2hparam, baseparam = param

        try:
            xlabels, xlabel2, xlabel3, plot_data, lo_data, hi_data, xticks, xticks_minor, xticks_set = \
            pickle.loads(pickle_dict[filename])
        except (IndexError,KeyError):
            xlabels,xlabel2,xlabel3,plot_data,lo_data,hi_data,xticks,xticks_minor,xticks_set = [],[],[],[],[],[],[],[],[]

        try:
            old_data = pd.read_csv(filename.replace('_live','_archive'),index_col = 0)
            prevset, prevgame = old_data.loc[max(len(old_data) - len(data) - 1,0)]['Set Score'], \
                                old_data.loc[max(len(old_data) - len(data) - 1,0)]['Game Score']
        except IOError:
            prevset, prevgame = data['Set Score'].values[0], data['Game Score'].values[0]

        if len(xlabels) == 0:
            count = 1
        else:
            count = max(xticks) + 1

        for i,row in data.iterrows():
            if row['Game Score'] != prevgame and row['Set Score'] == prevset:
                xticks_minor.append(count)
                xlabels.append('')
                xlabel2.append('')
                xlabel3.append('')
                count = count + 1

            if row['Set Score'] != prevset:
                xticks_minor.append(count)
                xticks_set.append(count)
                xlabels.append('')
                xlabel2.append('')
                xlabel3.append('')
                count = count + 1

            if row['Game Score'] != '6-6' and row['Point Score'] == '0-0':
                row['Point Score'] = 'Start of Game'

            if row['Game Score'] == '6-6':
                p1, p2 = row['Point Score'].split('-')
                row['Point Score'] = tiebreak_state((int(p1),int(p2),
                                                     row['Serving?']))

            row['Point Score'] = row['Point Score'].replace('A','AD')

            prob, lo, hi, a, b = get_posterior_spec(prior_data, real_data, matchup_data, 
                                                      surface_data, all_matches, h2h_data, matchup_grid,
                                                      matchup, 
                                                      row['Set Score'],
                                                      row['Game Score'], 
                                                      row['Serving?'], 
                                                      row['Point Score'], surface,
                                                      players = (row['Player 1 ID'],
                                                                 row['Player 2 ID']), 
                                                      date = datecutoff, months = 3,
                                                      point_param = pointparam, 
                                                      matchup_param = matchparam, 
                                                      s_param = sparam, r_param = rparam, 
                                                      h2h_param = h2hparam,
                                                      base_param = baseparam,
                                                      show=False, return_estimate=True)
            prob, lo, hi = prob*100, lo*100, hi*100
            print 'New Prob: ' + str(prob)

            xlabels.append(row['Game Score'])
            xlabel2.append(row['Point Score'])
            xlabel3.append(row['Set Score'])
            xticks.append(count)

            plot_data.append(prob)
            lo_data.append(lo)
            hi_data.append(hi)

            prevset, prevgame = row['Set Score'], row['Game Score']
            count = count + 1

        p_str = pickle.dumps([xlabels, xlabel2, xlabel3, plot_data, 
                              xticks, xticks_minor, xticks_set])
        pickle_dict[filename] = p_str

        game_ticks = [(x+y)/2. for x,y in zip([0]+xticks_minor,
                                             xticks_minor + [max(xticks)+1])]
        game_labels = [xlabels[int(k)-1] for k in game_ticks]

        set_ticks = [(x+y)/2. for x,y in zip([0]+xticks_set,
                                             xticks_set + [max(xticks)+1])]
        set_labels = [xlabel3[int(k)-1] for k in set_ticks]

        fig = plt.figure(figsize=(30,15))
        gs = gridspec.GridSpec(1, 2, width_ratios=[7, 1])
        ax = plt.subplot(gs[0])
        ax.plot(xticks, plot_data, marker='o', color='g', lw=7, ms=20)
        ax.fill_between(xticks, lo_data, hi_data, facecolor = 'green', alpha = 0.3)
        ax.plot(xticks, [100.-z for z in plot_data], marker='o', color='r', lw=7, ms=20)
        ax.fill_between(xticks, [100.-z for z in hi_data], [100.-z for z in lo_data], 
                        facecolor = 'red', alpha = 0.3)
        ax2 = ax.twiny()

        ax.set_xticks(game_ticks)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(game_labels, **hfont)
        ax.set_xlim(0,max(xticks)+1)

        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=37)
        ax.set_ylabel('Win Percentage\n', fontsize = 35, **hfont)
        ax.set_ylim(0,100)
        ax.plot((0,max(xticks)+1),(50,50), linestyle = '--', alpha = 0.7, 
                lw = 1.5, color = 'k')
        ax.plot((0,max(xticks)+1),(20,20), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        ax.plot((0,max(xticks)+1),(40,40), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        ax.plot((0,max(xticks)+1),(60,60), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        ax.plot((0,max(xticks)+1),(80,80), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        for val in xticks_minor:
            ax.plot((val,val),(0,100), linestyle = '--', alpha = 0.7, 
                    lw = 1.5, color = 'k')

        for val in xticks_set:
            ax.plot((val,val),(0,100), linestyle = '-', alpha = 0.9, 
                    lw = 2.25, color = 'k')

        for x,y1,y,y2 in zip(xticks[1:-1], plot_data[:-2], 
                       plot_data[1:-1], plot_data[2:]):
            if y - y1 > 3 and y - y2 > 3:
                ax.annotate(xlabel2[x-1].replace('Start of Game','0-0'), 
                                xy=(x, y + 4), fontsize = 30,
                                horizontalalignment='center', 
                                verticalalignment='center')
            elif y1 - y > 3 and y2 - y > 3:
                ax.annotate(xlabel2[x-1].replace('Start of Game','0-0'), 
                                xy=(x, y - 4), fontsize = 30,
                                horizontalalignment='center', 
                                verticalalignment='center')

        s1, s2 = row['Set Score'].split('-')
        g1, g2 = row['Game Score'].split('-')
        if s1 > s2 or (s1 == s2 and g1 > g2):
            title_update = row['Player 1'][:-3] + ' Leads ' + xlabel3[-1] + ' ' + xlabels[-1] + ' ' + xlabel2[-1]
        elif s1 == s2 and g1 == g2:
            title_update = 'Tied at ' + xlabel3[-1] + ' ' + xlabels[-1] + ' ' + xlabel2[-1]
        else:
            if xlabel2[-1] == 'Start of Game':
                plabel = xlabel2[-1]
            else:
                plabel = xlabel2[-1].split('-')[1] + '-' + xlabel2[-1].split('-')[0]
            schange = xlabel3[-1].split('-')[1] + '-' + xlabel3[-1].split('-')[0] + ' ' +  \
                      xlabels[-1].split('-')[1] + '-' + xlabels[-1].split('-')[0] + ' ' + plabel
            title_update = row['Player 2'][:-3] + ' Leads ' + schange
                      
        
        ax.set_title('\n' + row['Player 1'] + ' vs. ' + row['Player 2'] + timestamp + '\n' +
                     title_update + '\n\n', fontsize=45, **hfont)
        ax.grid( 'off', axis='x' )
        ax.grid( 'off', axis='x', which='minor' )
        ax.tick_params( axis='x', which='minor', length=30 )
        ax.tick_params( axis='x', which='major', bottom='off', top='off' )

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(set_ticks)
        ax.set_xticks(xticks_set, minor=True)
        ax2.set_xticklabels(set_labels, **hfont)
        ax2.tick_params(axis='x', labelsize=30)

        axd = plt.subplot(gs[1])
        x = np.arange(0., 1., 0.001)
        axd.plot(beta.pdf(x, a, b),x*100,
                'g-', lw=3, label='beta pdf')
        axd.plot(beta.pdf(x, b, a),x*100,
                'r-', lw=3, label='beta pdf')
        axd.fill_betweenx(x*100, 0., beta.pdf(x, a, b), alpha = 0.3, facecolor = 'green')
        axd.fill_betweenx(x*100, 0., beta.pdf(x, b, a), alpha = 0.3, facecolor = 'red')
        axd.plot((0,max(beta.pdf(x,a,b))*1.1),(50,50), linestyle = '--', alpha = 0.7, 
                lw = 1.5, color = 'k')
        axd.plot((0,max(beta.pdf(x,a,b))*1.1),(20,20), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        axd.plot((0,max(beta.pdf(x,a,b))*1.1),(40,40), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        axd.plot((0,max(beta.pdf(x,a,b))*1.1),(60,60), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        axd.plot((0,max(beta.pdf(x,a,b))*1.1),(80,80), linestyle = '--', alpha = 0.55, 
                lw = 1, color = 'k')
        axd.set_ylim(0,100)
        axd.set_yticklabels([])
        axd.set_xticklabels([])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.75, bottom = 0.1, left = 0.15, right = 0.875)
        plt.savefig(filename[:-4]+'.png')
        plt.show()
    
    check = pd.DataFrame([[1]])
    check.to_csv(filename.replace('_live','_updated'))
    
    remove_live = pd.DataFrame(columns = ['Player 1', 'Player 1 ID',
                                          'Player 2', 'Player 2 ID',
                                          'Set Score', 'Game Score', 
                                          'Point Score', 'Serving?'])
    remove_live.to_csv(filename)
    
    return pickle_dict

#--------------------------------------------------------------------------------------------
# Identifies Matches to Update and Calls Function Above
#--------------------------------------------------------------------------------------------

def plot_matches_for_date(date, params, full_data, all_matches, 
                          rankings, surface = None, training_data = None,
                          pickle_dict = dict(), timestamp = ''):
    csv_files = glob.glob("Scraped Matches/" + date + "*_live.csv")
    for filename in csv_files:
        pickle_dict = plot_match(date, params, filename, full_data, all_matches,
                                          rankings, surface, training_data, pickle_dict, timestamp)
    return pickle_dict, 

#--------------------------------------------------------------------------------------------
# Concatenate Images to Update Live Scoreboard
#--------------------------------------------------------------------------------------------

def concatenate_images():
    date = str(datetime.today().strftime('%Y-%m-%d'))
    png_files = glob.glob("Scraped Matches/" + date + "*.png")
    print str(len(png_files)) + ' Live Matches Found'
    
    new_im = Image.new('RGB', (800,400*max(1,len(png_files))))
    
    for i,png_file in enumerate(png_files):
        im = Image.open(png_file)
        im.thumbnail((800,400))
        new_im.paste(im, (0,400*i))

    new_im.save("Scraped Matches/Live_Scoreboard.png")
    
#--------------------------------------------------------------------------------------------
# The function to call for tracking live scores when the live scraper is in effect.
#--------------------------------------------------------------------------------------------

def track_live_scores(full_data, all_matches, param, seconds, rankings, t_data = None):
    datestr = str(datetime.today().strftime('%Y-%m-%d'))
    dateint = int(datestr.replace('-',''))
    
    if t_data == None:
        prior_data, real_data, matchup_data, surface_data, all_matches_b, h2h_data = \
                    setup_all_datasets(full_data, all_matches, dateint)
    
        training_data = (prior_data, real_data, matchup_data, surface_data,
                         all_matches_b, h2h_data)
    else:
        training_data = t_data
    
    start = time.time()
    end = time.time()
    pickle_data = dict()
    
    while end - start < seconds:
        print 'Updating Graphs'
        tstamp = ' at ' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ':' + str(datetime.now().second)
        pickle_data = plot_matches_for_date(datestr, param, full_data, all_matches, 
                                            rankings, surface = 'Hard',
                                            training_data = training_data,
                                            pickle_dict = pickle_data,
                                            timestamp = tstamp)
        concatenate_images()
        time.sleep(10)
        end = time.time()


#--------------------------------------------------------------------------------------------
# Part 8: Main Function
#
# Description: Ties All the Functions Together and Makes Live Probabilities
#--------------------------------------------------------------------------------------------

def main():
    datecutoff = 20151231
    full_data, all_matches, rankings = create_clean_data()
    print 'Data Cleaned'
    
    prior_data, real_data, matchup_data, surface_data, all_matches, h2h_data = \
                setup_all_datasets(full_data, all_matches, datecutoff)
    matchup_grid = setup_matchup_grid()
    training_data = prior_data, real_data, matchup_data, surface_data, all_matches, h2h_data    
    print 'Data Grouped'
    
    track_live_scores(full_data, all_matches, (5,10,5,5,10,100), 15000, rankings,
                      t_data = training_data)

main()