#--------------------------------------------------------------------------------------------
# Project: Tennis_PBP_Project.py
#
# Description: This script processes Jeff Sackmann's point-by-point data into a set of 
# statistics that measure the chance of winning a match given all game situations.
#--------------------------------------------------------------------------------------------

# Import Packages

import numpy as np
import pandas as pd

#--------------------------------------------------------------------------------------------
# These dictionaries and functions help map game states to easily comprehensible game
# situations (in both regular game and tiebreak scenarios)
#--------------------------------------------------------------------------------------------

all_game_states = {
    (0,0,1): 'Start of Game',
    (0,0,2): 'Start of Game',
    (15,0,1): 'Server 15-0',
    (0,15,2): 'Server 15-0',
    (0,15,1): 'Server 0-15',
    (15,0,2): 'Server 0-15',
    (15,15,1): 'Server 15-15',
    (15,15,2): 'Server 15-15',
    (30,0,1): 'Server 30-0',
    (0,30,2): 'Server 30-0',
    (0,30,1): 'Server 0-30',
    (30,0,2): 'Server 0-30',
    (30,15,1): 'Server 30-15',
    (15,30,2): 'Server 30-15',
    (15,30,1): 'Server 15-30',
    (30,15,2): 'Server 15-30',
    (30,30,1): 'Server 30-30',
    (30,30,2): 'Server 30-30',
    (40,0,1): 'Server 40-0',
    (0,40,2): 'Server 40-0',
    (40,15,1): 'Server 40-15',
    (15,40,2): 'Server 40-15',
    (40,30,1): 'Server 40-30',
    (30,40,2): 'Server 40-30',
    (0,40,1): 'Server 0-40',
    (40,0,2): 'Server 0-40',
    (15,40,1): 'Server 15-40',
    (40,15,2): 'Server 15-40',
    (30,40,1): 'Server 30-40',
    (40,30,2): 'Server 30-40',
    (40,40,1): 'Server 40-40',
    (40,40,2): 'Server 40-40',
    (45,40,1): 'Server AD-40',
    (40,45,2): 'Server AD-40',
    (40,45,1): 'Server 40-AD',
    (45,40,2): 'Server 40-AD'
}


def tiebreak_state(state):
    p1,p2,server = state
    if p1 >= 6 and p2 >=6:
        if p1 > p2 and server == 1:
            return 'Server 7-6'
        elif p1 > p2 and server == 2:
            return 'Server 6-7'
        elif p1 < p2 and server == 1:
            return 'Server 6-7'
        elif p1 < p2 and server == 2:
            return 'Server 7-6'
        else:
            return 'Server 6-6'
    elif server == 1:
        return 'Server ' + str(p1)+'-'+str(p2)
    elif server == 2:
        return 'Server ' + str(p2)+'-'+str(p1)
    

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

def tiebreak_coordinates(game_state):
    if game_state not in coordinate_states.keys():
        print game_state#[7:]
        g1, g2 = game_state[7:].split('-')
        #print g1, g2
        return (int(g1)+int(g2), int(g1)-int(g2))
    else:
        return coordinate_states[game_state]


#--------------------------------------------------------------------------------------------
# This function adds a new row to the point by point dataset. It does certain edits to
# the raw data to ensure consistency, (e.g. converting game points to changing the game
# score, set score, serve/return situations etc). It also uses specific rules on the ATP
# World Tour to make necessary adjustments (e.g. Grand Slams are best-of-5 matches, and 
# fifth set tiebreakers only apply to the US Open.)
#--------------------------------------------------------------------------------------------

def getNewRow(player1, player2, winner, p1set, p2set, p1game, p2game, p1score, p2score, server, pts, char, tourney):
    flag = 0
    if char == '/':
        if server == 1:
            server = 2
        elif server == 2:
            server = 1
        flag = 1
    elif char == ';':
        if p1score == 45:
            p1game = p1game + 1
        elif p2score == 45:
            p2game = p2game + 1
        elif p1score == 0:
            p1game = p1game + 1
        elif p2score == 0:
            p2game = p2game + 1
        p1score = 0
        p2score = 0
        if server == 1:
            server = 2
        elif server == 2:
            server = 1
        flag = 1
        
    elif char == '.':
        if p1game > p2game:
            p1set = p1set + 1
        elif p1game < p2game:
            p2set = p2set + 1
        else:
            #print player1, player2, str(p1game) + '-' + str(p2game), str(p1score) + '-' + str(p2score)
            if p1score > p2score:
                p1set = p1set + 1
            elif p1score < p2score:
                p2set = p2set + 1
            if (p1score + p2score) % 4 >= 2:
                if server == 1:
                    server = 2
                elif server == 2:
                    server = 1
        p1game = 0
        p2game = 0
        p1score = 0
        p2score = 0
        if server == 1:
            server = 2
        elif server == 2:
            server = 1  
        flag = 1
    elif p1game == 6 and p2game == 6 and ((tourney not in ["Men'sAustralianOpen","Men'sAustralianOpen.",
                                                           'MensAustralianOpen', 'MensAustralianOpen.html',
                                                           "Men'sFrenchOpen", "Men'sFrenchOpen.",'MensFrenchOpen',
                                                           'MensFrenchOpen.html', "Gentlemen'sWimbledonSingles",
                                                           "Gentlemen'sWimbledonSingles.","Gentlemen'sWimbledonSingles.html",
                                                           #"Men'sUSOpen", "Men'sUSOpen.","Men'sUSOpen.html",
                                                           'DavisCup','DavisCup-Live', 'DavisCup.html', 'DavisCupLive']) \
     or p1set != 2 or p2set != 2):
        if server == 1:
            if char in ['S','A']:
                p1score = p1score + 1
            elif char in ['R','D']:
                p2score = p2score + 1
        elif server == 2:
            if char in ['S','A']:
                p2score = p2score + 1
            elif char in ['R','D']:
                p1score = p1score + 1    
    elif server == 1:
        if char in ['S','A']:
            if p2score != 45:                    
                p1score = pts[(pts.index(p1score)+1) % len(pts)]
            else:
                p2score = 40
        elif char in ['R','D']:
            if p1score != 45:
                p2score = pts[(pts.index(p2score)+1) % len(pts)]
            else:
                p1score = 40
    elif server == 2:
        if char in ['S','A']:
            if p1score != 45:                    
                p2score = pts[(pts.index(p2score)+1) % len(pts)]
            else:
                p1score = 40
        elif char in ['R','D']:
            if p2score != 45:
                p1score = pts[(pts.index(p1score)+1) % len(pts)]
            else:
                p2score = 40
                
    return [player1, player2, winner, p1set, p2set, p1game, p2game, p1score, p2score, server, pts, char, flag]            


#--------------------------------------------------------------------------------------------
# This function extracts the necessary information from each row of the raw_data (which
# represents a tennis match) and calls the function above, to split a match into its point
# components. This can be adjusted to include best-of-3 or best-of-5 matches.
#--------------------------------------------------------------------------------------------

def tourDataSet(raw_data, three_set=True, five_set=False):
    master_data = []
    for i,row in raw_data.iterrows():
        five_set_tourneys = ["Men'sAustralianOpen","Men'sAustralianOpen.",'MensAustralianOpen', 'MensAustralianOpen.html',
                     "Men'sFrenchOpen", "Men'sFrenchOpen.",'MensFrenchOpen', 'MensFrenchOpen.html',
                     "Gentlemen'sWimbledonSingles", "Gentlemen'sWimbledonSingles.","Gentlemen'sWimbledonSingles.html",
                     "Men'sUSOpen", "Men'sUSOpen.","Men'sUSOpen.html"]
        only_both = ['DavisCup','DavisCup-Live', 'DavisCup.html', 'DavisCupLive',
                     "Men'sAustralianOpenWildcardPlayoff"]
        
        pts = [0,15,30,40,45]
        player1 = row['server1']
        player2 = row['server2']
        winner = row['winner']
        score = row['score']
        year = row['date'].split(' ')[-1]
        tourney = row['tny_name']
        server = 1
        p1set,p2set,p1game,p2game,p1score,p2score = 0,0,0,0,0,0
        
        eligible = True
        if three_set == True and five_set == False:
            if tourney in five_set_tourneys or tourney in only_both:
                eligible = False
        if three_set == False and five_set == True:
            if tourney not in five_set_tourneys or tourney in only_both:
                eligible = False
        
        if eligible:
            master_data.append([player1, player2, winner, p1set, p2set, p1game, p2game, p1score, p2score,
                            server, pts, '', score, year, tourney, i])
            for char in row['pbp']:
                newrow = getNewRow(player1, player2, winner, p1set, p2set, 
                                   p1game, p2game, p1score, p2score, server, pts, char, tourney)
                flag = newrow[-1]
                if flag == 0:
                    master_data.append(newrow[:-1] + [score, year, tourney, i])
                elif flag == 1:
                    master_data[-1] = newrow[:-1] + [score, year, tourney, i]
                player1, player2, winner, p1set, p2set, p1game, p2game, p1score, p2score, server, pts, char = newrow[:-1]
            master_data.pop()
        
    return pd.DataFrame(master_data,columns=['Player 1', 'Player 2', 'Winner', 'p1Set', 'p2Set', 'p1Game', 'p2Game', 
                                             'p1Score', 'p2Score', 'Server','Points','Result','Score',
                                             'Year','Tourney','MatchNum'])    


#--------------------------------------------------------------------------------------------
# Step 1: This function extracts initial data from the CSV files compiled by Jeff Sackmann
#--------------------------------------------------------------------------------------------

def compile_initial_point_data(challenger_15_idx, challenger_archive_idx):
    pbp_2015 = pd.read_csv('tennis_pointbypoint-master/pbp_matches_atp_main_current.csv')
    pbp_archive = pd.read_csv('tennis_pointbypoint-master/pbp_matches_atp_main_archive.csv')
    pbp_2015_q = pd.read_csv('tennis_pointbypoint-master/pbp_matches_atp_qual_current.csv')
    pbp_archive_q = pd.read_csv('tennis_pointbypoint-master/pbp_matches_atp_qual_archive.csv')
    pbp_2015_ch = pd.read_csv('tennis_pointbypoint-master/pbp_matches_ch_main_current.csv')
    pbp_archive_ch = pd.read_csv('tennis_pointbypoint-master/pbp_matches_ch_main_archive.csv')
    pbp_2015_q_ch = pd.read_csv('tennis_pointbypoint-master/pbp_matches_ch_qual_current.csv')
    pbp_archive_q_ch = pd.read_csv('tennis_pointbypoint-master/pbp_matches_ch_qual_archive.csv')


    data_15 = tourDataSet(pbp_2015)
    data_archive = tourDataSet(pbp_archive.loc[[x not in [2165,2381,2543] 
                                                for x in pbp_archive.index.values]],three_set=True,five_set=False)

    data_15_ch = tourDataSet(pbp_2015_ch[[x not in challenger_15_idx for x in pbp_2015_ch.index.values]])
    data_archive_ch = tourDataSet(pbp_archive_ch[[x not in challenger_archive_idx for x in pbp_archive_ch.index.values]])

    all_point_data = pd.concat([data_15,data_archive, data_15_ch,data_archive_ch],axis=0)
    
    return all_point_data


#--------------------------------------------------------------------------------------------
# Step 2: This function adds additional columns to point data dataset for ease of analysis
# and interpretation
#--------------------------------------------------------------------------------------------

def add_columns_to_point_data(all_point_data):
    states = [(x,y,z) for x,y,z in zip(all_point_data['p1Score'],all_point_data['p2Score'],all_point_data['Server'])]
    all_point_data['Game State'] = [all_game_states[x] if x in all_game_states.keys() else 'Tiebreak' for x in states]
    all_point_data['Server Winner'] = [x == y for x,y in zip(all_point_data['Winner'],all_point_data['Server'])]
    all_point_data['Game Score Server View'] = [str(x)+'-'+str(y) if z == 1 else str(y)+'-'+str(x)
                                               for x,y,z in zip(all_point_data['p1Game'],all_point_data['p2Game'], 
                                                                all_point_data['Server'])]
    all_point_data['Set Score Server View'] = [str(x)+'-'+str(y) if z == 1 else str(y)+'-'+str(x)
                                               for x,y,z in zip(all_point_data['p1Set'],all_point_data['p2Set'], 
                                                                all_point_data['Server'])]

    all_point_data['Game State New'] = [tiebreak_state(x) if y == '6-6' else z 
                                        for x,y,z in zip(states,all_point_data['Game Score Server View'],
                                                         all_point_data['Game State'])]
    return all_point_data

#--------------------------------------------------------------------------------------------
# Step 2.5: This function identifies bad scorelines and removes it from our analysis. While
# Jeff Sackmann's data is robust, there are still a few situations where the point-by-point
# data does not lead to a sensible scores and has to be removed.
#--------------------------------------------------------------------------------------------

def identify_bad_states(all_point_data):
    bad_states = all_point_data[((all_point_data['Game State'] == 'Tiebreak') & 
                                (all_point_data['Game Score Server View'] != '6-6')) |
                                ([x in ['4-6', '0-6', '6-4', '4-7', '8-4', '4-8','7-6', '6-3',
                                        '7-4', '5-7', '7-5', '6-7', '2-6', '1-6'] 
                                  for x in all_point_data['Game Score Server View']]) |
                                ([x in ['2-0','0-2'] for x in all_point_data['Set Score Server View']])]
    challenger_tour_15_idx = []
    challenger_tour_15_q_idx = []
    challenger_tour_archive_idx = []
    challenger_tour_archive_q_idx = []

    for i,row in bad_states.iterrows():
        #print 'Challenger' in row['Year']
        if 'Challenger' in row['Tourney'] and 'Qualifying' not in row['Tourney'] and '15' in row['Year']:
            challenger_tour_15_idx.append(row['MatchNum'])
        elif 'Challenger' in row['Tourney'] and 'Qualifying' not in row['Tourney'] and '15' not in row['Year']:
            challenger_tour_archive_idx.append(row['MatchNum'])

    challenger_15_idx = np.unique(challenger_tour_15_idx)
    challenger_archive_idx = np.unique(challenger_tour_archive_idx)
    
    return challenger_15_idx, challenger_archive_idx


#--------------------------------------------------------------------------------------------
# Step 3: This function summarizes the point data dataset, to get the match win % from
# all game situations.
#--------------------------------------------------------------------------------------------

def summarize_results(all_point_data):

    score_state_all = pd.DataFrame(all_point_data.groupby(['Set Score Server View','Game Score Server View', 
                                         'Game State New']).count()['Player 1'])
    score_state_sum = pd.DataFrame(all_point_data.groupby(['Set Score Server View','Game Score Server View', 
                                         'Game State New', 'Server Winner']).count()['Player 1'])

    all_results = []
    for set_score, game_score, game_state in score_state_all.index.values:
        #print set_score, game_score, game_state

        try:
            x, y = tiebreak_coordinates(game_state)
            if game_score == '6-6':
                if x % 4 == 1 or x % 4 == 2:
                    all_results.append((set_score,game_score,'Serving',game_state[7:]+ ' (R)', 
                                1.0*float(score_state_sum.loc[(set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(set_score,game_score,game_state)]),
                                int(score_state_all.loc[(set_score,game_score,game_state)]),
                                x, y))
                else:
                    all_results.append((set_score,game_score,'Serving',game_state[7:]+ ' (S)', 
                                1.0*float(score_state_sum.loc[(set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(set_score,game_score,game_state)]),
                                int(score_state_all.loc[(set_score,game_score,game_state)]),
                                x, y))
            elif 'Server' in game_state:
                all_results.append((set_score,game_score,'Serving',game_state[7:], 
                                1.0*float(score_state_sum.loc[(set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(set_score,game_score,game_state)]),
                                int(score_state_all.loc[(set_score,game_score,game_state)]),
                                x, y))
            else:
                all_results.append((set_score,game_score,'Serving',game_state, 
                                1.0*float(score_state_sum.loc[(set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(set_score,game_score,game_state)]),
                                int(score_state_all.loc[(set_score,game_score,game_state)]),
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

            all_results.append((newsscore,newgscore,'Returning',new_score, 
                                1.0 - 1.0*float(score_state_sum.loc[(set_score, game_score, game_state, True)])/
                                float(score_state_all.loc[(set_score,game_score,game_state)]),
                                int(score_state_all.loc[(set_score,game_score,game_state)]),
                                x, -1*y))
        except KeyError:
            x, y = tiebreak_coordinates(game_state)
            if game_score == '6-6':
                if x % 4 == 1 or x % 4 == 2:
                    all_results.append((set_score, game_score, 'Serving',game_state[7:]+ ' (R)', 0.,
                                int(score_state_all.loc[(set_score,game_score,game_state)]), x, y))
                else:
                    all_results.append((set_score, game_score, 'Serving',game_state[7:]+ ' (S)', 0.,
                                int(score_state_all.loc[(set_score,game_score,game_state)]), x, y))
            elif 'Server' in game_state:
                all_results.append((set_score, game_score, 'Serving',game_state[7:], 0.,
                                int(score_state_all.loc[(set_score,game_score,game_state)]), x, y))
            else:
                all_results.append((set_score, game_score, 'Serving',game_state, 0.,
                                int(score_state_all.loc[(set_score,game_score,game_state)]), x, y))

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

            all_results.append((newsscore, newgscore, 'Returning',new_score, 1.,
                                int(score_state_all.loc[(set_score,game_score,game_state)]), x, -1*y))

#--------------------------------------------------------------------------------------------
# Step 4: This function organizes the grouped results to be exported to a CSV file
# for further analysis in our predictive model.
#--------------------------------------------------------------------------------------------

def send_to_csv(all_results):
    sum_data = pd.DataFrame(all_results,columns=['Set Score','Game Score','Serving at Start of Game?','Point Score','% Win',
                                                 'Number of Instances','Points Elapsed','Y'])
    sum_data['Stdev'] = [((x*(1-x))/y)**0.5 for x,y in zip(sum_data['% Win'],sum_data['Number of Instances'])]
    sum_data.to_csv('Match_Probs_best5_50_50.csv')

#--------------------------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------------------------

def main():
    challenger_15_idx, challenger_archive_idx = [], []
    all_point_data = compile_initial_point_data(challenger_15_idx, challenger_archive_idx)
    all_point_data = add_columns_to_point_data(all_point_data)
    
    # Check for Bad State
    challenger_15_idx, challenger_archive_idx = identify_bad_states(all_point_data)
    while len(challenger_15_idx) > 0 or len(challenger_archive_idx) > 0:
        all_point_data = compile_initial_point_data(challenger_15_idx, challenger_archive_idx)
        all_point_data = add_columns_to_point_data(all_point_data)
        challenger_15_idx, challenger_archive_idx = identify_bad_states(all_point_data)
        print str(len(challenger_15_idx) + len(challenger_archive_idx)) + ' Instances found to have Bad Scores'
    
    all_results = summarize_results(all_point_data)
    
    # Complete Operations
    send_to_csv(all_results)
    all_point_data.to_csv('All Point Data.csv')

main()