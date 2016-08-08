#--------------------------------------------------------------------------------------------
# Project: Scraping_Live_Tennis_Scores.py
#
# Description: This script scrapes real-time tennis scores on the ATP World Tour, and
# organizes the data in a fashion that can be imported into our predictive model to
# calculate a real-time probability of each player winning the match.
#--------------------------------------------------------------------------------------------

# Import Packages

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import urllib2
import time
import requests
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

#--------------------------------------------------------------------------------------------
# This function is a wrapper function that reads in real-time information and associates 
# player information with an ID
#--------------------------------------------------------------------------------------------

def run_scraper(seconds):
    
    # Configure Player Data
    players = pd.read_csv('tennis_atp-master/atp_players.csv',header=None)
    players.columns = ['Player ID','First','Last','L/R','DOB','Country']
    players.index = players['Last'] + np.array([' ' + str(x)[0] + '.' if str(x) != 'nan' else '' for x in players['First']])
    
    base_link = 'http://www.scoreboard.com/tennis/'
    driver = webdriver.Firefox()
    database = []

    count = 0
    driver.get(base_link)

    start = time.time()
    end = time.time()
    
    data_store = dict()

    while (end-start) < seconds:
        try:
            element = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "fs"))
            )
        except selenium.common.exceptions.TimeoutException:
            print str(count) + '- Time out at: ' +                   str(time.time() - start) + ' Seconds'
            break
        print str(count) + ' - Scores Found'
        html = driver.page_source
        database = database + parse_html(html,count)
        
        new_data = parse_database(database, players)
        
        for p1, p2 in new_data:
            frame = new_data[(p1,p2)]
            frame = pd.DataFrame(frame, columns = ['Player 1', 'Player 1 ID',
                                                   'Player 2', 'Player 2 ID',
                                                   'Set Score', 'Game Score', 
                                                    'Point Score', 'Serving?'])
            if (p1,p2) not in data_store.keys():
                frame.to_csv('Scraped Matches/' + str(datetime.today().strftime('%Y-%m-%d')) + 
                             '_' + str(p1) + '_' + str(p2) + '_live.csv')
                data_store[(p1,p2)] = frame
            else:
                new_frame = frame[len(data_store[(p1,p2)]):]
                new_frame = pd.DataFrame(new_frame, columns = ['Player 1', 'Player 1 ID',
                                                               'Player 2', 'Player 2 ID',
                                                               'Set Score', 'Game Score', 
                                                               'Point Score', 'Serving?'])
                new_frame.to_csv('Scraped Matches/' + 
                                 str(datetime.today().strftime('%Y-%m-%d')) + 
                                 '_' + str(p1) + '_' + str(p2) + '_live.csv')
                data_store[(p1,p2)] = frame
            
            check = pd.DataFrame([[0]])
            check.to_csv('Scraped Matches/' + str(datetime.today().strftime('%Y-%m-%d')) + 
                         '_' + str(p1) + '_' + str(p2) + '_updated.csv')
                
        for p1, p2 in data_store:
            frame = data_store[(p1,p2)]
            frame.to_csv('Scraped Matches/' + 
                          str(datetime.today().strftime('%Y-%m-%d')) + 
                          '_' + str(p1) + '_' + str(p2) + '_archive.csv')
            
        time.sleep(15)
        while sum([1 - pd.read_csv('Scraped Matches/' + 
                                str(datetime.today().strftime('%Y-%m-%d')) + 
                                '_' + str(p1) + '_' + str(p2) + '_updated.csv', 
                                index_col = 0).values[0][0] for p1, p2 in new_data]):
            print 'Waiting to Update'
            time.sleep(5)
        
        end = time.time()
        count = count + 1

    driver.quit()


#--------------------------------------------------------------------------------------------
# This function parses the HTML that is on http://scoreboard.com/tennis, and extracts
# in raw form:
# (1) All matches that are live on the ATP World Tour
# (2) The players, the score, and who's serving for each match
#--------------------------------------------------------------------------------------------

def parse_html(html, count):
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('div',{'id': 'fs'})
    body = table.find('tbody')
    rows = body.find_all('tr')
    filtered_rows = []

    for row in rows:
        if 'live' in ' '.join(row.attrs['class']):
            filtered_rows.append(row)

    all_player_scores = []
    for frow in filtered_rows:
        cells = frow.find_all('td')
        scores = [count]
        for cell in cells:
            if 'team' in ' '.join(cell.attrs['class']):
                scores.append(cell.find('span').text)
            elif 'serve' in ' '.join(cell.attrs['class']):
                if len(cell.findAll('span')) == 0:
                    scores.append('Returning')
                else:
                    scores.append('Serving')
            elif 'part' in ' '.join(cell.attrs['class']):
                if cell.text != u'\xa0':
                    scores.append(cell.text)
        all_player_scores.append(scores)
        
    return all_player_scores


#--------------------------------------------------------------------------------------------
# This function cleans the raw data that was extracted from the HTML. It also determines
# whether the score has been updated, and divides all data into an archive CSV and a 
# live CSV so the live scoreboard can be updated with new probabilities.
#--------------------------------------------------------------------------------------------

def parse_database(database, players):
    new_data = dict()
    player_matchup = dict()
    #match_num = dict()
    #count = 0
    for x,y in zip(database[0::2],database[1::2]):
        player1 = str(x[2]).split(' (')[0]
        player2 = str(y[2]).split(' (')[0]
        
        if str(player1) == 'SET':
            player1 = player_matchup[player2]
        if str(player2) == 'SET':
            player2 = player_matchup[player1]
            
        player_matchup[player1] = player2
        player_matchup[player2] = player1
            
        p1 = players.loc[player1]['Player ID']
        p2 = players.loc[player2]['Player ID']
        
        if type(p1) != np.int64:
            p1 = p1.values[-1]
        if type(p2) != np.int64:
            p2 = p2.values[-1]
        
        print p1, p2
        
        point_score = str(x[-1]) + '-' + str(y[-1])
        game_score = str(x[-2][0]) + '-' + str(y[-2][0])
        serving = x[1]
        if len(x) > 5:
            prior_sets = zip(x[3:-2],y[3:-2])
            w1, w2 = 0,0
            for a,b in prior_sets:
                if int(a) > int(b):
                    w1 = w1 + 1
                elif int(a) < int(b):
                    w2 = w2 + 1
            set_score = str(w1) + '-' + str(w2)
        else:
            set_score = '0-0'
            
        if (p1,p2) not in new_data:
            new_data[(p1,p2)] = []
            new_data[(p1,p2)].append([player1, p1, player2, p2, set_score, 
                                    game_score,point_score, serving])
            #count = count + 1
        else:
            row =  [player1, p1, player2, p2, set_score, game_score,
                    point_score, serving]
            if row != new_data[(p1,p2)][-1]:
                new_data[(p1,p2)].append([player1, p1, player2, p2, 
                                          set_score, game_score, point_score, serving])           
        
    return new_data


#--------------------------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------------------------

def main():
    run_scraper(15000)
    
main()




