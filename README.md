# 04-Tennis-Point-by-Point-Project
Live Point by Point Prediction of ATP Tennis Win Probability

![alt tag](https://github.com/haroldli93/04-Tennis-Point-by-Point-Project/blob/master/Scraped Matches/2016-08-01_106329_105065_live.png)

This project is an attempt to <b>model win odds of all ATP World Tour matches in real time</b>. It is an extension of a project I completed last year on predicting match outcomes in grand slam tournaments. Significant improvements I made to this model include the following:

1. While my previous model was Elo-based, it <b>lacked Bayesian elements</b>. The new model uses a probability-based model as a prior distribution of our predictions, and updates them based on match-specific information (i.e. surface, head-to-head record, player's recent form, rankings, etc.) to create a posterior distribution, which is our final prediction.

2. It predicts matches in <b>real-time</b>. Previously, I was only able to predict matches before it began, as I only extracted <b>match-level</b> data. Now that I have extracted <b>point-level</b> data, the new model can make predictions at any game situation.

The <b>Tennis PBP Project</b> scripts processes the point-by-point CSV files that Jeff Sackmann's Github has gracefully shared, and calculates a player's win percentage in all game situations. The <b>Bayesian Elements in PBP</b> scripts encompass the actual predictive model that outputs and visualizes the probability of a player winning a match. The <b>Scraping Live Tennis Scores</b> scripts scrape live tennis matches in real-time from [http://www.scoreboard.com/tennis](http://www.scoreboard.com/tennis) and processes the data into reliable match data that the predictive model can use.

The next steps of my project (if I have more time) will be to continuously feed tennis data into our database so that the model can continue to learn new match data and improve its predictions. Also more than happy to get feedback on this project, especially on other modeling approaches or factors that you think I can incorporate into my forecasts. Thanks!
