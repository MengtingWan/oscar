# ML for Academy Awards

This repo contains data and source code for using a machine learning model to predict academy awards.

Webpage: [Predition Results for 2019](https://mengtingwan.github.io/oscar2019.html)

## Data:
Data can be accessed in [./data/](./data), where a series of pre-Oscar awards are included (see descriptions [here](award_code.csv)).
This dataset is collected from [IMDb](https://www.imdb.com/). 

## Model:
A linear regression model is trained on the historical pre-Oscar (used as *features*) and Oscar awards (used as *labels*) and applied to predict the results for year 2019. 

Source code can be accessed in [model.py](model.py).

**Have fun!**