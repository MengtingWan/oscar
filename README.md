# Zero-shot predictions for Academy Awards from GPT

This repo contains data and source code for machine learning models to predict academy awards.

Webpage: [Predition Results for 2023](https://mengtingwan.github.io/oscars/oscar2023.html)

Prompts used to query GPT: [prompts](./gpt-prompts-oscars-2023.txt)

# ML models for the previous years

## Data:
Data can be accessed in [./data/](./data), where a series of pre-Oscar awards are included (see descriptions [here](./data/award_code.csv)).

## Model:
A linear regression model is trained on the historical pre-Oscar (used as *features*) and Oscar awards (used as *labels*) and applied to predict the results for year 2020. 

Source code can be accessed in [model.py](./src/model.py).

**Have fun!**
