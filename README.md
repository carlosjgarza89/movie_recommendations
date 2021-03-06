# Movie Recommendations
Author: Carlos Garza

## Overview
This notebook details the creation and deployment of a recommendation system for movies. Utilizing the CRISP-DM framework, singular value decomposition, and various model tuning techniques, the backend of a recommendation application that takes input from a user regarding personal taste in genre and films previously watched and outputs a user defined quantity of movie recommendations was created.

## Business Problem
A new streaming company called ML Movies wants to implement an active movie recommendation system for its users that takes user input to calculate a currated list of movie recommendations. Using a list of avaiable films that have previously been rated by other users, develop a recommendation algorithm that generates curated movie recommendations.

## Data
The data for this project is sourced from [MovieLens](https://grouplens.org/datasets/movielens/latest/). The data describes 100936 ratings across 9742 movies made by 610 users between March 29, 1996 and September 26, 2018. 

The data is organized in the files links.csv, movies.csv, ratings.csv, and tags.csv. All of the files as well as a README.txt can be found in the [ml-latest-small](ml-latest-small) folder.

## Methods
This project makes use of the Surprise library as its implementation of singular value decompositon uses a modified algorithm created by Simon Funk that ignores items that have not been rated by users. 

This will allow users to provide as much or as little input as they would like.

Baseline prediction models were creayed using singular value decomposition and various versions of K Nearest Neighbor algorithms to find the most accurate starting model. Singular value decomposition performed best, and the algorithms hyperparameters were tuned using a grid search to improve performance. 

Once hyperparameters were optimized, the model was used in the development of a .py file that can be run from a terminal. The file acts as the first draft of an interface for a user that takes input and returns movie recommendations.

## Results
The algorithm is capable of filtering results by genre, working with a much or as little user input as the user prefers, and returns a user defined quantity of top recommendations.

![algorithm_output](images/algorithm_output.png)

## Conclusions
- Using singular value decomposition, the model created has a RMSE of 0.85.
- The .py file using the SVD model provides the flexibility to filter by genre and is intuitive enough for a non technical audience.
- The algorithm consistantly finds movies to recommend that it predicts the user will rate at least 4/5.

## Future Work
In the future, this project can be imporved and expanded in a number of ways.
- Create GUI for a user to interact with the algorithm
- Code for possibility to select more than one genre
- Create more robust code that is more flexible with user input
- Create a way to save recommendations or save and update a user profile

## For More Information
For more detailed information, please review my full analysis in [my Jupyter Notebook](./master_notebook.ipynb) or my [presentation](./project_presentation.pdf).

For any additional questions, please contact me by email: **carlosjgarza89@gmail.com**

## Citation
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1???19:19. https://doi.org/10.1145/2827872

## Repository Structure
```
????????? README.md                           <- The top-level README for reviewers of this project
????????? master_notebook.ipynb               <- Narrative documentation of analysis in Jupyter notebook
????????? master_notebook.pdf                 <- PDF version of master notebook
????????? project_presentation.pdf            <- PDF version of project presentation
????????? ml-latest-small                     <- MovieLens Data
????????? images                              <- Both sourced externally and generated from code
????????? deployment                          <- Example scripts for a deployment app
```
