# Movie Recommendations

# Import Libraries
import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.prediction_algorithms import SVD

# import DataFrames
links = pd.read_csv('../ml-latest-small/links.csv')
movies = pd.read_csv('../ml-latest-small/movies.csv')
ratings = pd.read_csv('../ml-latest-small/ratings.csv')

# Preprocessing
ratings.drop('timestamp', axis=1, inplace=True)
links.drop('tmdbId', axis=1, inplace=True)

movies['genreList'] = movies['genres'].map(lambda x: x.split('|'))
movies.drop('genres', axis=1, inplace=True)

# gather list of genres for user convenience
genres = []
for movie in movies.genreList:
    for genre in movie:
        if genre in genres:
            continue
        else:
            genres.append(genre)

# Welcome user, ask if interested in a particular genre
print('------------------------')
print('\n Welcome to ML Movies! \n')
print('Let us find your new favorite movie! \n')

movie_genre = input('Limit your search to a specific genre? (y/[n]) ')

# Gather prefered genre, if applicable
if movie_genre == 'y':
	print('\n AVAILABLE GENRES: ')
	print(genres[:-1], '\n')
	continue_condition = False
	while continue_condition == False:
		genre_filter = input('From the list above, what genre do you prefer? ')
		if genre_filter in genres:
			continue_condition = True
		else:
			print('\n ERROR: check spelling. \n')
	# Adjust movies df to only include selected genre
	movies['to_keep'] = movies.genreList.map(lambda x: genre_filter in x)
	movies.drop(movies[movies['to_keep'] == False].index, inplace=True)
	movies.drop('to_keep', axis=1, inplace=True)

# Allow user to define quantity of movies to rate for algorithm
print('----------')
print(
	''' \n
	To calculate the best movies for you, 
	we will need to hear what you think of some movies
	you have already seen.
	\n
	''')
print('----------')

n = int(input('How many movies would you like to rate for the algorithm? '))

# Gather User Ratings
rating_list = []
while n > 0:
	rating_movie = movies.sample(1)
	print('\n', rating_movie['title'])
	rating = input('How do you rate this movie on a scale of 1-10, press n if you have not seen :\n')
	if rating == 'n':
		continue
	elif int(rating) not in list(range(1,11)):
		print('not a valid entry')
	else:
		rating_one_movie = {'userId':1000,'movieId':rating_movie['movieId'].values[0],'rating':float(rating)/2}
		rating_list.append(rating_one_movie)
		n -= 1

# Make Predictions
reader = Reader()
new_ratings = ratings.append(rating_list, ignore_index=True)
data = Dataset.load_from_df(new_ratings, reader).build_full_trainset()

#Model
print('\n working.... \n')
svd = SVD(n_factors=100, n_epochs=35, lr_all=0.007, reg_all=0.07)
svd.fit(data)

# Gather and sort recommendations
recommendation_list = []
for m_id in movies['movieId']:
	recommendation_list.append((m_id, 2*svd.predict(1000, m_id)[3]))

ranked_recommendations = sorted(recommendation_list, key=lambda x: x[1], reverse=True)

# Deliver Results
print('\n', 'Success!', '\n')
X = int(input('How many movie recommendations would you like to see? '))

i=0
while i<X:
	print('\n recommendation #', i+1)
	print('predicted movie rating: ', np.round(ranked_recommendations[i][1], 2))
	print('Title: ', movies[movies['movieId']==ranked_recommendations[i][0]]['title'])
	i += 1
