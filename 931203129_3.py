

"""Importing"""
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import MinMaxScaler
np.random.seed(0)


dataframe_usersong = pd.read_csv(r'C:\Users\eliem\PycharmProjects\MISHAR_3\user_song.csv', header =0, sep=',')
dataframe_test     = pd.read_csv(r'C:\Users\eliem\PycharmProjects\MISHAR_3\test.csv', header =0, sep=',')
matrice_usersong = dataframe_usersong.values
matrice_test     = dataframe_test.values[:, 1:]
user_vector      = np.unique(matrice_usersong[:,0])
song_vector      = np.unique(matrice_usersong[:,1])
df_c = pd.DataFrame(0,columns=song_vector,
                  index=user_vector)
for row in matrice_usersong:
    user = row[0]
    song = row[1]
    rank = row[2]
    df_c.loc[user, song] = rank

def als_algorithm(data, X, Y, num_iterations, regularization_factor):
    for iteration in range(num_iterations):
        # Fix X optimize Y
        for i in range(num_users):
            user_ratings = data[i, :]
            rated_songs = user_ratings.nonzero()[0]
            Y_rated = Y[rated_songs, :]
            X[i, :] = np.linalg.solve(np.dot(Y_rated.T, Y_rated) + regularization_factor * np.eye(rank), np.dot(Y_rated.T, user_ratings[rated_songs]))

        # Fix Y et optimize X
        for j in range(num_songs):
            song_ratings = data[:, j]
            rated_users = song_ratings.nonzero()[0]
            X_rated = X[rated_users, :]
            Y[j, :] = np.linalg.solve(np.dot(X_rated.T, X_rated) + regularization_factor * np.eye(rank), np.dot(X_rated.T, song_ratings[rated_users]))

    return X, Y

data = df_c.values
num_users =user_vector.shape[0]
num_songs = song_vector.shape[0]
rank = 20
num_iterations =500
regularization_factor = 0.01


X = np.zeros((num_users, rank))
Y = np.random.rand(num_songs, rank)


X, Y = als_algorithm(data, X, Y, num_iterations, regularization_factor)

hizouim = pd.DataFrame(X@Y.T, index= user_vector, columns=song_vector)

f4 = 0
for element in matrice_usersong:
    user = element[0]
    song = element[1]
    rank = element[2]
    e = (rank - hizouim.loc[user,song])**2
    f4 +=e

adding = np.zeros((matrice_test.shape[0], 1))
for k in range (matrice_test.shape[0]):
    user  = matrice_test[k][0]
    song = matrice_test[k][1]
    prediction = hizouim.loc[user, song]
    adding [k] = prediction
adding[adding < 0] = 0
column = adding.reshape(-1, 1)

# Concatenate the matrix and the column along the second axis
new_matrix = np.concatenate((matrice_test, column), axis=1)

result_df = pd.DataFrame(new_matrix, columns=['user_id', 'song_id', 'weight'])

result_df.to_csv('931203129_932191265_task4.csv', index=False)