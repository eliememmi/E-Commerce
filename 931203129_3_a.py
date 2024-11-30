"""Importing"""
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import MinMaxScaler
np.random.seed(0)

"""First Mission"""
scaler = MinMaxScaler()

df = pd.read_csv('user_song.csv')

#%%
df['weight'] = df['weight'].clip(lower=0)
r_ag = df['weight'].mean()
df['avg_weight_per_song'] = df.groupby('song_id')['weight'].transform('mean')
df['avg_weight_per_user'] = df.groupby('user_id')['weight'].transform('mean')
df['avg_weight_per_song'] -= r_ag
df['avg_weight_per_user'] -= r_ag
df['prevision']=r_ag + df['avg_weight_per_user']+df['avg_weight_per_song']
df['prevision'] = df['prevision'].clip(lower=0)

song_and_avg_weight_per_song= df[['song_id', 'avg_weight_per_song']].drop_duplicates()
user_and_avg_weight_per_user= df[['user_id', 'avg_weight_per_user']].drop_duplicates()



df_test= pd.read_csv('test.csv')
df_test=df_test.drop(df_test.columns[0], axis=1)

df_test_reponce = pd.merge(df_test, song_and_avg_weight_per_song, on='song_id')
df_test_reponce = pd.merge(df_test_reponce, user_and_avg_weight_per_user, on='user_id')

df_test_reponce['final_prevision'] = r_ag +df_test_reponce["avg_weight_per_song"] +df_test_reponce["avg_weight_per_user"]
df_test_reponce['final_prevision'] = df_test_reponce['final_prevision'].clip(lower=0)
test_df = pd.read_csv('test.csv')

pairs = [(row.user_id, row.song_id) for row in test_df.itertuples()]

values = []

# Accéder aux valeurs correspondantes pour chaque paire
for pair in pairs:
    user_id, song_id = pair
    value = df_test_reponce.loc[(df_test_reponce['user_id'] == user_id) & (df_test_reponce['song_id'] == song_id)]['final_prevision'].values[0]
    values.append((user_id,song_id,value))


result_df = pd.DataFrame(values, columns=['user_id', 'song_id', 'weight'])

# Sauvegarder le DataFrame dans un fichier CSV
result_df.to_csv('931203129_932191265_task1.csv', index=False)

f1 = ((df['prevision'] - r_ag - df['avg_weight_per_user'] - df['avg_weight_per_song']) ** 2).sum()


"""Second Mission"""
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
num_iterations =400
regularization_factor = 0.01


X = np.zeros((num_users, rank))
Y = np.random.rand(num_songs, rank)


X, Y = als_algorithm(data, X, Y, num_iterations, regularization_factor)

hizouim = pd.DataFrame(X@Y.T, index= user_vector, columns=song_vector)

f2 = 0
for element in matrice_usersong:
    user = element[0]
    song = element[1]
    rank = element[2]
    e = (rank - hizouim.loc[user,song])**2
    f2 +=e


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

result_df.to_csv('931203129_932191265_task2.csv', index=False)


"""Third Mission"""
scaler = MinMaxScaler()

df = pd.read_csv('user_song.csv')

# Créer la matrice en pivotant la DataFrame
matrix = df.pivot(index='user_id', columns='song_id', values='weight')
matrix = matrix.fillna(0)
user_ids = matrix.index

song_ids = matrix.columns

matrix = pd.DataFrame(scaler.fit_transform(matrix), columns=matrix.columns, index=matrix.index)
matrix_values = matrix.values




#  SVD
U, S, VT = np.linalg.svd(matrix_values)



k = 20
U = U[:, :k]
S = np.diag(S[:k])
VT = VT[:k, :]

predictions = U.dot(S).dot(VT)
predictions_df = pd.DataFrame(predictions, index=user_ids, columns=song_ids)
predictions_df = predictions_df.clip(lower=0)
predictions_df = pd.DataFrame(scaler.inverse_transform(predictions_df), columns=predictions_df.columns, index=predictions_df.index)
predictions= predictions_df.values

mask = matrix.ne(0)
squared_differences = np.where(mask, (matrix - predictions) ** 2, 0)

squared_diff_df = pd.DataFrame(squared_differences, index=matrix.index, columns=matrix.columns)

f3 = squared_diff_df.values.sum()

test_df = pd.read_csv('test.csv')

# Créer une liste de tuples (user_id, song_id)
pairs = [(row.user_id, row.song_id) for row in test_df.itertuples()]

values = []

for pair in pairs:
    user_id, song_id = pair
    value = predictions_df.loc[user_id,song_id]
    values.append((user_id,song_id,value))


result_df = pd.DataFrame(values, columns=['user_id', 'song_id', 'weight'])

result_df.to_csv('931203129_932191265_task3.csv', index=False)