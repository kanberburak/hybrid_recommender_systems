#############################################
# PROJE: Hybrid Recommender System Çözüm
#############################################

#############################################
# Adım 1: Verinin Hazırlanması
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', 5)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('movie.csv')
    rating = pd.read_csv('rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

random_user = 108170
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# weighted_rating'i 4'ten büyük olanları getirelim:
recommendation_df[recommendation_df["weighted_rating"] > 4]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].sort_values("weighted_rating", ascending=False)[0:5]

movie = pd.read_csv('movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]]).index


#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
# 5 öneri user-based 5 öneri item-based olacak şekilde 10 öneri yapınız.

# İpucu:

# user = 108170

# movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
# rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#
# Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınması:
# movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
#     sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
#

user = 108170

movie = pd.read_csv('movie.csv')
rating = pd.read_csv('rating.csv')

# Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınması:
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
    sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]


def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index

# Nihai olarak önerilecek filmler:

# 0                           Foxfire (1996)
# 1               Man from Earth, The (2007)
# 2                            Primer (2004)
# 3            Pokémon the Movie 2000 (2000)
# 4                            Selena (1997)
# 5                My Science Project (1985)
# 6                      Mediterraneo (1991)
# 7          Old Man and the Sea, The (1958)
# 8    National Lampoon's Senior Trip (1995)
# 9                     Clockwatchers (1997)