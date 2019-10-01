import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn 
from sklearn.decomposition import TruncatedSVD
book=pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
book.columns=['ISBN','bookTitle','bookAuthor','yearOfPublication','publisher','imageUrlS','imageUrlM','imageUrlL']
user = pd.read_csv('BX-Users.csv',sep=';', error_bad_lines=False, encoding = "latin-1")
user.columns=['userID','Location','Age']
rating=pd.read_csv('BX-Book-Ratings.csv',sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns=['userID','ISBN','bookRating']
rating.head()
user.head()
book.head()
combine_book_rating=pd.merge(rating,book, on='ISBN', how="left")
columns= ['yearOfPublication','publisher','bookAuthor','imageUrlS','imageUrlM','imageUrlL']
combine_book_rating=combine_book_rating.drop(columns,axis=1)
combine_book_rating.head()
combine_book_rating=combine_book_rating.dropna(axis=0, subset = ['bookTitle'])
book_ratingCount=combine_book_rating.groupby('bookTitle').agg({'bookRating':'count'}).reset_index()
book_ratingCount=bookratingCount.rename(columns={'bookRating':'totalRatingCount'})
book_ratingCount.head()
rating_with_totalRatingCount=combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle' , how="right")
rating_with_totalRatingCount.head()
pd.set_option('display.float_format',lambda x: '%.3f' %x)
print(book_ratingCount['totalRatingCount'].describe())
print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9,1,.01)))
popularity_threshold = 50
popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
popular_book.head()
combined=popular_book.merge(user, left_on='userID', right_on='userID',how="left")
us_canada_user=combined[combined['Location'].str.contains("usa|canada")]
us_canada_user=user_canada_user_rating.drop('Age',axis=1)
us_canada_user.head()
usa_canada_user_pivot=usa_canada_user.pivot_table(index='bookTitle',columns='userID',values='bookRating').fillna(0)
usa_canada_user_matrix=csr_matrix(usa_canada_user_pivot.values)
from sklearn.neighbors import NearestNeighbors
model_knn=NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(usa_canada_user_matrix)
title=usa_canada_user_pivot.index
listed=list(title)
print(listed)
query_index=np.random.choice(usa_canada_user_pivot.shape[0])
distances, indices=model_knn.kneighbors(usa_canada_user_pivot.iloc[query_index,:].values.reshape(1,-1), n_neighbors=6)
for i in range(0,len(distances.flatten())):
    if i==0:
        print('Recommendations for {0}:\n'.format(usa_canada_user_pivot.index[query_index]))
    else:
        print('{0}:{1}, with distance of {2}:'.format(i,usa_canada_user_pivot.index[indices.flatten()[i]],distances.flatten()[i]))
        
