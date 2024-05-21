import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('snowball_data')

amzon_df = pd.read_csv('amazon_product.csv')


amzon_df.head()

amzon_df.drop('id',axis=1)

amzon_df.info()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stem = [stemmer.stem(w) for w in tokens]
    return " ".join(stem)



amzon_df['stemmed_tokens'] = amzon_df.apply(lambda row: tokenize_stem(row['Title'] + ' ' + row['Description']), axis=1)

amzon_df.head(2)


amzon_df['stemmed_tokens']


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidvectorizer = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1,txt2):
    tfid_matrix = tfidvectorizer.fit_transform([txt1,txt2])
    return cosine_similarity(tfid_matrix)[0][1]






def search_product(query):
    stemmed_query = tokenize_stem(query)
    #calcualting cosine similarity between query and stemmed tokens columns
    amzon_df['similarity'] = amzon_df['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_query,x))
    res = amzon_df.sort_values(by=['similarity'],ascending=False).head(10)[['Title','Description','Category']]
    return res





search_product(' PURELL ES8 Professional HEALTHY SOAP Foam Refill, Fresh Scent Fragrance, 1200 mL Soap Refill for PURELL ES8 Touch-Free Dispenser (Pack of 2) - 7777-02 ')
amzon_df['Title'][10]

def evaluate_recommendation_system(ground_truth, recommended):
    # Calculate True Positives, False Positives, and False Negatives
    true_positives = len(set(ground_truth) & set(recommended))
    false_positives = len(recommended) - true_positives
    false_negatives = len(ground_truth) - true_positives

    # Calculate Precision, Recall, and F1-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score



# Example ground truth (relevant products)
ground_truth = amzon_df[ 'Title']


import streamlit as st
st.title("Search Engine and Product Recommendation System ON Am Data")
query = st.text_input("Enter Product Name")
sumbit = st.button('Search')
if sumbit:
    res = search_product(query)
    st.write(res)



if sumbit:
    recommended_products = search_product(query)['Title'].tolist()
    precision, recall, f1_score = evaluate_recommendation_system(ground_truth, recommended_products)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1-score:", f1_score)
    st.write("Recommended Products:", recommended_products)