# app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved TF-IDF model
with open('tfidf_model.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load dataset
data = pd.read_csv('restaurants_data.csv')
df = pd.DataFrame(data)

# Combine relevant text columns
df['textual_features'] = df['Cuisine types'] + ' ' + df['what_people_like_about_us']

# Apply TF-IDF to combined text column
tfidf_matrix = tfidf_vectorizer.transform(df['textual_features'])

def recommend_restaurants(user_food, user_location, top_n=3):
    # Transform user input into TF-IDF vector
    user_input = tfidf_vectorizer.transform([user_food])

    # Filter restaurants based on user location
    filtered_restaurants = df[df['Location'] == user_location]

    # Calculate cosine similarity between user input and restaurant TF-IDF vectors
    similarities = cosine_similarity(user_input, tfidf_matrix[filtered_restaurants.index])

    # Get top N similar restaurants
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    recommendations = filtered_restaurants.iloc[top_indices]

    return recommendations[['Restaurent name', 'Cuisine types', 'Rating', 'Price', 'Distance', 'Location', 'what_people_like_about_us']]

# Streamlit UI
def main():
    st.title("Restaurant Recommendation System")

  

    # Dropdown for cuisine types with search functionality
    user_food = st.selectbox("Enter preferred cuisine type:", df['Cuisine types'].unique())

    # Dropdown for locations
    user_location = st.selectbox("Enter location:", df['Location'].unique())

    if st.button("Recommend"):
        recommendations = recommend_restaurants(user_food, user_location)
        st.subheader("Recommendations based on your preferences:")
        st.table(recommendations)

if __name__ == "__main__":
    main()