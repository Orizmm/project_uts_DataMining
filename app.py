from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

# Load the data
df = pd.read_csv('recipe_final (1).csv')

# future selection X
scaler = StandardScaler()
X = scaler.fit_transform(df[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# X_Ingriedients
vectorized = TfidfVectorizer()
X_Ingriedients = vectorized.fit_transform(df['ingredients_list'])

# combine X_Ingriedients and X
X_Combine = np.hstack((X, X_Ingriedients.toarray()))

# Fit the model
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(X_Combine)

# function to get recommendations
def get_recommendations(features):
    input_features = scaler.transform([features[:7]])
    input_ingredients = vectorized.transform([features[7]])
    input_combined = np.hstack((input_features, input_ingredients.toarray()))
    distances, indices = model.kneighbors(input_combined, n_neighbors=5)
    recommendations = df.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(3)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']
        features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = get_recommendations(features)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'))

    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)