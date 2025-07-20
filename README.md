# Audience Rating Prediction

This project focuses on predicting audience ratings for movies using structured metadata and unstructured textual information (e.g., plot summaries, cast, and director names). The project predicts audience response for films and helps understand the effect of various factors on the audience response using machine learning Extreme Gradient Boosting model (XGBoost). 

## Key Objectives:

- Predict the audience rating of a movie based on its metadata, critics' consensus, cast, studio, and other features.
- Explore relationships between audience ratings and other features like runtime, critics' score, genres, etc.
- Integrate textual and categorical data into a predictive machine learning model.

## Methodology:

### Data Source: 
Rotten Tomatoes movie dataset (CSV file).

### Data Cleaning:

- Removed rows with missing values across selected columns.
- Categorized columns as categorical, textual, or numerical.

### Feature Engineering:

- Sentiment analysis using TextBlob on text fields (like critics' consensus).
- Term Frequency–Inverse Document Frequency (TF-IDF) vectorization for textual columns.
- One-hot encoding for categorical variables.

### Modeling:

- Used XGBoost Regressor for audience rating prediction.
- Performance evaluation using MAE and RMSE.

## Model Pipeline:

Raw Data → Cleaning → Feature Transformation → Text Sentiment Analysis + TF-IDF → Column Transformer → Train/Test Split → XGBoost Regressor → Evaluation

- **Text Processing:** TF-IDF for 'movie_info', 'cast', 'directors', etc.
- **Sentiment Scores:** Added polarity scores as new numerical features.
- **Pipeline:** Combined categorical, text, and numerical features using ColumnTransformer.

## Challenges Addressed:

- **Multimodal data:** Combining text, numerical, and categorical inputs efficiently.
- **Missing values:** Removing nulls while maintaining data diversity.
- **High-dimensional text features:** Handled using TF-IDF vectorization and dimensionality control.
- **Model overfitting:** Used appropriate train-test splits and use of a regularized model like XGBoost.

## Results:

The model achieved strong predictive performance:

- **Mean Absolute Error (MAE):** 0.54
- **Root Mean Squared Error (RMSE):** 0.99
- **Mean Squared Error (MSE):** 0.98

This means the model's predictions deviate around 5 % on average, relative to the typical 0–10 audience rating scale — indicating high accuracy.

## Impact:

This model:

- Help studios predict viewer ratings before release.
- Assists streaming platforms in recommending high-performing content.
- Identifies influencing features in audience response like cast, director, and critic sentiments.

## Technology and Tools:

| Category         | Description                                                                   |
|------------------|-------------------------------------------------------------------------------|
| **Language**      | Python                                                                       |
| **Libraries**     | pandas, numpy, matplotlib, seaborn, scikit-learn, textblob, xgboost          |
| **Framework**     | Scikit – learn for preprocessing and pipeline management                     |
| **Model Type**    | Supervised Regression using XGBoost Regressor                                |
| **Dataset Source**| Rotten Tomatoes Movie Dataset                                                 |
