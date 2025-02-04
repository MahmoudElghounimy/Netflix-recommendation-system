# Netflix Movie Recommendation System

This project builds a simple movie recommendation system using **content-based filtering**. The system recommends movies based on the similarity of the genres of a movie that the user inputs. The recommendations are based on the cosine similarity between the genre features of movies. The model uses the **TF-IDF Vectorizer** to represent genres in a way that allows calculating these similarities.

## Key Steps and Workflow

### 1. **Loading and Preparing the Data**
The dataset is loaded from a CSV file (`netflixData.csv`) using the pandas library. The file is assumed to contain various columns, but we are primarily interested in the following:

- `Title`: The name of the movie.
- `Description`: A brief summary or description of the movie.
- `Content Type`: The type of content (e.g., movie, show, documentary).
- `Genres`: The genre(s) associated with the movie.

```python
data = pd.read_csv("netflixData.csv")
print(data.head())

data = data[["Title", "Description", "Content Type", "Genres"]]
```

### 2. **Data Preprocessing**
The next step involves checking for and removing any missing data:

```python
print(data.isnull().sum())
data = data.dropna()
```

We also perform text cleaning on the `Title` column to ensure that there are no unwanted characters or extra spaces:

```python
import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)  # Remove square brackets and their content
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('\n', '', text)  # Remove newline characters
    text = re.sub('\w*\d\w*', '', text)  # Remove words with numbers
    text = [word for word in text.split(' ') if word not in stopword]  # Remove stopwords
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]  # Stemming
    text = " ".join(text)
    return text

data["Title"] = data["Title"].apply(clean)
```

This cleaning function:
- Converts text to lowercase
- Removes URLs, special characters, punctuation, and digits
- Removes stopwords (common words like "and", "the", etc.)
- Applies stemming to reduce words to their root form

### 3. **Feature Extraction**
We then extract the genre information and convert it into a matrix of TF-IDF features. This is done using the **`TfidfVectorizer`** from `sklearn`:

```python
feature = data["Genres"].tolist()
tfidf = text.TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
```

The **TF-IDF** (Term Frequency-Inverse Document Frequency) is used here to quantify the importance of each word in the context of the genre. The `stop_words="english"` option removes common English words from the text during the transformation process.

### 4. **Calculating Cosine Similarity**
After transforming the genre data into numerical vectors, we calculate the **cosine similarity** between these vectors to determine how similar different movies are to each other based on their genres:

```python
similarity = cosine_similarity(tfidf_matrix)
```

This produces a similarity matrix, where each entry represents the similarity between two movies.

### 5. **Creating a Movie Recommendation Function**
To recommend movies based on the input movie title, we map each movie title to its index in the dataset. Using this index, we fetch the similarity scores for the input movie, sort them by similarity, and return the top 10 most similar movies:

```python
indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

def netFlix_recommendation(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movieindices]
```

### 6. **Testing the Recommendation System**
Finally, you can test the recommendation system by inputting a movie title (e.g., "girlfriend") and getting the top 10 most similar movie recommendations:

```python
print(netFlix_recommendation("girlfriend", similarity))
```

This will return a list of movie titles that are most similar to the input movie based on genre similarity.

