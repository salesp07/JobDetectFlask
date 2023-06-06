from flask import Blueprint, request, jsonify
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
# To get stop words.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLTK's lemmatizer, stop words and punctuation
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
tokenizer = RegexpTokenizer(r'\w+')

# Define a function to preprocess the text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text (Split sentence into array of words with no punctuation.)
    tokens = tokenizer.tokenize(text)
    # Ignore single character words and digits.
    tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize the words
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # # Stem the words
    # tokens = [ps.stem(token) for token in tokens]
    # Join the words back into a single string
    text = ' '.join(tokens)
    return text

views = Blueprint('views', __name__)

@views.route('/predict', methods=['POST'])
def home():
    try:
        content = request.get_json(force=True)
        text = content['text']
        text = preprocess_text(text)
        f = open('vectorizer.pkl', 'rb')
        cv = pickle.load(f)
        X = cv.transform([text])
        f = open('final_model.pkl', 'rb')
        clf = pickle.load(f)
        predictions = clf.predict(X)
        probas = clf.predict_proba(X)
        return jsonify({'preds': predictions.tolist(), 'probas': probas.tolist()})
    except Exception as e:
        return jsonify({'errMsg': str(e)})
