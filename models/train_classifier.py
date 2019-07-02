import sys
# import libraries
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''
    The function to load the data.

    Parameters:
        database_filepath (text): The file path for the database.

    Returns:
        X (DataFrame): Message data (features).
        Y (DataFrame): Categories (target).
        category_names (list): Labels for 36 categories.
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='DisasterResponse', con=engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.values
    return X,Y,category_names

def tokenize(text):
    '''
    The function to clean and tokenize the text.

    Parameters:
        text (text): The message text form the database.

    Returns:
        clean_tokens (list): text after cleaned and tokenized.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        if clean_tok != '':
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    The function to build the model.

    Returns:
        cv (model): model after GridSearchCV.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'vect__ngram_range':[(1,1),(1,2)],
                  'clf__estimator__n_estimators':[20, 50]
                  }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The function to evaluate model performance.

    Parameters:
        model (model): The built model.
        X_test (DataFrame): Part of X used for testing.
        Y_test (DataFrame): Part of Y used for evaluate the model.
        category_names (list): Labels for 36 categories.
    '''
    y_pred = model.predict(X_test)
    for i in range(36):
        print("\n -------------------",category_names[i],"-------------------\n")
        print(classification_report(Y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    '''
    The function to save model as a pickle file.

    Parameters:
        model (model): The built model.
        model_filepath (text): The file path for the model.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
