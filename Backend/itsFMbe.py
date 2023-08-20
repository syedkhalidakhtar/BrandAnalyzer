import code
import pandas as pd
import numpy as np

# creating some variables to store info
positive = 0
negative = 0
neutral = 0
i=0
# # * Read Dataframe stored .csv` format
# making data frame from csv file
df = pd.read_csv("sample_dataset1.csv")
# print(df)


# # * Check how many tweets there are in total
print("No of Analyzing Tweets: {}".format(len(df)))

# * Print a tweet and its sentiment based on a tweet ID
print(" want to see one of any review")
print(f" Enter the number between {0} and {len(df)}")

review_s_no = int(input())
review = df.iloc[review_s_no]
# review = S no., Gender, Category, Price,     Brand,     Product,      Review,        Sentiment
            # 5,    Male,   T-Shirt,  439,  Kook N Keech,   Men Olive Green Printed Round Neck Pure Cotton T-shirt,"Quality of fabric is not good, ij just one wash it has got hole in it.",                                                               Negative

print("review: {}".format(review["Review"]))
print("product sentiment: {}".format(review["Sentiment"]))


# # * Import the `pyplot` module from the matplotlib package 
import matplotlib.pyplot as plt
sentiment_count = df["Sentiment"].value_counts()

plt.pie(sentiment_count, labels=sentiment_count.index,
        autopct='%1.1f%%',shadow=True,startangle=140)
plt.show()

# # Print the count of positive and negative reviews 
print("Number of + reviews: {}".format(df[df["Sentiment"]=="Positive"].count()[0]))
print("Number of - reviews: {}".format(df[df["Sentiment"]=="Negative"].count()[0]))


from wordcloud import WordCloud

# # # What are the words most often present in positive reviews
pos_reviews = df[df["Sentiment"]=="Positive"]
txt = " ".join(review.lower() for review in pos_reviews["Review"])
wordcloud = WordCloud().generate(txt)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# # and negative reviews 
neg_reviews = df[df["Sentiment"]=="Negative"]
txt = " ".join(review.lower() for review in neg_reviews["Review"])
wordcloud = WordCloud().generate(txt)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# # Text Normalization
# Import regex package
import re
# #  Twitter features
# # Example of a random tweet that can be found on Twitter
review = "RT @AIOutsider I love this! 👍 https://AIOutsider.com #NLP #Fun"
# #  RT Tag
# # R : match "R" character
# # T : match "T" character
# # \s : match any whitespace character
# # + : match one or more of the preceding tokens

# #### Handle the RT Tag
# #  Replace occurences of `RT` with a default value
# def replace_review(review, default_replace=""):
#   review = re.sub('RT\s+', default_replace, review)
#   return review

# print("Processed tweet: {}".format(replace_review(review)))

# #  **2.1.2** @User Tag
# # * `\B` : match any position that is not a word boundary
# # * `@` : match "@" character
# # * `\w` : match any word character 
# # * `+` : match one or more of the preceding tokens

# #### Handle the User Tag
# # * Replace `@_Someone_` with a default user tag

# def replace_user(review, default_replace="user"):
#   review = re.sub('\B@\w+', default_replace, review)
#   return review

# print("Processed review: {}".format(replace_user(review)))

# #  **2.1.3** Emojis
# # * Install the `emoji` package
# pip install emoji --upgrade

# # * Import the installed package
# import emoji
# # * Replace emojis with a meaningful text
# def demojize(review):
#   review = emoji.demojize(review)
#   return review

# print("Processed review: {}".format(demojize(review)))

# #  **2.1.4** URL
# #### Need a hint?
# # * `(http|https)` : capturing group matching either http or https
# # * `:` : match the ":" character
# # * `\/` : match the "/" charachter
# # * `\S` : match any character that is not whitespace
# # * `+` : match one or more of the preceding tokens

# #### Handle the URL
# # * Replace occurences of `http://` or `https://` with a default value
# def replace_url(review, default_replace=""):
#   review = re.sub('(http|https):\/\/\S+', default_replace, review)
#   return review

# print("Processed review: {}".format(replace_url(review)))

# ###  **2.1.5** Hashtags
# # * Replace occurences of `#_something_` with a default value

# def replace_hashtag(review, default_replace=""):
#   review = re.sub('#+', default_replace, review)
#   return review

# print("Processed review: {}".format(replace_hashtag(review)))

# ## **2.2** Word Features

# # Let's now have a look at some other features that are not really Twitter-dependant
# review = "LOOOOOOOOK at this ... I'd like it so much!"

# ### **2.2.1** Remove upper capitalization
# # * Lower case each letter in a specific tweet
# def to_lowercase(review):
#   review = review.lower()
#   return review

# print("Processed tweet: {}".format(to_lowercase(review)))

# ###**2.2.2** Word repetition
# # * Replace word repetition with a single occurence ("oooooo" becomes "oo")

# def word_repetition(review):
#   review = re.sub(r'(.)\1+', r'\1\1', review )
#   return review 

# print("Processed review : {}".format(word_repetition(review)))

# ### **2.2.3** Punctuation repetition
# # * Replace punctuation repetition with a single occurence ("!!!!!" becomes "!")

# def punct_repetition(review, default_replace=""):
#   review = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, review)
#   return review

# print("Processed review: {}".format(punct_repetition(review)))

# ###  Word contraction
# # * Install the `contractions` package
# pip install contractions

# # * Import the installed package
# import contractions

# # * Use `contractions_dict` to list most common contractions
# print(contractions.contractions_dict)

# # * Create a `_fix_contractions` function used to replace contractions with their extended forms by using the contractions dictionnary
# def _fix_contractions(tweet):
#   for k, v in contractions.contractions_dict.items():
#     tweet = tweet.replace(k, v)
#   return tweet

# print("Processed tweet: {}".format(_fix_contractions(tweet)))

# # * Create a `_fix_contractions` function used to replace contractions with their extended forms by using the contractions package

# def fix_contractions(tweet):
#   tweet = contractions.fix(tweet)
#   return tweet

# print("Processed tweet: {}".format(fix_contractions(tweet)))

# <img src='https://drive.google.com/uc?export=view&id=1GYj-wj-so8jQ9-VDz1ayehgVh39Jmd4H' width=250px>

# ## **2.3** Tokenization
# # * Install the `NLTK` package

# pip install nltk

# ### **2.3.1** Easy Tokenization
# # * Import `NLTK`
# # * Import the `word_tokenize` module from NLTK 
# # * Download the `Punkt` tokenizer model from NLTK
# import nltk
# from nltk.tokenize import word_tokenize
# nltk.download('punkt')

# # * Simple tweet to be tokenized
# tweet = "These are 5 different words!"

# # * Create a `tokenize()` function that takes a tweet as input and returns a list of tokens
# def tokenize(tweet):
#   tokens = word_tokenize(tweet)
#   return tokens

# # * Use the `tokenize()` function to print the tokenized version of a tweet
# print(type(tokenize(tweet)))
# print("Tweet tokens: {}".format(tokenize(tweet)))

# ###**2.3.2** Custom Tokenization
# # * Import the `string` package 
# import string


# # * Retrieve english punctuation signs by using the `string` package
# print(string.punctuation)

# # * Import the `stopwords` module from NLTK
# # * Download `stopwords` data from NLTK
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# # * Create a set of english stopwords
# stop_words = set(stopwords.words('english'))
# print(stop_words)

# # * Remove some stopwords from the set
# stop_words.discard('not')
# print(stop_words)

# # * Create a `custom_tokenize` function
# def custom_tokenize(tweet,
#                     keep_punct = False,
#                     keep_alnum = False,
#                     keep_stop = False):
  
#   token_list = word_tokenize(tweet)

#   if not keep_punct:
#     token_list = [token for token in token_list
#                   if token not in string.punctuation]

#   if not keep_alnum:
#     token_list = [token for token in token_list if token.isalpha()]
  
#   if not keep_stop:
#     stop_words = set(stopwords.words('english'))
#     stop_words.discard('not')
#     token_list = [token for token in token_list if not token in stop_words]

#   return token_list

# #   * Test the function with a particular tweet
# tweet = "these are 5 different words!"
# print("Tweet tokens: {}".format(custom_tokenize(tweet, 
#                                                 keep_punct=True, 
#                                                 keep_alnum=True, 
#                                                 keep_stop=True)))
# print("Tweet tokens: {}".format(custom_tokenize(tweet, keep_stop=True)))
# print("Tweet tokens: {}".format(custom_tokenize(tweet, keep_alnum=True)))

# <img src='https://drive.google.com/uc?export=view&id=1GYj-wj-so8jQ9-VDz1ayehgVh39Jmd4H' width=250px>

# ## **2.4** Stemming

# # * Import different libraries and modules used for stemming
# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer
# from nltk.stem.snowball import SnowballStemmer

# # * List of tokens to stem (remember that we stem tokens and not entire sentences)
# tokens = ["manager", "management", "managing"]

# # * Stemmers can be defined by directly using NLTK
# porter_stemmer = PorterStemmer()
# lancaster_stemmer = LancasterStemmer()
# snoball_stemmer = SnowballStemmer('english')

# # * Create a `stem_tokens` function that takes the list of tokens as input and returns a list of stemmed tokens
# def stem_tokens(tokens, stemmer):
#   token_list = []
#   for token in tokens:
#     token_list.append(stemmer.stem(token))
#   return token_list

# # * Print the different results and compare the stemmed tokens
# print("Porter stems: {}".format(stem_tokens(tokens, porter_stemmer)))
# print("Lancaster stems: {}".format(stem_tokens(tokens, lancaster_stemmer)))
# print("Snowball stems: {}".format(stem_tokens(tokens, snoball_stemmer))

# # * Check over-stemming and under-stemming
# tokens = ["international", "companies", "had", "interns"]

# print("Porter stems: {}".format(stem_tokens(tokens, porter_stemmer)))
# print("Lancaster stems: {}".format(stem_tokens(tokens, lancaster_stemmer)))
# print("Snowball stems: {}".format(stem_tokens(tokens, snoball_stemmer)))

# <img src='https://drive.google.com/uc?export=view&id=1GYj-wj-so8jQ9-VDz1ayehgVh39Jmd4H' width=250px>

# ## **2.5** Lemmatization

# # * Import different libraries and modules used for lemmatization
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# nltk.download('wordnet')

# # * List of tokens to lemmatize (remember that we lemmatize tokens and not entire sentences)
# tokens = ["international", "companies", "had", "interns"]

# # * Part of Speech (POS) tagging
# word_type = {"international": wordnet.ADJ, 
#              "companies": wordnet.NOUN, 
#              "had": wordnet.VERB, 
#              "interns": wordnet.NOUN
#              }

# # * Create the lemmatizer by using the `WordNet` module
# lemmatizer = WordNetLemmatizer()

# # * Create a `lemmatize_tokens` function that takes the list of tokens as input and returns a list of lemmatized tokens
# def lemmatize_tokens(tokens, word_type, lemmatizer):
#   token_list = []
#   for token in tokens:
#     token_list.append(lemmatizer.lemmatize(token, word_type[token]))
#   return token_list

# print("Tweet lemma: {}".format(
#     lemmatize_tokens(tokens, word_type, lemmatizer)))

# <img src='https://drive.google.com/uc?export=view&id=1GYj-wj-so8jQ9-VDz1ayehgVh39Jmd4H' width=250px>

# ## **2.6** Putting it all together
# # * Long and complex tweet to be processed
# complex_tweet = r"""RT @AIOutsider : he looooook, 
# THis is a big and complex TWeet!!! 👍 ... 
# We'd be glad if you couldn't normalize it! 
# Check https://t.co/7777 and LET ME KNOW!!! #NLP #Fun"""

# # * Create a custom `process_tweet` function that can be used to process tweets end-to-end
# # * **Note**: this function will be used as a base for the following sections, so be careful!

# def process_tweet(tweet, verbose=False):
#   if verbose: print("Initial tweet: {}".format(tweet))

#   ## Twitter Features
#   tweet = replace_retweet(tweet) # replace retweet
#   tweet = replace_user(tweet, "") # replace user tag
#   tweet = replace_url(tweet) # replace url
#   tweet = replace_hashtag(tweet) # replace hashtag
#   if verbose: print("Post Twitter processing tweet: {}".format(tweet))

#   ## Word Features
#   tweet = to_lowercase(tweet) # lower case
#   tweet = fix_contractions(tweet) # replace contractions
#   tweet = punct_repetition(tweet) # replace punctuation repetition
#   tweet = word_repetition(tweet) # replace word repetition
#   tweet = demojize(tweet) # replace emojis
#   if verbose: print("Post Word processing tweet: {}".format(tweet))

#   ## Tokenization & Stemming
#   tokens = custom_tokenize(tweet, keep_alnum=False, keep_stop=False) # tokenize
#   stemmer = SnowballStemmer("english") # define stemmer
#   stem = stem_tokens(tokens, stemmer) # stem tokens

#   return stem

# #   * Test your `process_tweet` function!
# print(process_tweet(complex_tweet, verbose=False))

# # * Look at some more examples! 
# # * **Note:** it's totally possible you encounter some strange tweet processing (happens if the original tweet is initially strangely written) 

# import random

# for i in range(5):
#   tweet_id = random.randint(0,len(df))
#   tweet = df.iloc[tweet_id]["tweet_text"]
#   print(process_tweet(tweet, verbose=True))
#   print("\n")


# # `Section 3` Text Representation </h2>

# ## **3.1** Processing Tweets

# # * Install the `Scikit-Learn` package which is very useful for a lot of different ML tasks. 
# # * **Note:** make sure it is installed and up-to-date (once installed/updated, you might be asked to reload Colab). 
# pip install -U scikit-learn

# # * Apply `process_tweet` function created in section 2 to the entire DataFrame
# # * Convert sentiment to 1 for "positive" and 0 for "negative" sentiment
# print("do it yourself")

# # * Convert DataFrame to two lists: one for the tweet tokens (X) and one for the tweet sentiment (y)
# print("code it yourself")


# print(X)
# print(y)


# ## **3.2** Positive/Negative Frequency
# # * Corpus of tweet tokens used for the first method
# corpus = [["i", "love", "nlp"],
#           ["i", "miss", "you"],
#           ["i", "love", "you"],
#           ["you", "are", "happy", "to", "learn"],
#           ["i", "lost", "my", "computer"],
#           ["i", "am", "so", "sad"]]

# sentiment = [1, 0, 1, 1, 0, 0]


# * Create a `build_freqs` function used to build a dictionnary with the word and sentiment as index and the count of occurence as value


# <table style="width:100%">
#   <tr>
#     <th>Word</th>
#     <th>Positive</th>
#     <th>Negative</th>
#   </tr>
#   <tr>
#     <td>love</td>
#     <td>dict[(love, 1)]</td>
#     <td>dict[(love, 0)]</td>
#   </tr>
#   <tr>
#     <td>lost</td>
#     <td>dict[(lost, 1)]</td>
#     <td>dict[(lost, 0)]</td>
#   </tr>
#   <tr>
#     <td>happy</td>
#     <td>dict[(happy, 1)]</td>
#     <td>dict[(happy, 0)]</td>
#   </tr>
# </table>


# #code


# # * Build the frequency dictionnary on the corpus by using the function
# #code

# print(freqs)

# # * Build the frequency dictionnary on the entire dataset by using the function

# #code

# print("Frequency of word 'love' in + tweets: {}".format(freqs_all[("love", 1)]))
# print("Frequency of word 'love' in - tweets: {}".format(freqs_all[("love", 0)]))

# # * Create a `tweet_to_freqs` function used to convert tweets to a 2-d array by using the frequency dictionnary
# #code

# # * Print the 2-d vector by using the `tweet_to_freqs` function and the *corpus* dictionnary
# print(tweet_to_freq(["i", "love", "nlp"], freqs))

# # * Print the 2-d vector by using the `tweet_to_freqs` function and the *dataset* dictionnary
# print(tweet_to_freq(["i", "love", "nlp"], freqs_all))

# # * Plot word vectors in a chart and see where they locate
# fig, ax = plt.subplots(figsize = (8, 8))

# word1 = "happi"
# word2 = "sad"

# def word_features(word, freqs):
#   x = np.zeros((2,))
#   if (word, 1) in freqs:
#     x[0] = np.log(freqs[(word, 1)] + 1)
#   if (word, 0) in freqs:
#     x[1] = np.log(freqs[(word, 0)] + 1)
#   return x

# x_axis = [word_features(word, freqs_all)[0] for word in [word1, word2]]
# y_axis = [word_features(word, freqs_all)[1] for word in [word1, word2]]

# ax.scatter(x_axis, y_axis)  

# plt.xlabel("Log Positive count")
# plt.ylabel("Log Negative count")

# ax.plot([0, 9], [0, 9], color = 'red')
# plt.text(x_axis[0], y_axis[0], word1)
# plt.text(x_axis[1], y_axis[1], word2)
# plt.show()

# <img src='https://drive.google.com/uc?export=view&id=1GYj-wj-so8jQ9-VDz1ayehgVh39Jmd4H' width=250px>

# ## **3.3** Bag of Word
# # * Corpus of tweet tokens used for the second method
# corpus = [["love", "nlp"],
#           ["miss", "you"],
#           ["hate", "hate", "hate", "love"],
#           ["happy", "love", "hate"],
#           ["i", "lost", "my", "computer"],
#           ["i", "am", "so", "sad"]]

# # * Import `CountVectorizer` from the Scikit-learn Library
# from sklearn.feature_extraction.text import CountVectorize

# # * Create a `fit_cv` function used to build the Bag-of-Words vectorizer with the corpus
# #code

# # * Get the vectorizer features (matrix columns)
# #code

# print("There are {} features in this corpus".format(len(ft)))
# print(ft)

# # * Convert the corpus to a matrix by using the vectorize
# # code 

# # * Print the matrix shape
# print("Matrix shape is: {}".format()

# # * Convert the matrix to an array
# #code

# # * Transform a new tweet by using the vectorizer
# new_tweet = [["lost", "lost", "miss", "miss"]]
# cv_vect.transform(new_tweet).toarray()

# unknown_tweet = [["John", "drives", "cars"]]
# cv_vect.transform(unknown_tweet).toarray()


# ## **3.4** Term Frequency – Inverse Document Frequency (TF-IDF)

# # * Corpus of tweet tokens used for the third method
# corpus = [["love", "nlp"],
#           ["miss", "you"],
#           ["hate", "hate", "hate", "love"],
#           ["happy", "love", "hate"],
#           ["i", "lost", "my", "computer"],
#           ["i", "am", "so", "sad"]]

# # * Import `TfidfVectorizer` from the Scikit-learn Library
# from sklearn.feature_extraction.text import TfidfVectorizer

# # * Create a `fit_tfidf` function used to build the TF-IDF vectorizer with the corpus
# #code

# # * Use the `fit_cv` function to fit the vectorizer on the corpus, and transform the corpus
# #code

# # * Get the vectorizer features (matrix columns)
# #code

# print("There are {} features in this corpus".format(len(ft)))
# print(ft)

# # * Print the matrix shape
# print(tf_mtx.shape)

# # * Convert the matrix to an array
# tf_mtx.toarray()

# # * Transform a new tweet by using the vectorizer
# new_tweet = [["I", "hate", "nlp"]]
# tf_vect.transform(new_tweet).toarray()


# # `Section 4` Sentiment Model
# ## Helper function

# # This function will be used to plot the confusion matrix for the different models we will create
# import seaborn as sn

# def plot_confusion(cm):
#   plt.figure(figsize = (5,5))
#   sn.heatmap(cm, annot=True, cmap="Blues", fmt='.0f')
#   plt.xlabel("Prediction")
#   plt.ylabel("True value")
#   plt.title("Confusion Matrix")
#   return sn

# ## **4.1** Train/Test Split
# # * Check what X and y looks like
# print(X)
# print(y)

# # * Import the `train_test_split` function from the Scikit-Learn package
# from sklearn.model_selection import train_test_split

# # * Use the `train_test_split` function to split arrays of X and y into training and testing variables
# #code

# # * Print the size of these news variables
# print("Size of X_train: {}".format(len(X_train)))
# print("Size of y_train: {}".format(len(y_train)))
# print("\n")
# print("Size of X_test: {}".format(len(X_test)))
# print("Size of y_test: {}".format(len(y_test)))
# print("\n")
# print("Train proportion: {:.0%}".format(len(X_train)/
#                                         (len(X_train)+len(X_test))))


# # * Print random tweets, just to verify everything goes as expected
# id = random.randint(0,len(X_train))
# print("Train tweet: {}".format(X_train[id]))
# print("Sentiment: {}".format(y_train[id]))

# <img src='https://drive.google.com/uc?export=view&id=1GYj-wj-so8jQ9-VDz1ayehgVh39Jmd4H' width=250px>

# ## **4.2** Logistic Regression
# ### **4.2.1** Model
# # * Import the `LogisticRegression` model from Scikit-Learn
# from sklearn.linear_model import LogisticRegression


# # * Create a `fit_lr` function used to fit a Logistic Regression model on X and y *training* data
# #code


# ### **4.2.2** Pos/Neg Frequency
# # * Use the `build_freqs` function on training data to create a frequency dictionnary
# # * Use the frequency dictionnary together with the `tweet_to_freq` function to convert X_train and X_test data to 2-d vectors
# #code

# # * Fit the Logistic Regression model on training data by using the `fit_lr` function
# # * Print the model coefficients (betas and intercept)

# #code

# ### **4.2.3** Count Vector
# # * Use the `fit_cv` function on training data to build the Bag-of-Words vectorizer
# # * Transform X_train and X_test data by using the vectorizer

# #code

# # * Fit the Logistic Regression model on training data by using the `fit_lr` function
# #code

# ## **4.2.4** TF-IDF
# # * Use the `fit_cv` function on training data to build the Bag-of-Words vectorizer
# # * Transform X_train and X_test data by using the vectorizer
# #code

# # * Fit the Logistic Regression model on training data by using the `fit_lr` function
# #code


# ## **4.3** Performance Metrics
# # * Import the `accuracy score` and `confusion matrix` from Scikit-Learn
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# ### **4.3.1** Positive/Negative Frequencies
# # * Use the fitted `model_lr_pn` (positive/negative frequencies) to predict X_test
# #code

# # * Print the model accuracy by comparing predictions and real sentiments
# print("LR Model Accuracy: {:.2%}".format())

# # * Plot the confusion matrix by using the `plot_confusion` helper function
# #code

# ### **4.3.2** Count Vector
# # * Use the fitted `model_lr_cv` (Bag-of-words) to predict X_test
# #code

# # * Print the model accuracy by comparing predictions and real sentiments
# print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_cv)))

# # * Plot the confusion matrix by using the `plot_confusion` helper function
# plot_confusion(confusion_matrix(y_test, y_pred_lr_cv))

# ###**4.3.3** TF-IDF
# # * Use the fitted `model_lr_tf` (TF-IDF) to predict X_test

# y_pred_lr_tf = model_lr_tf.predict(X_test_tf)

# # * Print the model accuracy by comparing predictions and real sentiments
# print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_tf)))

# # * Plot the confusion matrix by using the `plot_confusion` helper function
# plot_confusion(confusion_matrix(y_test, y_pred_lr_tf))

# ## **4.4** Mini-Pipeline
# # * Final tweet used to check if the model works as well as expected
# # * **Note:** don't hesitate to input your own tweet!

# your_tweet = """RT @AIOutsider: tune in for more amazing NLP content! 
# And don't forget to visit https://AIOutsider.com ...""".

# # * Create a `predict_tweet` function used to pre-process, transform and predict tweet sentiment
# #code

# # * ... Predict your tweet sentiment by using the `predict_tweet` function!
# predict_tweet(your_tweet)




