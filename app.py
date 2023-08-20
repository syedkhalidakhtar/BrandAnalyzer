## Import Libraries
from flask import jsonify, Flask, request, render_template

import pandas as pd
import matplotlib.pyplot as plt

from this import s

import numpy as np

import re
import emoji
import random 
from matplotlib.transforms import Transform
import nltk
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.stem.snowball import SnowballStemmer 
import string 
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
from flask import Blueprint, render_template, request

app = Flask(__name__,template_folder='template')

df = pd.read_csv("male_female.csv")
# query = ["Male","Shirt","lowerrange"]     // This is how a query after selection on UI will look like
query = []

df = df.fillna(0)

####
#CODE RELATED TO SENTIMENT ANALYSIS
####

#data cleaning
def replace_user(review, default_replace="user"):
   review = re.sub('\B@\w+', default_replace, review)
   return review

# * Replace emojis with a meaningful text
def demojize(review):
    review = emoji.demojize(review)
    return review

# * Replace occurences of `http://` or `https://` with a default value
def replace_url(review, default_replace=""):
   review = re.sub('(http|https):\/\/\S+', default_replace, review)
   return review

# * Replace occurences of `#_something_` with a default value
def replace_hashtag(review, default_replace=""):
   review = re.sub('#+', default_replace, review)
   return review

# Word Features
# * Lower case each letter in a specific tweet
def to_lowercase(review):
   review = review.lower()
   return review

# Word repetition
def word_repetition(review):
   review = re.sub(r'(.)\1+', r'\1\1', review )
   return review 

# * Replace punctuation repetition with a single occurence ("!!!!!" becomes "!")
def punct_repetition(review, default_replace=""):
    review = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, review)
    return review
        

# * Create a `_fix_contractions` function used to replace contractions with their extended forms by using the contractions dictionnary
def fix_contractions(review):
  review = contractions.fix(review)
  return review

# * Create a `custom_tokenize` function
def custom_tokenize(review,
                    keep_punct = False,
                    keep_alnum = False,
                    keep_stop = False):
  
  token_list = word_tokenize(review)

  if not keep_punct:
    token_list = [token for token in token_list
                  if token not in string.punctuation]

  if not keep_alnum:
    token_list = [token for token in token_list if token.isalpha()]
  
  if not keep_stop:
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    token_list = [token for token in token_list if not token in stop_words]

  return token_list

# * Create a `stem_tokens` function that takes the list of tokens as input and returns a list of stemmed tokens

def stem_tokens(tokens, stemmer):
  token_list = []
  for token in tokens:
    token_list.append(stemmer.stem(token))
  return token_list

# # Putting it all together
def process_review(review, verbose=False):

  if verbose:
            print("Initial review: {}".format(review))
           ##  Features
            review = replace_user(review, "") # replace user tag
            review = replace_url(review)      # replace url
            review = replace_hashtag(review)  # replace hashtag
  if verbose:
            print("Post processing review: {}".format(review))
              ## Word Features
            review = to_lowercase(review)      # lower case
            review = fix_contractions(review) # replace contractions
            review = punct_repetition(review)  # replace punctuation repetition
            review = word_repetition(review)   # replace word repetition
            review = demojize(review)          # replace emojis
  if verbose:
            print("Post Word processing review: {}".format(review))
         
         #   ## Tokenization & Stemming
  tokens = custom_tokenize(review, keep_alnum=False, keep_stop=False) # tokenize
  stemmer = SnowballStemmer("english") # define stemmer
  stem = stem_tokens(tokens, stemmer) # stem tokens

  return stem 

# * Create a `fit_tfidf` function used to build the TF-IDF vectorizer with the corpus
def fit_tfidf(review_corpus):
  tf_vect = TfidfVectorizer(preprocessor = lambda x:x, tokenizer = lambda x:x)
  tf_vect.fit(review_corpus)
  return tf_vect

# Section 4 Sentiment Model
def plot_confusion(cm):
    plt.figure(figsize = (5,5))
    sn.heatmap(cm, annot=True, cmap="Blues", fmt='.0f')
    plt.xlabel("Prediction")
    plt.ylabel("True value")
    plt.title("Confusion Matrix")
    return sn

## **4.1** Train/Test Split
## **4.2** Logistic Regression
### **4.2.1** Model

# * Create a `fit_lr` function used to fit a Logistic Regression model on X and y *training* data
def fit_lr(x_train, y_train):
   model = LogisticRegression()
   model.fit(x_train,y_train)
   return model

# * Create a `predict_tweet` function used to pre-process, transform and predict tweet sentiment
def predict_review(review):
  processed_review = process_review(review)
  transfrmed_review = tf.transform([processed_review])
  prediction = model_lr_tf.predict(transfrmed_review)
  if prediction == 1:
      return 1
  else:
      return 0
#      return "prediction is positive sentiment"
#   else:
#      return "prediction is negative sentiment"





df["Review"] = df["Review"].astype(str)
df["tokens"] = df["Review"].apply(process_review)
# df["review_sentiment"] = df["Label"]
print(df.head(10))


#Convert DataFrame to two lists: one for the tweet tokens(X) and one for the tweet sentiment(y)
#print("code it yourself")
x = df["tokens"].tolist()
y = df["Label"].tolist()
# print(x)
# print(y)

## **3.4** Term Frequency – Inverse Document Frequency (TF-IDF)
# * Use the `fit_cv` function to fit the vectorizer on the corpus, and transform the corpus
tf_vect = fit_tfidf(x)
tf_mtx = tf_vect.transform(x)
ft = tf_vect.get_feature_names()      # * Get the vectorizer features (matrix columns)
print("There are {} features in this corpus".format(len(ft)))
print(ft)
print(tf_mtx.shape)                        # * Print the matrix shape
tf_mtx.toarray()                           # * Convert the matrix to an array
new_tweet = [["Its", "nice", "product"]]   # * Transform a new tweet by using the vectorizer
tf_vect.transform(new_tweet).toarray()

## **4.1** Train/Test Split
print(x)                             # * Check what X and y looks like
print(y)
# * Use the `train_test_split` function to split arrays of X and y into training and testing variables
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=0, train_size=0.50)
# df1 = 
print("Size of X_train: {}".format(len(X_train)))
print("Size of y_train: {}".format(len(y_train)))
print("\n")
print("Size of X_test: {}".format(len(X_test)))
print("Size of y_test: {}".format(len(y_test)))
print("\n")
print("Train proportion: {:.0%}".format(len(X_train)/
                                                (len(X_train)+len(X_test))))
# * Print random tweets, just to verify everything goes as expected
id = random.randint(0,len(X_train))
print("Train review: {}".format(X_train[id]))
print("Sentiment: {}".format(y_train[id]))
## **4.2** Logistic Regression
### **4.2.1** Model ## **4.2.4** TF-IDF
# * Use the `fit_cv` function on training data to build the Bag-of-Words vectorizer
# * Transform X_train and X_test data by using the vectorizer
tf = fit_tfidf(X_train)
# x_train_tf = tf.Transform(X_train)
x_train_tf = tf.transform(X_train)
x_test_tf = tf.transform(X_test)
# * Fit the Logistic Regression model on training data by using the `fit_lr` function
model_lr_tf = fit_lr(x_train_tf,y_train)
## **4.3** Performance Metrics   ###**4.3.3** TF-IDF
# * Use the fitted `model_lr_tf` (TF-IDF) to predict X_test
y_pred_lr_tf = model_lr_tf.predict(x_test_tf)
# df["sentiment"] = y_pred_lr_tf 
print(df.head(10))
# * Print the model accuracy by comparing predictions and real sentiments
print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_tf)))

# * Plot the confusion matrix by using the `plot_confusion` helper function
plot_confusion(confusion_matrix(y_test, y_pred_lr_tf))
plt.show()



df["Sentiment"] = df["Review"].apply(predict_review)
print(df.head(10))



####
#CODE RELATED TO DATA SELECTION ACCORDING TO QUERY
####


def male_product(male_reviews):

    if query[1]=="Shirt" :
      male_reviews = male_reviews[male_reviews["Category"]=="Shirt"]
      print("male product: {}".format(male_reviews))
      return male_reviews
    elif query[1]=="T-shirt":
      male_reviews =  male_reviews[male_reviews["Category"]=="T-Shirt"]
      return male_reviews
    elif query[1]=="Hoodies":
      male_reviews = male_reviews[male_reviews["Category"]=="Hoodies"]
      return male_reviews
    elif query[1]=="Trousers":
      male_reviews = male_reviews[male_reviews["Category"]=="Trousers"]
      return male_reviews

def female_product(female_reviews):

                if query[1]=="T-shirt":
                 female_reviews = female_reviews[female_reviews["Category"]=="T-shirt"]
                #  print("female product: {}".format(female_reviews))
                 return female_reviews
                elif query[1]=="Tops":
                    female_reviews = female_reviews[female_reviews["Category"]=="Tops"]
                    return female_reviews
                elif query[1]=="Sarees":
                    female_reviews = female_reviews[female_reviews["Category"]=="Sarees"]
                    return female_reviews
                elif query[1]=="Kurta&Kurtis":
                    female_reviews = female_reviews[female_reviews["Category"]=="Kurta&Kurtis"]
                    return female_reviews

def Pricerange(fe_male_reviews):
            if query[2]== "lowerrange": 
                fe_male_reviews = fe_male_reviews[fe_male_reviews["Price"]<=700]
                # print("male product with price range: {}".format(fe_male_reviews))
                return fe_male_reviews
            elif query[2] == "midrange": 
                # fe_male_reviews = fe_male_reviews[fe_male_reviews["Price"]>700] and fe_male_reviews[fe_male_reviews["Price"]>700] 
                fe_male_reviews1 = fe_male_reviews[(fe_male_reviews["Price"]>700) & (fe_male_reviews["Price"]<=1400)]
                # fe_male_reviews2 = fe_male_reviews[fe_male_reviews["Price"]<=1400]
                # fe_male_reviews = pd.concat([fe_male_reviews1, fe_male_reviews2 ],ignore_index=True)
                # print("male product with price range: {}".format(fe_male_reviews))
                fe_male_reviews = fe_male_reviews1
                return fe_male_reviews
            elif query[2] == "highrange": 
                fe_male_reviews = fe_male_reviews[fe_male_reviews["Price"]>1400]
                # print("male product with price range: {}".format(fe_male_reviews))
                return fe_male_reviews
    
def Male_Female_data(df):
    #     #On selecting gender
            if query[0] == "Male":
                male_reviews = df[df["Gender"]=="Male"]
                print("male data: {}".format(male_reviews))
                male_reviews = male_product(male_reviews)
                male_reviews = Pricerange(male_reviews)
                    # print("male data: {}".format(male_reviews))
                return male_reviews

            elif query[0] == "Female":
                    female_reviews = df[df["Gender"]=="Female"]
                    print("Female data: {}".format(female_reviews))
                    female_reviews = female_product(female_reviews)
                    print("Female Product data: {}".format(female_reviews))
                    female_reviews = Pricerange(female_reviews)
                    print("Female Product with price range data: {}".format(female_reviews))
                    return female_reviews
		


####
#CODE RELATED TO FLASK-BASED FRONTEND
####



@app.route("/")
def hello_world():
    title = "Brand Analyzer"
    return render_template("home.html", title=title)

@app.route("/home.html")
def backtohome():
    title = "Brand Analyzer"
    return render_template("home.html", title=title)

@app.route("/about")
def aboutus():
    title = "About us"
    return render_template("about.html", title=title)

@app.route('/gender.html')
def Next():
    title="Brand Analyzer"
    return render_template("gender.html",title=title)

@app.route('/gender.html', methods= ['POST'])
def post_gender():
    title = "Brand Analyzer"
    gender_type= request.form.get("gendertype")
    if gender_type=="Male":
        query.append(gender_type)
        return render_template("male.html",title=title)
    else:
        query.append(gender_type)
        return render_template("female.html", title=title)

@app.route("/male.html")
def male():
    title ="Brand Analyzer"
    return render_template("pricerange.html", title=title)

@app.route("/male.html", methods=['POST'])
def post_maleproducts():
    title = "Brand Analyzer"
    product_type=request.form.get("male_product")
    query.append(product_type)
    return render_template("pricerange.html", title=title)


@app.route("/female.html")
def female():
    title ="Band Analyzer"
    return render_template("pricerange.html", title=title)

@app.route("/female.html", methods=['POST'])
def post_femaleproducts():
    title="Brand Analyzer"
    product_type=request.form.get("female_product")
    query.append(product_type)
    return render_template("pricerange.html", title = title)


@app.route("/pricerange.html")
def price():
    title="Brand Analyzer"
    return render_template("pricerange.html", title=title)
 
@app.route("/pricerange.html", methods=['POST'])
def pricerange():
    title="Brand Analyzer"
    price_range=request.form.get("range")
    query.append(price_range)
    var1 = Male_Female_data(df)
    pos = var1[var1["Sentiment"] == 1.0]
    neg = var1[var1["Sentiment"] == 0.0]
    values = pos['Brand']. value_counts()    #counting the unique value frequency
    # print(values)
    new_values=values.to_dict()
    updict = {'Brand' : 'Positive Review'}
    updict.update(new_values)
    print('new value')
    print(updict)
    # if price_range == "lowerrange":
    return render_template("display.html", title=title, data=updict)
    

        
####
#main
####

if __name__ == "__main__":
    app.run()