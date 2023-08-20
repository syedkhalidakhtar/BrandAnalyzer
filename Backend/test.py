import re
import emoji
import contractions
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

review = "RT @AIOutsider I love this! 👍 https://AIOutsider.com #NLP #Fun"

def replace_review(review, default_replace=""):
  review = re.sub('RT\s+', default_replace, review)
  return review

updated_review = replace_review(review)
print("Processed tweet: {}".format(updated_review))

def replace_user(updated_review, default_replace=""):
  review = re.sub('\B@\w+', default_replace, updated_review)
  review = re.sub('#+', default_replace, review)
  return review

updated_review = replace_user(updated_review)
print("Processed review: {}".format(updated_review))

# pip install emoji --upgrade
def demojize(review):
  review = emoji.demojize(review)
  return review

updated_review = demojize(updated_review)
print("Processed review: {}".format(demojize(updated_review)))

def replace_url(review, default_replace=""):
  review = re.sub('(http|https):\/\/\S+', default_replace, review)
  return review

updated_review = replace_url(updated_review)

def to_lowercase(review):
  review = review.lower()
  return review

updated_review = to_lowercase(updated_review)

def word_repetition(review):
  review = re.sub(r'(.)\1+', r'\1\1', review )
  return review 

updated_review = word_repetition(updated_review)

def punct_repetition(review, default_replace=""):
  review = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, review)
  return review

updated_review = punct_repetition(updated_review)

#  Word contraction
print(contractions.contractions_dict)
def _fix_contractions(review):
  for k, v in contractions.contractions_dict.items():
    review = review.replace(k, v)
  return review

updated_review = _fix_contractions(updated_review)

def fix_contractions(review):
  review = contractions.fix(review)
  return review

updated_review = fix_contractions(updated_review)


nltk.download('punkt') 

tweet = "These are 5 different words!"

def tokenize(tweet):
  tokens = word_tokenize(tweet)
  return tokens 

print(type(tokenize(tweet)))
print("Tweet tokens: {}".format(tokenize(tweet)))

# * Retrieve english punctuation signs by using the `string` package
print(string.punctuation)

nltk.download('stopwords')

# * Create a set of english stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)

# * Remove some stopwords from the set
stop_words.discard('not')
print(stop_words)

