__author__ = 'Kevin Anthony Smith'

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import json
import pickle
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

tweets = []
ids = []
for line in open('xab.txt').readlines():   
    data = json.loads(line)
    if data['text'] in ids:
        continue
    ids.append(data['text'])
    lower = data['text'].lower()
    tweets.append(lower)
    #print data

# Extract the vocabulary of keywords
vocab = dict()
for text in tweets:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if vocab.has_key(term):
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1

# Remove terms whose frequencies are less than a threshold (e.g., 20)
vocab = {term: freq for term, freq in vocab.items() if freq > 25}
# Generate an id (starting from 0) for each term in vocab
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
#print 'The list of keywords that are considered: '
#print vocab
f = open('project_vocab.pkl', 'w')
f.write(pickle.dumps(vocab))
f.close()

#Create y
#Deadppol: 0, #Jungle Book: 1, 
y = []
final = []
original = [0,0,0,0,0,0,0,0,0,0]
dsfa = 0
for tweet in tweets:
    if tweet.find('junglebook') != -1:
        y.append(0)
        final.append(tweet)
        original[0] += 1
    elif tweet.find('deadpool') != -1:
        if original[1] > 10000:
            continue
        y.append(1)
        final.append(tweet)
        original[1] += 1

    elif (tweet.find('theboss') != -1) | (tweet.find('thebossfilm') != -1):
        y.append(2)
        final.append(tweet)
        original[2] += 1
    elif tweet.find('batmanvsuperman') != -1:
        y.append(3)
        final.append(tweet)
        original[3] += 1
    elif tweet.find('zootopia') != -1:
        y.append(4)
        final.append(tweet)
        original[4] += 1
    elif tweet.find('criminalmovie') != -1:
        y.append(5)
        final.append(tweet)
        original[5] += 1
    elif tweet.find('mybigfatgreekwedding2') != -1:
        y.append(6)
        final.append(tweet)
        original[6] += 1
    elif tweet.find('hardcorehenry') != -1:
        y.append(7)
        final.append(tweet)
        original[7] += 1
        #dsfa = dsfa + 1
    elif (tweet.find('10 cloverfield lane') != -1) | (tweet.find('10cloverfieldln') != -1):
        y.append(8)
        final.append(tweet)
        original[8] += 1
    elif (tweet.find('therevenant') != -1) | (tweet.find('the revenant') != -1):
        y.append(9)
        final.append(tweet)
        original[9] += 1
#print y
#print dsfa

## Generate X and y
X = []
for tweet_text in final:
    x = [0] * len(vocab)
    terms = [term for term in tweet_text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    X.append(x)

# 10 folder cross validation to estimate the best w and b
clf = LogisticRegression()
clf.fit(X, y)

f = open('trained_project_classifier.pkl', 'w')
f.write(pickle.dumps(clf))
f.close()

#tweets = []
#ids = []
#for line in open('movie_data3.txt').readlines():   
#    data = json.loads(line)
#    if data['text'] in ids:
#        continue
#    ids.append(data['text'])
#    lower = data['text'].lower()
#    tweets.append(lower)
### Generate X and y
#X = []
#for tweet_text in tweets:
#    x = [0] * len(vocab)
#    terms = [term for term in tweet_text.split() if len(term) > 2]
#    for term in terms:
#        if vocab.has_key(term):
#            x[vocab[term]] += 1
#    X.append(x)
##print X
#final = []
#original = [0,0,0,0,0,0,0,0,0,0]
#for tweet in tweets:
#    if tweet.find('junglebook') != -1:
#        final.append(tweet)
#        original[0] += 1
#    elif tweet.find('deadpool') != -1:
#        final.append(tweet)
#        original[1] += 1
#    elif (tweet.find('theboss') != -1) | (tweet.find('thebossfilm') != -1):
#        final.append(tweet)
#        original[2] += 1
#    elif tweet.find('batmanvsuperman') != -1:
#        final.append(tweet)
#        original[3] += 1
#    elif tweet.find('zootopia') != -1:
#        final.append(tweet)
#        original[4] += 1
#    elif tweet.find('criminalmovie') != -1:
#        final.append(tweet)
#        original[5] += 1
#    elif tweet.find('mybigfatgreekwedding2') != -1:
#        final.append(tweet)
#        original[6] += 1
#    elif tweet.find('hardcorehenry') != -1:
#        final.append(tweet)
#        original[7] += 1
#        #dsfa = dsfa + 1
#    elif (tweet.find('10 cloverfield lane') != -1) | (tweet.find('10cloverfieldln') != -1):
#        final.append(tweet)
#        original[8] += 1
#    elif (tweet.find('therevenant') != -1) | (tweet.find('the revenant') != -1):
#        final.append(tweet)
#        original[9] += 1
##print y
#    
#
##y2 = []
#np.set_printoptions(threshold=np.nan)
#y2 = clf.predict(X)
#print y2

#count = [0,0,0,0,0,0,0,0,0,0]
#for preds in y2:
#    count[preds] += 1
#print "                         Predicted    Original"
#print "Jungle Book:                ", count[0], "     ", original[0]
#print "Deadpool:                   ", count[1], "     ", original[1]
#print "The Boss:                   ", count[2], "     ", original[2]
#print "Batman V Superman:          ", count[3], "     ", original[3]
#print "Zootopia:                   ", count[4], "     ", original[4]
#print "Criminal:                   ", count[5], "     ", original[5]
#print "My Big Fat Greek Wedding 2: ", count[6], "     ", original[6]
#print "Hardcore Henry:             ", count[7], "     ", original[7]
#print "10 Cloverfield Lane         ", count[8], "     ", original[8]
#print "The Revenant:               ", count[9], "     ", original[9]
# predict the class labels of new tweets
#tweets = []
#for line in open('testing_tweets.txt').readlines():
#    tweets.append(json.loads(line))
#
## Generate X for testing tweets
#X = []
#for tweet_id, tweet_text in tweets:
#    x = [0] * len(vocab)
#    terms = [term for term in tweet_text.split() if len(term) > 2]
#    for term in terms:
#        if vocab.has_key(term):
#            x[vocab[term]] += 1
#    X.append(x)
#y = clf.predict(X)
## lr.predict_proba(X) will return you the predict probabilities
#
#f = open('project_predictions.txt', 'w')
#f.write(y2)
#f.close()
