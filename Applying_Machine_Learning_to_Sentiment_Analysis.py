
# coding: utf-8

# ## Obtaining the IMDb movie review dataset

# In[49]:


import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = '/Users/xuhuahu/Python_Machine_Learning/aclImdb/%s/%s' % (s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file), 'r') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
df.columns = ['review', 'sentiment']


# In[50]:


import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)


# In[51]:


df = pd.read_csv('./movie_data.csv')
df.head(3)


# In[4]:


import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)


# In[5]:


print(count.vocabulary_)


# In[6]:


print(bag.toarray())


# ## Assessing word relevancy via frequency-inverse document frequency

# In[7]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# In[8]:


import re
def preprocessor(text):
    text = re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-','')
    return text


# In[9]:


print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("</a>This :) is :( a test :-)!"))


# In[10]:


df['review'] = df['review'].apply(preprocessor)


# In[11]:


def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')


# In[12]:


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('runners like running and thus they run')


# In[13]:


import nltk
nltk.download('stopwords')


# In[14]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]


# ## Training a logistic regression model for document classification

# In[15]:


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000: , 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# In[16]:


from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words':[stop, None],
               'vect__tokenizer':[tokenizer, tokenizer_porter],
               'clf__penalty':['l1','l2'],
               'clf__C':[1.0, 10.0, 100.0]},
              {'vect__ngram_range':[(1,1)],
               'vect__stop_words':[stop, None],
               'vect__tokenizer':[tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty':['l1','l2'],
               'clf__C':[1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)


# In[17]:


print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)


# In[18]:


print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# ## Working with bigger data - online alogrithms and out-of-core learning

# In[52]:


import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


# In[53]:


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) #skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# In[54]:


print(next(stream_docs(path='./movie_data.csv')))


# In[55]:


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# In[56]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21, 
                         preprocessor=None, 
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')


# In[57]:


import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


# In[36]:


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))


# In[58]:


clf = clf.partial_fit(X_test, y_test)


# In[ ]:




