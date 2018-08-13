#%%
from sklearn.feature_extraction.text import CountVectorizer

text = ['Hello world', 'Hello friend', 'Hello enemy']
print("Original text is\n{}".format('\n'.join(text)))

vectorizer = CountVectorizer(min_df=0)

# call `fit` to build the vocabulary
vectorizer.fit(text)

# call `transform` to convert text to a bag of words
x = vectorizer.transform(text)

# CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
# convert back to a "normal" numpy array
x = x.toarray()

print("")
print("Transformed text vector is \n{}".format(x))

# `get_feature_names` tracks which word is associated with each column of the transformed x
print("")
print("Words for each feature:")
print(vectorizer.get_feature_names())

# Notice that the bag of words treatment doesn't preserve information about the *order* of words, 
# just their frequency
\
#%%

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

critics = pd.read_csv('critics.csv')

# call `fit_transform' to build the vocabulary and convert text to a bag of words
X = vectorizer.fit_transform(critics.quote)

# convert back to a "normal" numpy array
X = X.toarray()

print("")
print("Transformed text vector is \n{}".format(X))


# `get_feature_names` tracks which word is associated with each column of the transformed x
print("")
print("Words for each feature:")
print(vectorizer.get_feature_names())

