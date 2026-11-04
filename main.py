import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

df = pd.read_csv("spam.csv",encoding="latin-1")

# keep only require columns
df = df[['v1','v2']]

# rename columna
df.columns = ['Categaries','Message']

# remove duplicaate message
df = df.drop_duplicates()

# remove duplicate value
df = df.dropna()

# see cleaned data
print(df.head())
print(df.shape)
print(df['Categaries'].value_counts())

## use matplot lib for visualization
# categaries_counts = df['Categaries'].value_counts()
# categaries_counts.plot(kind='bar')

# plt.xlabel('Categaries')
# plt.ylabel('not spam Message')
# plt.show()

# convert into numerical 
df['Categaries'] = df['Categaries'].map({
    'ham':0,
    'spam':1
})
print(df.head())
print(df['Categaries'].value_counts())

print(df['Message'].head(10))

# convert lower case 
df['Message'] = df['Message'].str.lower()
print(df['Message'].head(10))

# remove punctuation word
df['Message'] = df['Message'].apply(
    lambda test:test.translate(str.maketrans('', '', string.punctuation))
)
print(df['Message'].head(10))

# remove extra spaces
df['Message'] = df['Message'].str.strip()
df['Message'] = df['Message'].str.replace(r'\s+', ' ', regex=True)

print(df['Message'].head(10))

# use tf- idf
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['Message'])
y = df['Categaries']
print(X.shape)
print(y.shape)

# split the model
X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# use naive_bytes
model = MultinomialNB()
model.fit(X_train,y_train)
print("model tained successfully")

# make prediction
y_pred = model.predict(X_test)
print(y_pred[:20]) 

# compare
compairision = pd.DataFrame({
    "Actual":y_test,
    "predicted":y_pred
})
print(compairision.head(10))

# find accuracy score
accuracy = accuracy_score(y_test,y_pred)
print("accuracy",accuracy)
print(accuracy*100,"%")

# use classificaion
print(classification_report(
    y_test,
    y_pred,
    target_names = ['not spam','spam']


))

# use confusion metrix
cm = confusion_matrix(y_pred,y_test)
display = ConfusionMatrixDisplay(
  confusion_matrix=cm,
  display_labels=['not spam','spam']

)
display.plot()
plt.title("spam confusion metrix")
plt.show()

print(cm)

# to identify message
message = ["Congratulations! You have won a free prize. Claim now!"]
message_tfidf = tfidf.transform(message)
presiction = model.predict(message_tfidf)
if presiction[0] == 1:
    print(" spam")
else:
    print("not spam")


# store data in pkl
joblib.dump(model,"spam_model.pkl")  # to predict spam or not spam

# save the df idf 
joblib.dump(tfidf,"tfidf_model.pkl")  # convert text into numerical values



