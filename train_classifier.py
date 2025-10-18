import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pkl', 'rb'))

# sometimes error occurs if size of data is not matched.
filtered_data = []
filtered_labels = []

for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == 42:
        filtered_data.append(d)
        filtered_labels.append(l)
# for i, row in enumerate(data_dict['data']):
#     print(i, len(row))

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples are classified correctly'.format(score*100))

m = open('model.p', 'wb')
pickle.dump(model, m)
m.close()
