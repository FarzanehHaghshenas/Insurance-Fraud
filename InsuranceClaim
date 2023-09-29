import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('Datasets/claims/insurance_claims.csv')

feats = [‘policy_state’,’insured_sex’,’insured_education_level’,’insured_occupation’,’insured_hobbies’,’insured_relationship’,’collision_type’,’incident_severity’,’authorities_contacted’,’incident_state’,’incident_city’,’incident_location’,’property_damage’,’police_report_available’,’auto_make’,’auto_model’,’fraud_reported’,’incident_type’]
df_final = pd.get_dummies(df,columns=feats,drop_first=True)

X = df_final.drop([‘fraud_reported_Y’,’policy_csl’,’policy_bind_date’,’incident_date’],axis=1).values
y = df_final[‘fraud_reported_Y’].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classsifier.add(
        Dense(3, kernel_initializer = ‘uniform’,
              activation = ‘relu’, input_dim=5))
classifier.add(
     Dense(1, kernel_initializer = ‘uniform’,
           activation = ‘sigmoid’))
classifier.compile(optimizer= ‘adam’,
                  loss = ‘binary_crossentropy’,
                  metrics = [‘accuracy’])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



cm = confusion_matrix(y_test, y_pred)
