#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[8]:


data = pd.read_csv("./Cristano_Ronaldo_Final_v1/data.csv")


# In[9]:


data.head()


# In[6]:


data.describe()


# In[4]:


data.info()


# In[156]:


# data.info()


# In[157]:


# data_features_selected = ['location_x','location_y']
##testing uniqueness of labels

# unique_power_of_shot=np.unique(np.asarray(data["power_of_shot"]),return_counts=True)
# print(unique_power_of_shot)

def plot_power(data):
    power = (data['area_of_shot'])#*60+(data['remaining_sec']))#.dropna())#.values.astype(int)
    plt.hist(power)
    plt.title("remaining_min.1  distribution")


# In[ ]:


#plot_power(data)


# In[5]:



(data['range_of_shot'].unique())


# ##### data preparation

# In[6]:


le = LabelEncoder()


# In[7]:


ans = data['is_goal']
data['is_goal']=data['is_goal'].fillna(0)

data['location_x']=data['location_x'].fillna(np.mean(data['location_x']))
data['location_x'] = (data['location_x']-np.mean(data['location_x']))/np.std(data['location_x'])

data['location_y']=data['location_y'].fillna(np.mean(data['location_y']))
data['location_y'] = (data['location_y']-np.mean(data['location_y']))/np.std(data['location_y'])

data['remaining_sec']=data['remaining_min']*60+(data['remaining_sec'])
data['remaining_sec']=data['remaining_sec'].fillna(np.mean(data['remaining_sec']))
data['remaining_sec'] = (data['remaining_sec']-np.mean(data['remaining_sec']))/np.std(data['remaining_sec'])

data['power_of_shot']=data['power_of_shot'].fillna(np.mean(data['power_of_shot']))
data['power_of_shot'] = (data['power_of_shot']-np.mean(data['power_of_shot']))/np.std(data['power_of_shot'])

data['knockout_match']=data['knockout_match'] #0 1  encodeded

data['distance_of_shot']=data['distance_of_shot'].fillna(np.mean(data['distance_of_shot']))
data['distance_of_shot'] = (data['distance_of_shot']-np.mean(data['distance_of_shot']))/np.std(data['distance_of_shot'])





# In[10]:


data.info()


# In[8]:


#data['area_of_shot']=data['area_of_shot'].fillna(np.mean(data['area_of_shot']))
np.unique(np.asarray(data['area_of_shot'].dropna()),return_counts=True)
data['area_of_shot']=data['area_of_shot'].fillna('Center(C)')
data['area_of_shot'] =  le.fit_transform(data["area_of_shot"])#label encoder

np.unique(np.asarray(data['shot_basics'].dropna()),return_counts=True)
data['shot_basics']=data['shot_basics'].fillna('Mid Range')
data['shot_basics'] =  le.fit_transform(data["shot_basics"])#label encoder
##nan not handeled

#data['shot_basics']=data['shot_basics'].fillna('Mid Range')

data['range_of_shot']=data['range_of_shot'].fillna('Less Than 8 ft')

data['range_of_shot'] =  le.fit_transform(data["range_of_shot"])#label encoder



data['knockout_match']=data['knockout_match'].fillna(0)

data['knockout_match'] =  le.fit_transform(data["knockout_match"])#label encoder



# In[9]:


np.unique(np.asarray(data['range_of_shot'].dropna()),return_counts=True)


# In[10]:


np.sum([6220, 4751, 5076,   70, 7064])


# In[11]:


#df = data.dropna(subset=['area_of_shot','shot_basics','range_of_shot','knockout_match'],inplace=True)


# In[12]:


data.info()


# In[13]:


df = data[['location_x','location_y','remaining_sec','power_of_shot','knockout_match','distance_of_shot','area_of_shot','shot_basics']]


# In[14]:


df.info()


# In[192]:


df.head()


# In[193]:


y = np.asarray(data['is_goal'])
y.shape


# In[194]:


X = np.asarray(df)
Y = y


# #### making model

# In[195]:


from keras.models import Sequential
from keras.layers import Dense
#X = X[:,:-3]
X.shape


# In[177]:


classifier = Sequential()
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=5))
#Second  Hidden Layer
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


# In[178]:


classifier.summary()


# In[179]:


classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


# In[181]:


#classifier.fit(X,Y, batch_size=10, epochs=100,validation_split=0.2)


# In[101]:


classifier.evaluate(X,Y)


# In[102]:


X[0]


# In[273]:


from sklearn.ensemble import RandomForestClassifier


# In[274]:


rf = RandomForestClassifier(criterion='entropy')


# In[275]:


rf.fit(X, Y)


# In[276]:


rf.score(X,Y)


# In[277]:


print(rf.predict_log_proba(X))


# In[278]:


print(rf.predict_proba(X))
print(rf.classes_)


# In[279]:


sample = pd.read_csv("./Cristano_Ronaldo_Final_v1/sample_submission.csv")


# In[280]:


x = np.asarray(sample['shot_id_number'])
x[0]


# In[282]:


allans = np.asarray(skdt.predict_proba(X))
import csv
with open('Sample_Submission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    m = x.shape[0]
    writer.writerow(["shot_id_number","is_goal"])
    for i in range(m):
        l = [x[i],allans[(x[i]-1)][1]]
        writer.writerow(l)
    csvFile.close()


# In[224]:


np.asarray(datatotal)[0]


# In[212]:


np.unique(rf.predict(X),return_counts=True)


# In[234]:


allans[17]


# In[215]:


from sklearn.tree import DecisionTreeClassifier


# In[216]:


skdt = DecisionTreeClassifier(criterion='entropy')


# In[217]:


skdt.fit(X, Y)


# In[262]:


skdt.score(X,Y)


# In[265]:


print(skdt.predict_proba(X))


# In[ ]:





# In[ ]:





# In[257]:


allans[32]


# In[260]:


allans = np.asarray(data['is_goal'])#skdt.predict_proba(X)
import csv
with open('Sample_Submission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    m = x.shape[0]
    writer.writerow(["shot_id_number","is_goal"])
    for i in range(m):
        if(allans[(x[i]-1)]==0):
            l = [x[i],0.5]#allans[(x[i]-1)]]
        else:
            l = [x[i],0.75]#allans[(x[i]-1)]]
        print(x[i],allans[(x[i]-1)])
        writer.writerow(l)
    csvFile.close()


# In[ ]:




