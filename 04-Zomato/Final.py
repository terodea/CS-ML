
# coding: utf-8

# In[1]:


geodictPune={"18.5074" :"73.8077","18.4874" :"74.1334","18.4669" :"73.8265","18.4898" :"73.8203",
              "18.4865" :"73.7968","18.5176" :"73.8417","18.4616" :"73.8505","18.5156" :"73.7819",
              "18.5416" :"73.8024","18.5018" :"73.8636","18.5167" :"73.8562","18.5699" :"73.8506",
              "18.5596" :"73.8171","18.4972" :"73.7960","18.6261" :"73.7390","18.5529" :"73.8796",
              "18.4848" :"73.8860","18.5636" :"73.9326","18.5679" :"73.9143","18.6084" :"73.7856",
              "18.5188" :"73.8303","18.5515" :"73.9348","18.5726" :"73.8782","18.5293" :"73.9100",
              "18.5042" :"73.9014","18.5089" :"73.9260","18.4829" :"73.9017","18.5789" :"73.7707",
              "18.5535" :"73.7547","19.1383" :"77.3210","18.4454" :"73.7801","18.4422" :"73.8096",
              "18.7381" :"73.6389","18.4667" :"73.7804","18.5229" :"73.7610","18.4966" :"73.9416",
              "18.4475" :"73.8232","18.4923" :"73.8547","18.5122" :"73.8860","18.5362" :"73.8940"}

geodictMumbai={"19.1136" :"72.8697","19.0607" :"72.8362","19.1465" :"72.9305","19.2372" :"72.8441",
                "18.9718" :"72.8436","18.9067" :"72.8147","18.5255" :"73.8795","18.9477" :"72.8342",
                "18.9703" :"72.8061","18.9127" :"72.8213","19.0328" :"72.8964","18.9322": "72.8264",
                "19.0213" :"72.8424","19.4946" :"72.8604","19.0075": "72.8360","18.9572": "72.8197",
                "19.1551" :"72.8679","18.9327" :"72.8316","19.1405": "72.8422","19.1106": "72.8326",
                "19.1998" :"72.8426","19.0717" :"72.8341","19.0600": "72.8900","19.0269" :"72.8553",
                "19.1802" :"72.8554","19.1726": "72.9425","15.9414": "77.4257","18.9256" :"72.8242",
                "18.9561" :"72.8157","19.0022": "72.8416","19.0158": "72.8280","19.0949" :"72.8865",
                "19.0843" :"72.8360","19.0390": "72.8619","19.0771": "72.9990","19.1351" :"72.8146",
                "19.1031" :"72.8467","19.0800": "72.8988","19.0912": "72.9209","18.9746" :"72.8065",
                "18.9872" :"72.8290"}

geodictBanglore={"12.970900":"77.604800","13.031359":"77.570240","13.004200":"77.604600",
                 "12.970900":"77.576300","12.991170":"77.585500","12.971270":"77.666940",
                 "13.184570":"77.479280","12.978680":"77.577130","12.957980":"77.605600",
                 "13.202540":"78.931260","12.649710":"77.200370","12.987940":"77.609590",
                 "12.972180":"77.586730","13.184570":"77.479280","12.980940":"77.586110",
                 "13.184570":"77.479280","13.026440":"77.909550","12.983590":"77.434850",
                 "12.707950":"77.746020","12.969290":"77.587760","13.184570":"77.479280",
                 "13.184570":"77.479280","12.869560":"74.866860","15.872160":"74.528570",
                 "13.184570":"77.479280","13.099670":"80.231290","12.476620":"76.765110"}

geodictDelhi={"28.8540":"77.0918","28.7535":"77.1948","28.7004":"77.2208","28.7193":"77.1736",
               "28.7324":"77.1442","28.7192":"77.1007","28.8055":"77.0463","28.6823":"77.0349",
               "28.6968":"77.0644","28.6959":"77.0805","28.6841":"77.0633","28.7495":"77.0565",
               "28.7164":"77.1546","28.6818":"77.1285","28.6780":"77.1581","28.7002":"77.1638",
               "28.7159":"77.1911","28.6570":"77.2122","28.6506":"77.2303","28.6486":"77.2340",
               "28.6139":"77.2090","28.5821":"77.2485","28.5778":"77.2244","28.5335":"77.2109",
               "28.5212":"77.1790","28.4959":"77.1848","28.4962":"77.2376","28.5374":"77.2597",
               "28.5049":"77.2739","28.5040":"77.3018","28.5603":"77.2913","28.6046":"77.3068",
               "28.6123":"77.3255","28.6200":"77.2924","28.6903":"77.2657","28.6890":"77.2815",
               "28.7034":"77.2840","30.1994":"77.1456","28.6388":"77.0738","28.6213":"77.0613",
               "28.5921":"77.0460","28.6094":"77.0543","28.6090":"76.9855","28.5349":"77.0558",
               "28.5901":"77.0888","28.5961":"77.1587","28.6415":"77.1209","28.6249":"77.1109",
               "28.6391":"77.0868","28.6219":"77.0878","28.6544":"77.1689","28.6623":"77.1411",
               "28.6721":"77.1205"}

geodictGoa={"15.5994":"73.8390","15.496777":"73.827827","15.6002":"73.8125",
            "15.5889":"73.9654","15.5959":"74.0594","15.7087":"73.8184",
            "15.5723":"73.8184","15.3991":"74.0124","15.3874":"73.8154",
            "15.2832":"73.9862","15.3841":"74.1181","14.9931":"74.0476",
            "15.2302":"74.1504"}

geodictKolkata={"22.5477":"88.3553","22.5176":"88.3840","22.5159":"88.3651",
                "22.4981":"88.3108","22.5867":"88.4171","22.6250":"88.4386",
                "22.5577":"88.3867","22.5975":"88.3707","22.5609":"88.3541",
                "22.5184":"88.3535","22.5170":"88.3658","22.5332":"88.3459",
                "22.4940":"88.3707","22.6218":"88.4180","22.5437":"88.3549",
                "22.5765":"88.4796"}

geodictIndore={"22.7533":"75.8937","22.7244":"75.8839","22.6928":"75.8684",
               "22.6400":"75.8040","22.7182":"75.8749","22.7143":"75.8687",
               "22.6709":"75.8275","22.7147":"75.8520","22.7198":"75.8571",
               "22.7066":"75.8770","22.7217":"75.8628","22.6980":"75.8683",
               "22.7368":"75.9086","22.6745":"75.8326"}

geodictHyderabad={"17.3984":"78.5583","17.4096":"78.5441","17.4265":"78.4511",
                  "17.4930":"78.4058","17.4615":"78.5004","17.5125":"78.3522",
                  "17.3807":"78.3245","17.542881":"78.481445","17.3930":"78.4730",
                  "17.3730":"78.5476","17.4375":"78.4483","17.4237":"78.4584",
                  "17.3990":"78.4153","17.4948":"78.3996","17.4447":"78.4664",
                  "17.4483":"78.3915","17.4622":"78.3568"}

geodictNorth={"29.3803":"79.4636","29.3780":"79.4662","29.3844":"79.4563",
              "29.3919":"79.4605",
              "27.0467":"88.2619","27.0615":"88.2765",
              "32.732998":"74.864273"}
              
geodictVijaywada={"16.5028":"80.6396","16.5218":"80.6091","16.5424":"80.5800",
                  "16.5179":"80.6507","16.5179":"80.6507","16.5129":"80.7020",
                  "16.5140":"80.6285","16.5226":"80.6672","16.4779":"80.7020",
                  "16.5515":"80.6521","16.5209":"80.6829","16.5209":"80.6829"}

geodictVisakhapatnam={"17.690474":"83.231049","17.7107":"83.3135","17.7425":"83.3389",
                      "17.7307":"83.3087","17.6881":"83.2131","17.7262":"83.3155",
                      "17.7409":"83.2493","17.7409":"83.2493","17.7447":"83.2319",
                      "17.7358":"83.2733","17.7384":"83.3015","17.9075":"83.4270",
                      "17.7742":"83.2319","17.8059":"83.2131"}

geodictNashik={"20.0059":"73.7934","19.997454":"73.789803","19.9469":"73.7654"}

geodictNagpur={"21.1397":"79.0631","21.1477":"79.0843","21.1358":"79.0765",
               "21.1313":"79.0800","21.1500":"79.1376","21.1491":"79.0550",
               "21.1557":"79.0942","21.1821":"79.0860","21.1856":"79.0805",
               "21.146633":"79.088860"}

geodictAurangabad={"19.8812":"75.3820","19.8757":"75.3442","19.901054":"75.352478",
                   "19.839911":"75.236237","19.9298":"75.3536"}

              
geoList=[geodictPune,geodictMumbai,geodictBanglore,geodictDelhi,geodictGoa,geodictKolkata,
         geodictIndore,geodictHyderabad,geodictNorth,geodictVijaywada,geodictVisakhapatnam,
         geodictNashik,geodictNagpur,geodictAurangabad]


# In[4]:


user_key_akshay ="874629c888aeff465f2b12f518379b20"
user_key_pratik ="5017afc6ac73011c60c8d23be5996a03"
import requests
import simplejson as json

header = {"User-agent": "curl/7.43.0", "Accept": "application/json", "user_key": "5017afc6ac73011c60c8d23be5996a03"}
nearby_restaurants = []
for i in geoList:
    for key,value in i.items():
        r = (requests.get("https://developers.zomato.com/api/v2.1/geocode?lat="+key+"&lon="+value, headers=header).content).decode("utf-8")
        a = json.loads(r)
        for nearby_restaurant in a['nearby_restaurants']:
            nearby_restaurants.append([nearby_restaurant['restaurant']['id'],
                             nearby_restaurant['restaurant']['name'],
                             nearby_restaurant['restaurant']['location']['country_id'],
                             nearby_restaurant['restaurant']['location']['city'],
                             nearby_restaurant['restaurant']['location']['address'],
                             nearby_restaurant['restaurant']['location']['locality'],
                             nearby_restaurant['restaurant']['location']['locality_verbose'],
                             nearby_restaurant['restaurant']['location']['longitude'],
                             nearby_restaurant['restaurant']['location']['latitude'],
                             nearby_restaurant['restaurant']['cuisines'],
                             nearby_restaurant['restaurant']['average_cost_for_two'],
                             nearby_restaurant['restaurant']['currency'],
                             nearby_restaurant['restaurant']['has_table_booking'],
                             nearby_restaurant['restaurant']['has_online_delivery'],
                             nearby_restaurant['restaurant']['is_delivering_now'],
                             nearby_restaurant['restaurant']['switch_to_order_menu'],
                             nearby_restaurant['restaurant']['price_range'],
                             nearby_restaurant['restaurant']['user_rating']['aggregate_rating'],
                             nearby_restaurant['restaurant']['user_rating']['rating_color'],
                             nearby_restaurant['restaurant']['user_rating']['rating_text'],
                             nearby_restaurant['restaurant']['user_rating']['votes']])
                            
nearby_restaurants


# In[37]:


import pandas as pd
dataset = pd.DataFrame(nearby_restaurants)
dataset.to_csv("/home/student/Desktop/CCEE_Final_Project/Dataset/Zomato.csv",sep='\t',encoding='utf-8', index=False)


# ## DUMPING TO Hive

# In[135]:


from pyhive import hive
from TCLIService.ttypes import TOperationState
import pandas as pd

cursor = hive.connect(host='localhost', port=10000, username='hiveuser').cursor()
cursor.execute("CREATE EXTERNAL TABLE IF NOT EXISTS cc(c_code int,country string)ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'")
cursor.execute("LOAD DATA LOCAL INPATH '/home/student/Desktop/CCEE_Final_Project/Dataset/country_code.csv' OVERWRITE INTO TABLE cc")
#cursor.execute("CREATE EXTERNAL TABLE IF NOT EXISTS zomato_project_3(Restaurant_ID string,Restaurant_Name string,Country_Code INT,City string,Address string,Locality string,Locality_Verbose string,Longitude string,Latitude string,Cuisines string,Average_Cost_for_two string,Currency string,Has_Table_booking string,Has_Online_delivery string,Is_delivering_now string,Switch_to_order_menu string,Price_range int,Aggregate_rating string,Rating_color string,Rating_text string,Votes string)ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'")
#cursor.execute("LOAD DATA LOCAL INPATH '/home/student/Desktop/CCEE_Final_Project/Dataset/Zomato.csv' OVERWRITE INTO TABLE zomato_project_3")


# In[51]:


import pandas as pd
conn = hive.Connection(host='localhost', port=10000, username='hiveuser')
df = pd.read_sql("select * from zomato_project_3", conn)


# In[52]:


df.head()


# In[53]:


df.rename(index=str, columns={"zomato_project_3.restaurant_id":"Restaurant ID",
                              "zomato_project_3.restaurant_name":"Restaurant Name",
                              "zomato_project_3.country_code":"Country Code",
                             "zomato_project_3.city":"City", 
                             "zomato_project_3.address":"Address",
                             "zomato_project_3.locality":"Locality", 
                             "zomato_project_3.locality_verbose":"Locality Verbose",
                             "zomato_project_3.longitude":"Longitude", 
                             "zomato_project_3.latitude":"Latitude", 
                             "zomato_project_3.cuisines":"Cuisines",
                             "zomato_project_3.average_cost_for_two":"Average Cost for two",
                             "zomato_project_3.currency":"Currency",
                             "zomato_project_3.has_table_booking":"Has Table booking",
                             "zomato_project_3.has_online_delivery":"Has Online delivery",
                             "zomato_project_3.is_delivering_now":"Is delivering now",
                             "zomato_project_3.switch_to_order_menu":"Switch to order menu",
                             "zomato_project_3.price_range":"Price range",
                             "zomato_project_3.aggregate_rating":"Aggregate rating",
                             "zomato_project_3.rating_color":"Rating color",
                             "zomato_project_3.rating_text":"Rating text",
                             "zomato_project_3.votes":"Votes"})
 


# In[54]:


df.to_csv("/home/student/Desktop/CCEE_Final_Project/Dataset/MLZomato.csv",sep='\t',encoding='utf-8',index=False)


# # ML Model Building

# ## MODEL 1 : RNN From Scratch
# ### Objective : Prediction Of Ratings.

# # Data Analysis For INDIAN City

# In[55]:


import pandas as pd
df = pd.read_csv('/home/student/Desktop/CCEE_Final_Project/Dataset/MLZomato.csv', encoding = 'utf-8',  sep='\t')
dfIndia = df[df['zomato_project_3.country_code']==1.0]
dfIndia = dfIndia.reset_index(drop=True) #resetting the indices
dfIndia.head(n=5)


# In[91]:


'''
Encoding the Data and dropping the 'Not rated rows'

cleanup = {'zomattest3.has_table_booking': {'Yes': 1, 'No': 0}, #Encoding Yes as 1 and No as 0
           'zomattest3.has_online_delivery': {'Yes': 1, 'No': 0},
           'zomattest3.is_delivering_now' : {'Yes': 1, 'No': 0},
           'zomattest3.rating_text' : {'Not rated': 0, 'Poor': 1, 'Average': 2, 'Good': 3, 'Very Good': 4, 'Excellent' : 5}}
dfIndia.replace(cleanup, inplace = True)
noRatng = dfIndia[dfIndia['zomattest3.rating_text']==0]
print(noRatng['zomattest3.rating_text'].count())

dfIndia = dfIndia[dfIndia['zomattest3.rating_text']!=0]
dfIndia.head()
'''


# In[56]:


'''
Calculating the number of 0s in the column, Avg. Cost of two and replacing it with the mean
'''
totalzero = (dfIndia['zomato_project_3.average_cost_for_two']== 0).sum()
print(totalzero)
n_sum = dfIndia['zomato_project_3.average_cost_for_two'].sum()
n_total = dfIndia['zomato_project_3.average_cost_for_two'].count()
print(n_sum/n_total)
cleanzero = {'zomato_project_3.average_cost_for_two': {0: 700}}
dfIndia.replace(cleanzero, inplace = True) 


# In[74]:


'''
We can take only the Main cuisine of the restaurant and do One-hot encoding on the different types of cusines.
'''
dftemp = pd.concat([dfIndia, dfIndia['zomato_project_3.cuisines'].str.split(',',expand=True)], axis = 1) #Expanding the different values in cuisines
dftemp = dftemp.rename(columns={0:'zomato_project_3.cuisines_1'}) #Renaming the Main Cusine as Cuisine 1
dftemp = dftemp.drop(1,axis =1) # Dropping all other Cuisine Values
dftemp = dftemp.drop(2,axis =1)
dftemp = dftemp.drop(3,axis =1)
dftemp = dftemp.drop(4,axis =1)
dftemp = dftemp.drop(5,axis =1)
dftemp = dftemp.drop(6,axis =1)
dftemp = dftemp.drop(7,axis =1)

dfClean = pd.get_dummies(dftemp, columns = ['zomato_project_3.cuisines']) #One-hot encoding
dfClean


# In[75]:


dfClean.columns


# In[76]:


'''
Dropping all the Unwanted Columns. Reordering the columns and saving it to a new CSV file.
'''
dfCleanIndia = dfClean.drop(['zomato_project_3.cuisines_1','zomato_project_3.country_code', 'zomato_project_3.rating_color', 'zomato_project_3.switch_to_order_menu', 'zomato_project_3.currency', 'zomato_project_3.address', 'zomato_project_3.locality', 'zomato_project_3.locality_verbose'],  axis =1)
cols = list(dfCleanIndia.columns.values)
cols.pop(cols.index('zomato_project_3.rating_text'))
cols.pop(cols.index('zomato_project_3.aggregate_rating'))
dfCleanIndia = dfCleanIndia[cols+['zomato_project_3.rating_text']+['zomato_project_3.aggregate_rating']]
dfCleanIndia


# In[77]:


print(dfCleanIndia['zomato_project_3.rating_text'].value_counts())


# In[78]:


def partition(x):
    if x=='Average':
        return '2'
    if x=='Good':
        return '3'
    if x=='Very Good':
        return '4'
    if x=='Excellent':
        return '5'
    else:
        return '1'

actualScore = dfCleanIndia['zomato_project_3.rating_text']
positiveNegative = actualScore.map(partition) 
dfCleanIndia['zomato_project_3.rating_text'] = positiveNegative


# In[79]:


dfCleanIndia.to_csv('/home/student/Desktop/CCEE_Final_Project/Dataset/ZomatoIndiaCleaned.csv', encoding = 'utf-8',  sep='\t', index=False)


# # ML Models On Cleaned DataSet

# In[80]:


import pandas as pd
df = pd.read_csv('/home/student/Desktop/CCEE_Final_Project/Dataset/ZomatoIndiaCleaned.csv', encoding = 'utf-8', sep = '\t')
df


# In[81]:


from sklearn.utils import shuffle
df = shuffle(df) #Shuffling the data
df.head()


# In[82]:


df.shape


# ## Let us build a Classification Model for Prediciting the rating of a restaurant.
# ### Target Value: Rating Text (Values: 1,2,3,4,5)

# In[83]:


'''
Let us Visualize how each feature is correlated to the target value

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cols = ['zomattest3.average_cost_for_two', 'zomattest3.has_table_booking', 'zomattest3.has_online_delivery',  'zomattest3.is_delivering_now', 'zomattest3.price_range', 'zomattest3.votes', 'zomattest3.rating_text',]
cor_matrix = np.corrcoef(df[cols].values.T) # We transpose to get the data by columns. Columns become rows.
sns.set(font_scale=1)
cor_heat_map = sns.heatmap(cor_matrix,
 cbar=True,
 annot=True,
 square=True,
 fmt='.2f',
 annot_kws={'size':10},
 yticklabels=cols,
 xticklabels=cols)
plt.show()
'''


# From the above Heat map, we can see that the Target Value (Rating text) is correlated well with the 'number of votes', 'price Range' and 'average cost for two', with correlation values being 0.42, 0.37 and 0.32 respectively. 
# 
# We can also draw many other inferences like, Average cost for two and price range has a very high correlation value of 0.83 which says that Average cost for two and price range are highly dependent on each other.

# In[137]:


x = df.iloc[:,6:914] #Features
y = df.iloc[:,915] #Target Value - Rating Text
y=y.fillna(0).astype(int) 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.4, random_state = 1) # Splitting the Data into Training and Test with test size as 20%


# ### Let us Start building Machine Learning Models and predict the accuracy score

# ### Model 2: Decision Tree Classification

# In[102]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_train = X_train.fillna(X_train.mean())
dtree_gini = DecisionTreeClassifier(criterion = "gini", random_state = 1, max_depth=5, min_samples_leaf=10)
dtree_gini.fit(X_train, y_train)
dtree_gini_pred = dtree_gini.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, dtree_gini_pred)*100)
cm = confusion_matrix(y_test, dtree_gini_pred)
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test,y_pred=dtree_gini_pred)) #Printing the Classification report to view precision, recall and f1 scores 


# ### Model 3: Decision Tree With Ada Boost

# In[103]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),
    n_estimators=60,
    learning_rate=1.5, algorithm="SAMME")
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
print('Decision Tree with Ada Boost Accuracy:', accuracy_score(y_test, ada_pred)*100)
print(classification_report(y_true=y_test,y_pred=ada_pred))


# ### Model 4: Random Forest Classification

# In[104]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, oob_score=True)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, rf_pred)
print('Random Forest Accuracy:', accuracy*100)
print('Random Forest Out-of-bag score estimate:', rf.oob_score_*100)
print(classification_report(y_true=y_test,y_pred=rf_pred))


# ### Model 5: K Nearest Neighbor Classification

# In[105]:


from sklearn import neighbors
knn=neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('K Nearest Neighbor Accuracy:', accuracy_score(y_test, knn_pred)*100)
print(classification_report(y_true=y_test,y_pred=knn_pred))


# ### Model 6: Artificial Neural Networks (Multi Layer Perceptron)

# In[106]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

nn_pred = mlp.predict(X_test)
score = accuracy_score(y_test, nn_pred)
print('Artificial Neural Network Accuracy:', score*100)
print(classification_report(y_true=y_test,y_pred=nn_pred))


# ### Model 6: XGBoost

# In[138]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Deep Learning from Scratch

# In[110]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
np.random.seed(1) # set a seed so that the results are consistent

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

# Function for setting the structure of Network
def layer_sizes(X, Y):
    
    n_x = X.shape[0]  #Input layer nodes  
    n_h = 4 #total hidden layer
    n_y = 6 #output layer nodes
    
    return (n_x, n_h, n_y)

# Initializing the parameters
def initialize_parameters(n_x, n_h, n_y):
        
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    
    assert (W1.shape == (n_h, n_x)) ##
    assert (b1.shape == (n_h, 1))  ## To confirm the shapes of Parameters
    assert (W2.shape == (n_y, n_h)) ## As these are the main reason of most errors
    assert (b2.shape == (n_y, 1))  ##
    
    parameters = {"W1": W1, ##
                  "b1": b1, ## Putting all parameters together so that
                  "W2": W2, ## we do not end up with confiusing due to many parameters
                  "b2": b2} ##
    
    return parameters


# forward propagation
def forward_propagation(X, parameters):
    
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
   
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    #assert(A2.shape == (6, X.shape[1]))
    
    cache = {"Z1": Z1,  
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    cost = -np.sum((Y * np.log(A2) + (1 - Y) * np.log(1 - A2)))/m
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

#backward Propagation
def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dz2 = A2 - Y
    dW2 = np.dot(dz2, A1.T)/m
    db2 = np.sum(dz2, axis = 1, keepdims = True)/m
    
    dz1 = np.dot(W2.T, dz2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dz1, X.T)/m
    db1 = np.sum(dz1, axis = 1, keepdims = True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):
    
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    dW2 = grads['dW2']
    db1 = grads['db1']
    db2 = grads['db2']    
    
    # Update rule for each parameter
    W1 = W1 - (learning_rate * dW1)
    W2 = W2 - (learning_rate * dW2)
    b1 = b1 - (learning_rate * db1)
    b2 = b2 - (learning_rate * db2)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=True):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


def predict(parameters, X):
    
    # Computes probabilities using forward propagation, and classifies.
    train_y, cache = forward_propagation(X, parameters)
    predicted_y = np.zeros_like(train_y)
    max_term = train_y.argmax(0)

    for i in range(X.shape[1]):
        predicted_y[max_term[i], i] = 1
    
    ratings = np.array(['Not rated', 'Poor', 'Average', 'Good', 'Very Good', 'Excellent'])
    predicted_ratings = np.zeros((1, X.shape[1]))
    predicted_ratings = pd.Series(ratings[predicted_y.argmax(0)])
    
    return predicted_y, predicted_ratings
    
    
def preprocessing(data):
    
    
    y = data['Rating text']
    X = data.copy()

    text_col = ['Restaurant Name', 'City', 'Address', 'Cuisines', 'Rating color', 'Rating text', 'Currency',
                'Locality Verbose', 'Locality']
    for i in text_col:
        del X[i]

    ratings = ['Not rated', 'Poor', 'Average', 'Good', 'Very Good', 'Excellent']

    print("Type of ratings : ", ratings)

    
    for i in range(len(ratings)):
        y = y.replace(to_replace = ratings[i], value= i)

    sm_column = ['Has Table booking', 'Has Online delivery', 'Switch to order menu', 'Is delivering now']
    
    
    for i in sm_column:
        X[i] = X[i].replace(to_replace = 'Yes', value = 1)
        X[i] = X[i].replace(to_replace = 'No', value = 0)

    
    #Feature scaling of data using
    scaler = StandardScaler()
    x_scaler = scaler.fit_transform(X)
    
    
    new_data = pd.DataFrame(x_scaler, columns= X.columns)
    new_data['Rating'] = y
    
    return new_data
    
    
def dataset(data):

    new_data = preprocessing(data)
    
# Splitting The training data and test data
    X = np.array(new_data.iloc[:, :new_data.shape[1] - 1]).T
    n, m = X.shape
    
    X_train = X[:, :6999]
    X_test = X[:, 7000: m-1]
    
    # encoding the data for multiclassification
    
    Y = np.zeros((6,6))
    diag = np.eye(6)
    y = np.array(new_data.iloc[:, new_data.shape[1] - 1], dtype = int)
    Y = diag[y].T
    
    Y_train = Y[:, :6999]
    Y_test = Y[:, 7000: m-1]
    
# Text data to match at the end and getting Accuracy
    y_text_train = data['Rating text'][:6999]
    y_text_train = y_text_train.values.reshape(y_text_train.shape[0], 1)

    y_text_test = data['Rating text'][7000: 9550]
    y_text_test = y_text_test.values.reshape(y_text_test.shape[0], 1)
    
    print(m, n)

    return X_train, Y_train, X_test, Y_test, y_text_train, y_text_test


# In[111]:


data = pd.read_csv('/home/student/Desktop/CCEE_Final_Project/ML_Algo/zomato2.csv')
X_train, Y_train, X_test, Y_test, y_text_train, y_text_test = dataset(data)
print("X_train shape : ", X_train.shape, type(X_train))
print("X_test shape : ", X_test.shape, type(X_test))
print("Y_train shape : ", Y_train.shape, type(X_train))
print("Y_test shape : ", Y_test.shape, type(X_test))
print("y_text_train shape : ", y_text_train.shape, type(X_train))
print("y_text_test shape : ", y_text_test.shape, type(X_test))


# In[112]:


# Train the parameters
parameters = nn_model(X_train, Y_train, n_h = 4, num_iterations = 10000, print_cost=True)


# In[113]:


predicted_y_train, predicted_ratings_train = predict(parameters, X_train)
predicted_ratings_train = predicted_ratings_train.values.reshape(predicted_ratings_train.shape[0], 1)

predicted_y_test, predicted_ratings_test = predict(parameters, X_test)
predicted_ratings_test = predicted_ratings_test.values.reshape(predicted_ratings_test.shape[0], 1)


# In[114]:


print('Accuracy on training set : ', np.mean(predicted_ratings_train == y_text_train) * 100)
print('Accuracy on training set : ', np.mean(predicted_ratings_test == y_text_test) * 100)


# In[116]:


new_data = pd.read_csv('/home/student/Desktop/CCEE_Final_Project/ML_Algo/new.csv', header = None)
X_new_data = np.array(new_data.iloc[:, :new_data.shape[1] - 1].T)
#real_y = np.array(new_data.iloc[:, new_data.shape[1] -1], dtype = str)


# # Prediction Of Ratings

# In[124]:


real_y = data['Rating text']
# Example of precdition
i = 2519
a, prediction = predict(parameters, X_new_data[:, i].reshape(12,1))
prediction = np.array(prediction)
print('prediction : ' + prediction)
print('real value : ' + real_y[i])


# # Conclusions:
# 
# 1. Data Analysis Showed that Rating is dependent on Average cost of two and also number of votes
# 2. Correlation Matrix showed that the Average cost for two and price range are highly dependent on each other
# 3. Correlation Matrix also showed that Rating text is correlated well with the 'number of votes', 'price Range' and 'average cost for two'
# 4. Of all the 5 Models, the best accuracy that we are getting is for the decision tree classifier with the accuracy of  around 67%. This means that the Decision Tree model is predicting nearly 67% of the test data accurately
# 5. Future works include: Hyper Parameter Tuning and using other boosting techniques like Xgboost
# 6. Can improve the accuracy by converting the data into a binary classification problem and combining it
