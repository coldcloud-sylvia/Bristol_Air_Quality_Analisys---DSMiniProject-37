#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[60]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from fancyimpute import KNN
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree


# In[4]:


file_path = 'D:/学习资料/DataScience/TB2/Mini-project/'
path_nonzipcode = 'process_data/X_variables/X_output/'
path_zipcode = 'process_data/zipcode_info/Zipcode Information/'
path_other = 'raw_data/Other_x_variables/Other Air Pollution Data/'

path_y = 'process_data/'


# In[6]:


#input data without coordinates
Meteo_df = pd.read_csv(file_path+path_nonzipcode+ 'Meteorology_df.csv')
EV_charge_df = pd.read_csv(file_path+path_nonzipcode+'ev_charging_points_df.csv')

#input data with zipcodes
Traffic_df = pd.read_csv(file_path+path_zipcode+'traffic.csv')
Covid_df = pd.read_csv(file_path+path_zipcode+'covid19-zipcode.csv')
Tree_df = pd.read_csv(file_path+path_zipcode+ 'tree_df-zipcode2.csv')
Tree_new_df = pd.read_csv(file_path+path_nonzipcode+ 'tree_new_df.csv')


#input:tree_data with air cleaning ability
tree_airclean_df = pd.read_csv(file_path+'process_data/X_variables/'+'tree_airclean_ability.csv') 
#TODO:input and clean other x variavles data(na an outliers)


#input AQI data
AQI_hour_df = pd.read_csv(file_path+path_y+'Hour_data_20230310.csv')
AQI_hour_locate_df = pd.read_csv(file_path+path_y+'Hour_locate_data-zipcode.csv')


# In[7]:


EV_charge_df=EV_charge_df.rename(columns={'postcode':'ZipCode'}).drop(['id'],axis=1)
Traffic_df=Traffic_df.rename(columns={'hour':'Hour','Zip Code':'ZipCode'})
for df in [AQI_hour_locate_df,Traffic_df,Covid_df,Tree_df,EV_charge_df]:
    df['left_ZipCode']=df['ZipCode'].str.split(' ',expand=True)[0]


# In[8]:


Meteo_df=Meteo_df.groupby(['Year','Month','Day','Hour']).mean().reset_index(drop=False)


# In[9]:


EV_charge_df = EV_charge_df[['Year','Month','Day','Hour','left_ZipCode','power_output']]
EV_charge_df=EV_charge_df.groupby(['Year','Month','Day','Hour','left_ZipCode']).mean().reset_index(drop=False)
EV_charge_df.head(5)


# In[10]:


print(EV_charge_df.shape)


# In[11]:


Traffic_df=Traffic_df[['Year','Month','Day','Hour','left_ZipCode','Hourly Flow','Hourly Profile','Link']]
print(Traffic_df.shape)
Traffic_df.head(5)


# In[12]:


Traffic_df=Traffic_df.groupby(['Year','Month','Day','Hour','left_ZipCode']).mean().reset_index(drop=False)


# In[13]:


Traffic_df.head(20)


# In[14]:


Covid_df['Year'] = pd.to_datetime(Covid_df['Specimen date'],format='%d/%m/%Y').dt.year
Covid_df['Month'] = pd.to_datetime(Covid_df['Specimen date'],format='%d/%m/%Y').dt.month
Covid_df['Day'] = pd.to_datetime(Covid_df['Specimen date'],format='%d/%m/%Y').dt.day
Covid_df = Covid_df[['Year', 'Month','Day','left_ZipCode','Daily lab-confirmed cases', 'Cumulative lab-confirmed cases', 'Cumulative lab-confirmed cases rate']]
Covid_df=Covid_df.groupby(['Year','Month','Day']).mean().reset_index(drop=False)
Covid_df.head(5)


# In[15]:


#join_df=pd.merge(AQI_hour_locate_df[['Year','Month','Day']],Covid_df[['Year','Month','Day','Cumulative lab-confirmed cases']],how='left',on=['Year','Month','Day'] )
#join_df['Cumulative lab-confirmed cases'].isna().sum()
#Covid_df.left_ZipCode.unique()


# In[16]:


AQI_hour_locate_df.left_ZipCode.unique()


# In[17]:


Tree_df = Tree_df[['Year','Month','Day','Hour','left_ZipCode','Plot number','OBJECTID']]
Tree_df=Tree_df.groupby(['Year','Month','Day','Hour','left_ZipCode']).mean().reset_index(drop=False)
Tree_df.head(5)


# In[18]:


Tree_new_df = Tree_new_df.merge(Tree_df[['Year','Month','Day','Hour','left_ZipCode']],on=['Year','Month','Day','Hour'],how='left')
Tree_new_df = Tree_new_df[['Year','Month','Day','Hour','left_ZipCode','Site name','Latin name','Plot number']]
Tree_new_df.head(10)


# In[19]:


#Encode the tree types
le = LabelEncoder()
cols_to_transform = ['Site name','Latin name','Plot number']

Tree_new_df[cols_to_transform] = Tree_new_df[cols_to_transform].fillna(0) 
Tree_new_df[cols_to_transform] = Tree_new_df[cols_to_transform].apply(lambda x: le.fit_transform(x.astype(str)))


# In[53]:


Tree_new_df


# Tree_airclean_df append

# In[25]:


tree_airclean_df=tree_airclean_df.drop(['Air clean bility'],axis=1)
tree_airclean_df


# In[35]:


#rename columns
new_col_names = {'zipcode':'left_ZipCode','0': 'tree_airclean_0', '1': 'tree_airclean_1', '2': 'tree_airclean_2'}
tree_airclean_df = tree_airclean_df.rename(columns=new_col_names)
tree_airclean_df


# In[54]:


AQI_hour_locate_df=AQI_hour_locate_df[['Year','Month','Day','Hour','left_ZipCode','max_AQI']]
AQI_hour_locate_df=AQI_hour_locate_df[AQI_hour_locate_df['Year'] > 2018]
print(AQI_hour_locate_df['Year'].unique())
print(AQI_hour_locate_df.shape)
AQI_hour_locate_df.head(5)


# There exists no inner join outcome between AQI and covid ZipCode so Drop the zip code when doing merge.

# In[36]:


data = pd.merge(AQI_hour_locate_df, Traffic_df, how='left',on=['Year','Month','Day','Hour','left_ZipCode'])
print('Merged with Traffic:',data.shape,data.isna().sum())
data = data.merge(Meteo_df,on=['Year','Month','Day','Hour'],how='left')
print('Merged with Meteorology:',data.shape,data.isna().sum())
# There exists no inner join outcome between AQI and covid ZipCode so Drop the zip code when doing merge.
data = data.merge(Covid_df[['Year', 'Month','Day','Daily lab-confirmed cases', 'Cumulative lab-confirmed cases', 'Cumulative lab-confirmed cases rate']],on=['Year','Month','Day'],how='left')
print('Merged with Covid:',data.shape,data.isna().sum())
data = data.merge(tree_airclean_df,on=['Year','Month','Day','Hour','left_ZipCode'],how='left')
print('Merged with Tree：',data.shape,data.isna().sum())

#data = data.merge(EV_charge_df,on=['Year','Month','Day','Hour','left_ZipCode'],how='left')
#print('Merged with Tree：',data.shape,data.isna().sum())


# In[48]:


data=data.drop(columns=['Unnamed: 0','Type of tree'])


# In[55]:


data


# In[57]:


from sklearn.preprocessing import StandardScaler

# Find the columns that need scaling
scaling_columns = list(set(data.columns) - set(['Year', 'Month', 'Day', 'Hour', 'max_AQI','geo_point_2d', 'ZipCode','left_ZipCode']))

# Function for mean imputation
def fill_na_mean(data, columns):
    data_copy = data.copy()
    # Group by Year, Month, Day, and left_ZipCode, then fill NAs with the mean of the group
    data_copy[columns] = data_copy.groupby(['Year', 'Month', 'Day', 'left_ZipCode'])[columns].transform(lambda x: x.fillna(x.mean()))
    return data_copy

# Function for linear regression imputation
def fill_na_regression(data, columns):
    data_copy = data.copy()
    for col in columns:
        lr = LinearRegression()
        data_col_notnull = data_copy[data_copy[col].notnull()]
        data_col_null = data_copy[data_copy[col].isnull()]
        lr.fit(data_col_notnull[['Year', 'Month', 'Day', 'left_ZipCode']], data_col_notnull[col])
        data_copy.loc[data_col_null.index, col] = lr.predict(data_col_null[['Year', 'Month', 'Day', 'left_ZipCode']])
    return data_copy

# Perform mean imputation
data_filled_mean = fill_na_mean(data, scaling_columns)

# Perform linear regression imputation
data_filled_regression = fill_na_regression(data_filled_mean, scaling_columns)

# Scaling
scaler = StandardScaler()
data_scaled = data_filled_regression.copy()
data_scaled[scaling_columns] = scaler.fit_transform(data_filled_regression[scaling_columns])


data


# In[68]:


scaling_columns = list(set(data.columns) - set(['Year', 'Month', 'Day', 'Hour', 'max_AQI','geo_point_2d', 'ZipCode','left_ZipCode']))
data_filled_regression = fill_na_regression(data, scaling_columns)
#scaling_columns


# In[63]:


'''
# 按需要填充的列进行数据切片
scaling_columns = list(set(data.columns) - set(['Year', 'Month', 'Day', 'Hour', 'max_AQI','geo_point_2d', 'ZipCode','left_ZipCode']))
data_to_fill = data[scaling_columns]

# 进行KNN填充
filled_data = KNN(k=5).fit_transform(data_to_fill)

# 将填充后的数据放回原数据框
data[scaling_columns] = filled_data
'''


# Output the data for modeling

# In[39]:


data['left_ZipCode'] = data['left_ZipCode'].astype(str)
file='D:/学习资料/DataScience/TB2/Mini-project/model/'
data.to_csv(file+'AQI_X_data.csv', index=False)


# # Correlation_matrix

# In[46]:


#test_data=data
test_data=data[data['Year']==2022].reset_index(drop=True)
df_coor=test_data.corr()
df_coor.head()


# In[47]:


plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
fig=sns.heatmap(df_coor,annot=True, vmax=1, square=True, cmap="GnBu", fmt='.1f',annot_kws={"size": 9})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
fig


# In[59]:


fig.get_figure().savefig('df_corr.png',bbox_inches='tight',transparent=True)#保存图片
#bbox_inches让图片显示完整，transparent=True让图片背景透明


# In[80]:


test_data


# In[42]:


'''
import causalinference as cf


# 创建因果推断模型
causal = cf.CausalModel(test_data, treatment='left_ZipCode', outcome='max_AQI', common_causes=['Year', 'Month', 'Day', 'Hour'])

# 运行因果推断
causal.est_via_ols()
causal.est_via_matching()
causal.est_propensity_s()
causal.est_via_weighting()

# 查看因果效应
print(causal.summary())
'''


# In[43]:


'''
import pandas as pd
import dowhy.api

# 加载数据
test_data = pd.read_csv('test_data.csv')

# 创建一个因果模型
model = dowhy.api.Model(data=test_data,
                        treatment='left_ZipCode',
                        outcome='max_AQI',
                        common_causes=['Year', 'Month', 'Day', 'Hour', 'Hourly Flow', 'Hourly Profile', 'Temperature',
                                       'Dewpoint Temperature', 'Wind Speed', 'Wind Chill Temperature', 'Relative Humidity',
                                       'Daily lab-confirmed cases', 'Cumulative lab-confirmed cases',
                                       'Cumulative lab-confirmed cases rate', 'Site name', 'Latin name', 'Plot number'])

# 运行因果推断
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")

# 查看因果效应
print(estimate)  
'''


# # Try to apply LR and XGBoots

#  using 2021 data

# In[60]:


le = LabelEncoder()

cols_to_transform = ['left_ZipCode', 'Plot number']

data_21['left_ZipCode'] = data_21['left_ZipCode'].astype(str)

data_21[cols_to_transform] = data_21[cols_to_transform].apply(lambda x: le.fit_transform(x.astype(str)))
data_21=data_21.fillna(0)


# In[61]:


X = data_21
X=X.drop(['max_AQI'], axis=1)
y = data_21['max_AQI']


# In[62]:


X


# In[63]:


xgb_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
# Fit the Model
xgb_model.fit(X, y)


# In[64]:


from matplotlib import pyplot
from xgboost import plot_importance
fig2=plot_importance(xgb_model)
pyplot.show()


# In[65]:


fig2.get_figure().savefig('feature_inportance.png',bbox_inches='tight',transparent=True)#保存图片
#bbox_inches让图片显示完整，transparent=True让图片背景透明


# using 2019-2022 data

# In[66]:


data['left_ZipCode'] = data['left_ZipCode'].astype(str)
data_22=data[data['Year']==2022].reset_index(drop=True)

test_data=data_22
df_coor_22=test_data.corr()
df_coor_22.head()


# In[67]:


plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
fig=sns.heatmap(df_coor_22,annot=True, vmax=1, square=True, cmap="GnBu", fmt='.1f',annot_kws={"size": 9})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
fig


# In[68]:


le = LabelEncoder()

cols_to_transform = ['left_ZipCode', 'Plot number']

data_22['left_ZipCode'] = data_22['left_ZipCode'].astype(str)

data_22[cols_to_transform] = data_22[cols_to_transform].apply(lambda x: le.fit_transform(x.astype(str)))
data_22=data_22.fillna(0)


# In[69]:


X = data_22
X=X.drop(['max_AQI'], axis=1)
y = data_22['max_AQI']
xgb_model.fit(X, y)


# In[70]:


fig2=plot_importance(xgb_model)
pyplot.show()


# In[ ]:




