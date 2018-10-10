
# coding: utf-8

# In[148]:

#Libraries and Functions
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_KD_Tree (points,depth=1,k=2):
    n = len(points)
    
    if n <= 0:
        return None
    
    depth_axis = depth % k
    points = sorted(points, key=lambda point: point[depth_axis])
    
    return {
    'root':points[int (n/2)],
    'left':create_KD_Tree(points[:int(n/2)],depth+1),
    'right':create_KD_Tree(points[int(n/2) + 1:],depth+1)}

def haversine_dis (point1, point2):
    #earth's radius
    r = 6371
    lat_diff = math.radians(point2[0] - point1[0])
    long_diff = math.radians(point2[1] - point1[1])  

    a = math.sin(lat_diff/2) * math.sin(lat_diff/2) + math.cos(math.radians(point1[0])) * math.cos(math.radians(point2[0])) * math.sin(long_diff/2) * math.sin(long_diff/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    #print (r*c,point2)
    return (r*c)

def closest_point (user_point, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    if haversine_dis(user_point,p1) < haversine_dis(user_point,p2):
        return p1
    else:
        return p2

def nn_kdtree2 (user_loc,node,depth=1,k=2):
    
    depth_axis = depth % k

    next_branch = None
    other_branch = None

    if  user_loc[depth_axis] < node['root'][depth_axis]:
        next_branch = node['left']
        other_branch = node['right']
    else:
        next_branch = node['right']
        other_branch = node['left']

    
    if next_branch is None:
        cp = node['root']
    else:
        cp = nn_kdtree2(user_loc,next_branch,depth+1)
    
    #print(node['root'],cp)
    
    if haversine_dis(user_loc,cp) > haversine_dis(user_loc,node['root']):
        cp = node['root']
        cp = closest_point(user_loc,nn_kdtree2(user_loc,other_branch,depth+1),cp)
    
    return cp

def z_score_scaled (pop):
    pop_mean = np.mean(pop)
    pop_std = np.std(pop)
    ans = np.divide(np.subtract(pop,pop_mean),pop_std)
    
    #scaling
    old_range = np.max(pop) - np.min(pop)
    new_range = 10.0 - (-10.0)
    new_values = np.add(np.divide(np.multiply(np.subtract(pop,np.min(pop)),new_range),old_range),-10.0) 
    
    return new_values


# In[149]:


df_poi = pd.read_csv("//Users//saad//Saad//Jobs//Data Science//Assignment//ws-data-spark-master//data//POIList.csv")
df2_user = pd.read_csv("//Users//saad//Saad//Jobs//Data Science//Assignment//ws-data-spark-master//data//DataSample.csv")

df_poi.columns = ['poi_id','lat','long']

#df_user_no_id = df2_user[['TimeSt','Country','Province','City','Latitude','Longitude']]
df_user_no_id = df2_user.drop(['_ID'], axis=1)

#print (df_poi.dtypes)
#print (df2_user.dtypes)

df2_user_clean = df2_user.drop_duplicates([' TimeSt','Country','Province','City','Latitude','Longitude'])

print('length of clean dataframe: %d'%len(df2_user_clean))

a1 = np.array(df_poi['lat'])
a2 = np.array(df_poi['long'])
a3 = np.array(df_poi['poi_id'])
array_poi = np.vstack((a1, a2,a3)).T

a4 = np.array(df2_user_clean['Latitude'])
a5 = np.array(df2_user_clean['Longitude'])
array_userLoc = np.vstack((a4, a5)).T


# In[150]:


kdT = create_KD_Tree(array_poi)
print("\n\nk-d Tree: %s"%(kdT))


# In[151]:


pos_list = []
pos_dis = []

for point in array_userLoc:
    return_val = nn_kdtree2(point,kdT)
    pos_ans = return_val[2]
    dis_ans = haversine_dis(point,return_val[:2])
    pos_list.append(pos_ans)
    pos_dis.append(dis_ans)

df2_user_clean_copy = df2_user_clean.copy()
df2_user_clean_copy['closest_poi'] = pos_list
df2_user_clean_copy['dis_to_closest'] = pos_dis
df2_user_clean_copy = df2_user_clean_copy.replace('POI2','POI1')
#df2_user_clean_copy.head(10)

#print (df2_user_clean_copy['closest_poi'].unique())

df_poi_summary = df2_user_clean_copy['dis_to_closest'].groupby(df2_user_clean_copy['closest_poi']).describe()
df_poi_summary['density'] = df_poi_summary['count'] / (math.pi * pow(df_poi_summary['max'],2))
df_poi_summary.rename(columns={'max': 'radius_max'}, inplace=True)
print('\n\nResults Summary by POI: \n%s'%(df_poi_summary))


# In[152]:


ans = z_score_scaled(df_poi_summary['density'])
print('\n\nz score scaled: %s'%(ans))

