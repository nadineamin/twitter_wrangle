#!/usr/bin/env python
# coding: utf-8

# ## 1) Gathering Data

# In[185]:


#Importing the libraries needed
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import os
import json
import re
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[3]:


#Uploading the twitter-archive-enhanced.csv file and checking the data
twitter_df = pd.read_csv('twitter-archive-enhanced.csv')
twitter_df.head()

'''
#NOTE: DO NOT RUN THIS

import tweepy
from tweepy import OAuthHandler
from timeit import default_timer as timer

# Query Twitter API for each tweet in the Twitter archive and save JSON in a text file
# These are hidden to comply with Twitter's API terms and conditions
consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# NOTE TO REVIEWER: this student had mobile verification issues so the following
# Twitter API code was sent to this student from a Udacity instructor
# Tweet IDs for which to gather additional data via Twitter's API
tweet_ids = twitter_df.tweet_id.values
len(tweet_ids)

# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'w') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)
'''
# In[4]:


#Uploading the tweet_json.txt file and checking the data
json_file = []

with open('tweet-json.txt', 'r') as file:
    for line in file:
        tweet = json.loads(line)
        tweet_id = tweet['id']
        retweet_count = tweet['retweet_count']
        fav_count = tweet['favorite_count']
        json_file.append({'tweet_id':tweet_id,
                       'retweet_count': retweet_count,
                       'favorite_count': fav_count})
        
        
json_df = pd.DataFrame(json_file)
json_df.head()


# In[5]:


#Uploading the image-predictions.tsv file and checking the data
images_df = pd.read_csv('image-predictions.tsv', sep='\t')
images_df.head()


# ## 2) Assessing Data

# ### Visual Assessment:

# In `twitter-archived-enhanced.csv`:
# - Some dogs' names are "a", "actually", "all", "an", "by", "his", "life", "space", "such", "the", "this", "unacceptable" or "very". They should be converted to no name (I can check for any name that's lowercase and see if it's a valid dog name).
# - There are denominators that don't equal to 10 and some of them are wrong ratings.
# - Some numerators have exaggerated numbers.
# - Retweets and replies should be removed from the dataset.
# - Change "None" to null for columns doggo, floofer, pupper and puppo.
# - Doggo, floofer, pupper and puppo columns could be merged into one column. 
# 
# In `tweet-json.txt`:
# - The columns can be merged with twitter-archived-enhanced.csv via the tweet_id.
# 
# In `image-predictions.tsv`:
# - The columns can be merged with twitter-archived-enhanced.csv via the tweet_id.
# - Assign null values to p1, p1_conf, p2, p2_conf, p3 and p3_conf with values of p1_dog, p2_dog, p3_dog = False.
# - Remove p1_dog, p2_dog and p3_dog columns.
# - The dog type columns should be consistent in the format (all lowercase).
# - Change column names to be more representative.

# ### Programmatic Assessment:

# In[6]:


twitter_df.shape


# In[7]:


twitter_df.dtypes


# In[8]:


twitter_df.info()


# In[9]:


twitter_df.nunique()


# In[10]:


twitter_df.describe()


# In[11]:


json_df.shape


# In[12]:


json_df.dtypes


# In[13]:


json_df.info()


# In[14]:


json_df.nunique()


# In[15]:


images_df.shape


# In[16]:


images_df.dtypes


# In[17]:


images_df.info()


# In `twitter-archived-enhanced.csv`:
# - tweet_id should be converted to string.
# - timestamp and retweeted_status_timestamp should be converted to datetime.
# - There are two extra rows that are not in the other two files. Check if any id doesn't have a photo, likes or retweets.
# 
# In `tweet-json.txt`:
# - tweet_id should be converted to string.
# 
# In `image-predictions.tsv`:
# - tweet_id should be converted to string.

# ### 3) Cleaning Data

# In[18]:


#Create copies of the 3 DataFrames:
twitter_clean = twitter_df.copy()
json_clean = json_df.copy()
images_clean = images_df.copy()


# In[19]:


#Defining:
#Changing the datatype of tweet_id to string


# In[20]:


#Coding:
twitter_clean.tweet_id = twitter_clean.tweet_id.astype(str)
json_clean.tweet_id = json_clean.tweet_id.astype(str)
images_clean.tweet_id = images_clean.tweet_id.astype(str)


# In[21]:


#Testing:
twitter_clean.dtypes


# In[22]:


json_clean.dtypes


# In[23]:


images_clean.dtypes


# In[24]:


#Defining:
#Changing the datatype of timestamp and retweeted_status_timestamp to datetime:


# In[25]:


#Coding:
twitter_clean.timestamp = pd.to_datetime(twitter_clean.timestamp)
twitter_clean.retweeted_status_timestamp = pd.to_datetime(twitter_clean.retweeted_status_timestamp)


# In[26]:


#Testing:
twitter_clean.dtypes


# In[27]:


#Defining:
#Remove the retweets and replies by removing any rows with values other than 
#null in columns in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id,
#retweeted_status_user_id and retweeted_status_timestamp. Then delete these columns


# In[28]:


#Coding (deleting the rows):
twitter_clean = twitter_clean[twitter_clean.in_reply_to_status_id.isnull()]
twitter_clean = twitter_clean[twitter_clean.retweeted_status_id.isnull()]


# In[29]:


#Testing
twitter_clean.info()


# In[30]:


#Coding (deleting the columns):
twitter_clean = twitter_clean.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'], axis=1)


# In[31]:


#Testing
twitter_clean.info()


# In[32]:


#Defining:
#Change "None" to empty strings for columns doggo, floofer, pupper and puppo by using the replace method. Then after combining the four columns into one,changing the empty strings to null.


# In[33]:


#Coding:
twitter_clean.doggo = twitter_clean.doggo.replace("None","")
twitter_clean.floofer = twitter_clean.floofer.replace("None","")
twitter_clean.pupper = twitter_clean.pupper.replace("None","")
twitter_clean.puppo = twitter_clean.puppo.replace("None","")


# In[34]:


#Testing:
twitter_clean.doggo.value_counts()


# In[35]:


twitter_clean.floofer.value_counts()


# In[36]:


twitter_clean.pupper.value_counts()


# In[37]:


twitter_clean.puppo.value_counts()


# In[38]:


#Defining:
#Combine the doggo, floofer, pupper and puppo columns into one "classification" column


# In[39]:


#Coding:
twitter_clean['classification'] = twitter_clean.doggo + twitter_clean.floofer + twitter_clean.pupper + twitter_clean.puppo


# In[40]:


#Testing
twitter_clean.classification.value_counts()


# In[ ]:





# In[41]:


#Defining:
#Dropping the columns doggo, floofer, pupper and puppo as they have no purpose now


# In[42]:


#Coding:
twitter_clean = twitter_clean.drop(['doggo','floofer','pupper','puppo'], axis=1)


# In[43]:


#Testing:
twitter_clean.head(2)


# In[44]:


#Defining:
#Changing the blank strings to null in the classification column by using the replace method
#Changing the format of the dogs with several classifications by adding a "-" between the two strings


# In[45]:


#Coding:
twitter_clean.classification.replace(r'^\s*$', np.NaN, regex=True, inplace=True)
twitter_clean.classification.replace('doggopupper', 'doggo-pupper', inplace=True)
twitter_clean.classification.replace('doggopuppo', 'doggo-puppo', inplace=True)
twitter_clean.classification.replace('doggofloofer', 'doggo-floofer', inplace=True)


# In[46]:


#Testing:
twitter_clean.classification.value_counts()


# In[47]:


#Defining:
#Checking the wrong dog names (which are in lowercase) and "None" dog names and checking if the correct name is in the text. Otherwise, change dog name to null.


# In[48]:


#Coding:
reg = re.compile(r'(?:name(?:d)?)\s{1}(?:is\s)?([A-Za-z]+)')

for index, row in twitter_clean.iterrows():  
    if row['name'][0].islower() or row['name'] == 'None':
        try:
            name = re.findall(reg, row['text'])[0]
            twitter_clean.loc[index,'name'] = twitter_clean.loc[index,'name'].replace(row['name'], name)

        except IndexError:
            twitter_clean.loc[index,'name'] = np.NaN


# In[49]:


#Testing:
twitter_clean.name.value_counts()


# In[50]:


#Defining:
#Assign null values to p1, p1_conf, p2, p2_conf, p3 and p3_conf with values of p1_dog, p2_dog, p3_dog = False.


# In[51]:


#Coding:
images_clean.p1.replace(images_clean[images_clean.p1_dog == False].p1, np.NaN, inplace=True)
images_clean.p1_conf.replace(images_clean[images_clean.p1_dog == False].p1_conf, np.NaN, inplace=True)
images_clean.p2.replace(images_clean[images_clean.p2_dog == False].p2, np.NaN, inplace=True)
images_clean.p2_conf.replace(images_clean[images_clean.p2_dog == False].p2_conf, np.NaN, inplace=True)
images_clean.p3.replace(images_clean[images_clean.p3_dog == False].p3, np.NaN, inplace=True)
images_clean.p3_conf.replace(images_clean[images_clean.p3_dog == False].p3_conf, np.NaN, inplace=True)


# In[52]:


#Testing:
images_clean[images_clean.p1_dog == False].p1.value_counts()


# In[53]:


images_clean[images_clean.p1_dog == False].p1_conf.value_counts()


# In[54]:


images_clean[images_clean.p2_dog == False].p2.value_counts()


# In[55]:


images_clean[images_clean.p2_dog == False].p2_conf.value_counts()


# In[56]:


images_clean[images_clean.p3_dog == False].p3.value_counts()


# In[57]:


images_clean[images_clean.p3_dog == False].p3_conf.value_counts()


# In[58]:


images_clean.sample(10)


# In[59]:


#Defining:
#Removing p1_dog, p2_dog and p3_dog columns.


# In[60]:


#Coding:
images_clean = images_clean.drop(['p1_dog', 'p2_dog', 'p3_dog'], axis=1)


# In[61]:


#Testing:
images_clean.sample(5)


# In[62]:


#Defining:
#Unifying the format of p1, p2 and p3 to all lowercase characters.


# In[63]:


#Coding:
images_clean.p1 = images_clean.p1.str.lower()
images_clean.p2 = images_clean.p2.str.lower()
images_clean.p3 = images_clean.p3.str.lower()


# In[64]:


#Testing:
images_clean.sample(5)


# In[65]:


#Defining:
#Changing p1, p2 and p3 column names to prediction1, prediction2 and prediction3.
#Changing p1_conf, p2_conf and p3_conf to prediction1_conf, prediction2_conf and prediction3_conf.


# In[66]:


#Coding:
images_clean = images_clean.rename(columns={'p1':'prediction1', 'p2':'prediction2', 'p3':'prediction3', 
                                            'p1_conf':'prediction1_conf', 'p2_conf':'prediction2_conf', 'p3_conf':'prediction3_conf'})


# In[67]:


#Testing:
images_clean.head()


# In[68]:


#Defining:
#There are denominators that don't equal to 10 and some of them are wrong ratings
#These ratings could be changed manually


# In[69]:


#Coding:
twitter_clean.rating_denominator.value_counts()


# In[70]:


twitter_clean.rating_numerator.value_counts()


# In[71]:


twitter_clean[twitter_clean.rating_denominator != 10]


# In[72]:


twitter_clean.loc[433].text


# In[73]:


#Upon inspection, there are 7 puppies in the photo (that's why the rating is out of 70). To unify the format, I will divide the score by 7.
twitter_clean.rating_numerator = twitter_clean.rating_numerator.replace(84,12)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(70,10)


# In[74]:


twitter_clean.loc[516].text


# In[75]:


#There is no rating, so numerator and denominator should be set to 0.
twitter_clean.rating_numerator = twitter_clean.rating_numerator.replace(24,0)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(7,0)


# In[76]:


twitter_clean.loc[902].text


# In[77]:


#There are 15 dogs in the picture. Numerator and denominator should be divided by 15:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(165,11)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(150,10)


# In[78]:


twitter_clean.loc[1068].text


# In[79]:


#Since there are several occurrences of 9 as the numerator and 11 as the denominator, I used the loc method to index the specific row to change.
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(twitter_clean.loc[1068].rating_numerator,14)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1068].rating_denominator,10)


# In[80]:


twitter_clean.loc[1120].text


# In[81]:


#There are 17 dogs in the picture. Numerator and denominator should be divided by 17:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(204,12)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(170,10)


# In[82]:


twitter_clean.loc[1165].text


# In[83]:


twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(twitter_clean.loc[1165].rating_numerator,13)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(20,10)


# In[84]:


twitter_clean.loc[1202].text


# In[85]:


twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(50,11)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1202].rating_denominator,10)


# In[86]:


twitter_clean.loc[1228].text


# In[87]:


#There are 9 dogs in the picture. Numerator and denominator should be divided by 9:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(99,11)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(90,10)


# In[88]:


twitter_clean.loc[1254].text


# In[89]:


#There are 8 dogs in the picture. Numerator and denominator should be divided by 8:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(80,10)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1254].rating_denominator,10)


# In[90]:


twitter_clean.loc[1274].text


# In[91]:


#There are 5 dogs in the picture. Numerator and denominator should be divided by 5:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(45,9)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1274].rating_denominator,10)


# In[92]:


twitter_clean.loc[1351].text


# In[93]:


#There are 5 dogs in the picture. Numerator and denominator should be divided by 5:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(60,12)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1351].rating_denominator,10)


# In[94]:


twitter_clean.loc[1433].text


# In[95]:


#There are 4 dogs in the picture. Numerator and denominator should be divided by 4:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(44,11)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(40,10)


# In[96]:


twitter_clean.loc[1635].text


# In[97]:


#There are 11 dogs in the picture. Numerator and denominator should be divided by 11:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(121,11)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(110,10)


# In[98]:


twitter_clean.loc[1662].text


# In[99]:


twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(twitter_clean.loc[1662].rating_numerator,10)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1662].rating_denominator,10)


# In[100]:


twitter_clean.loc[1779].text


# In[101]:


#There are 12 dogs in the picture. Numerator and denominator should be divided by 12:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(144,12)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(120,10)


# In[102]:


twitter_clean.loc[1843].text


# In[103]:


#There are 8 dogs in the picture. Numerator and denominator should be divided by 8:
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(88,11)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(twitter_clean.loc[1843].rating_denominator,10)


# In[104]:


twitter_clean.loc[2335].text


# In[105]:


twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(twitter_clean.loc[2335].rating_numerator,9)
twitter_clean.rating_denominator = twitter_clean.rating_denominator.replace(2,10)


# In[106]:


#Testing
twitter_clean.rating_denominator.value_counts()


# In[107]:


twitter_clean.rating_numerator.value_counts()


# In[108]:


#Defining:
#Some numerators are still exaggerated (75, 1776, 26, 420 and 27)


# In[109]:


#Coding:
twitter_clean[twitter_clean.rating_numerator == 27]


# In[110]:


twitter_clean.loc[763].text


# In[111]:


#11.27 could be rounded to 11
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(27,11)


# In[112]:


twitter_clean[twitter_clean.rating_numerator == 75]


# In[113]:


twitter_clean.loc[695].text


# In[114]:


#9.75 could be rounded to 10
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(75,10)


# In[115]:


twitter_clean[twitter_clean.rating_numerator == 1776]


# In[116]:


twitter_clean.loc[979].text


# In[117]:


#No changes will be made in this case


# In[118]:


twitter_clean[twitter_clean.rating_numerator == 26]


# In[119]:


twitter_clean.loc[1712].text


# In[120]:


#11.26 could be rounded to 11
twitter_clean.rating_numerator  = twitter_clean.rating_numerator.replace(26,11)


# In[121]:


twitter_clean[twitter_clean.rating_numerator == 420]


# In[122]:


twitter_clean.loc[2074].text


# In[123]:


#No changes will be made in this case


# In[124]:


#Defining:
#Combining the 3 datasets into 1 dataset using tweet_id


# In[125]:


#Coding:
twitter_archive_master = pd.merge(twitter_clean, json_clean, on='tweet_id', how='left')
twitter_archive_master = pd.merge(twitter_archive_master, images_clean, on='tweet_id', how='left')


# In[126]:


#Testing:
twitter_archive_master.head()


# In[127]:


#Checking to see if any tweets don't have images
twitter_archive_master.info()


# In[128]:


#Defining:
#Removing any rows without images (jpg_url.isnull())


# In[129]:


#Coding:
twitter_archive_master = twitter_archive_master[twitter_archive_master.jpg_url.notnull()]


# In[130]:


#Testing:
twitter_archive_master.info()


# ### 4) Storing Data

# In[145]:


twitter_archive_master.to_csv('twitter_archive_master.csv')


# ### 5) Analyzing & Visualizing Data

# In[132]:


twitter_archive_master.head()


# In[133]:


twitter_archive_master.describe()


# In[186]:


plot_1 = twitter_archive_master['classification'].value_counts()
plot_1.plot(kind='bar', title='Dog Classification Count');
plt.xlabel('Dog Classification');
plt.ylabel('Count');


# Based on the above bar chart, "pupper" was the classification most used to describe the dogs, followed by doggo, puppo, etc.

# In[172]:


colors = ['blue','purple']
plot_2 = twitter_archive_master.groupby(['rating_numerator'])['retweet_count','favorite_count'].mean()
plot_2.plot(kind='bar', title='Average Retweets & Favorites by Numerator', color=colors)
plt.xlabel('Rating Numerator');
plt.ylabel('Count');


# In[158]:


#Getting the retweet and favorite ratios to create a more accurate plot.


# In[161]:


twitter_archive_master['retweet_ratio'] = twitter_archive_master['retweet_count'] / twitter_archive_master['retweet_count'].max()


# In[162]:


twitter_archive_master['favorite_ratio'] = twitter_archive_master['favorite_count'] / twitter_archive_master['favorite_count'].max()


# In[171]:


colors = ['blue','purple']
plot_3 = twitter_archive_master.groupby(['rating_numerator'])['retweet_ratio','favorite_ratio'].mean()
plot_3.plot(kind='bar', title='Average Retweets & Favorites by Numerator', color=colors)
plt.xlabel('Rating Numerator');
plt.ylabel('Ratio');


# In[134]:


twitter_archive_master.groupby('rating_numerator').mean()


# According to the chart and table above, the numerator with the greatest mean of favorite counts and retweet counts was 13.

# In[135]:


twitter_archive_master.groupby(twitter_archive_master['timestamp'].dt.year).mean()


# In[170]:


colors=['pink','orange','yellow']
plot_4 = twitter_archive_master.groupby(twitter_archive_master['timestamp'].dt.year)['rating_numerator'].mean()
plot_4.plot(kind='bar', title='Average Numerator by Year',color=colors)
plt.xlabel('Year');
plt.ylabel('Numerator');


# 2016 was the year with the highest mean value of the numerator.

# In[173]:


plot_5 = twitter_archive_master.groupby(twitter_archive_master['timestamp'].dt.year)['retweet_ratio','favorite_ratio'].mean()
plot_5.plot(kind='bar', title='Average Retweets & Favorites by Year',color=colors)
plt.xlabel('Year');
plt.ylabel('Ratio');


# 2017 was the year with the highes mean values of favorite counts and retweet counts.

# In[184]:


plot_6 = twitter_archive_master['prediction1'].value_counts()
plot_6[:10].plot(kind='barh', title='Dog Breed Count')
plt.xlabel('Dog Breed');
plt.ylabel('Count');


# In[ ]:





# In[ ]:




