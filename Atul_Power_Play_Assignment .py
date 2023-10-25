#!/usr/bin/env python
# coding: utf-8

# * Name : Atul Prajapati
# 
# * Task Submission Date: 21-10-2023
# 
# * Task: The goal is to identify patterns & insights from the dataset. Also, look out for insights/triggers that activated the users and subsequently engaged them. We understand this is very open-ended, and that's the point. Be creative.
# * Solution: We have two CSV files one of the having the information about the Events and on the other hand other file is having infomation about Org what project they are working on.

# ### Steps I followed 
# * Joined these two table via using EXCEL(XLOOKUP)
# * Cleaninig Process is done via EXCEL
# * I have Analysed the data using MySQL 

# In[25]:


#Importing the neccessary Library
import pandas as pd # For the Data cClaenini and Data Extraction
import numpy as np
import matplotlib.pyplot as plt # I am using This library for the Visualization of the data


# In[2]:


df = pd.read_csv("final_event.csv") #Reading the CSV File


# In[3]:


df.head() #Have look Top 10 Dataset


# In[4]:


df.describe() #Let's Describe the data or Statistics of the data


# #### Before Removing the Null values from the data we have 141 Null values

# In[5]:


df.isna().sum() #Let;s check the Null values in the Data and we got Null Values in project_id with 141 records are missing 


# In[6]:


# Step 1: Data Cleaning
df = df.dropna(subset=['project_id'])
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[7]:


# Assuming you have loaded your dataset into a DataFrame named 'final_event'
df['created_at_time'] = pd.to_datetime(df['created_at_time'])
df = df.sort_values(by='created_at_time')


# In[8]:


summary_stats = df.describe()
print(summary_stats)


# ###  Visualizing Data:
# Visualizing the data can help us understand its patterns. We'll create a time series plot to visualize how events are distributed over time.

# In[9]:


plt.figure(figsize=(12, 6))
plt.plot(df['created_at_time'], df['event'], 'b.', markersize=2)
plt.title('Event Distribution Over Time')
plt.xlabel('Date')
plt.ylabel('Event')
plt.grid(True)
plt.show()


# ### Time Series Decomposition:
# We can decompose the time series into its components (trend, seasonality, and residuals) to better understand underlying patterns.

# In[10]:


# Assuming 'final_event' is already sorted by 'created_at_time' as per the previous code
event_counts = df['event'].value_counts()
event_counts = event_counts.reset_index()
event_counts.columns = ['event', 'count']


# In[11]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(event_counts['count'], model='additive', period=1)
result.plot()
plt.show()


# In[12]:


event_counts = df['event'].value_counts()

# Plot the event counts
plt.figure(figsize=(12, 6))
event_counts.plot(kind='bar')
plt.title('Event Counts')
plt.xlabel('Event')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[13]:


# Set 'created_at_time' as the index
df.set_index('created_at_time', inplace=True)


# In[14]:


from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'event' column
df['event_encoded'] = label_encoder.fit_transform(df['event'])


# In[15]:


from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model
model = ARIMA(df['event_encoded'], order=(5, 1, 0))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Plot the residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()


# In[16]:


import seaborn as sns


# In[17]:


correlation_matrix = df.corr()


# In[18]:


unique_users = df['user_id'].nunique()
unique_orgs = df['org_id'].nunique()
unique_projects = df['project_id'].nunique()

print(f"Number of Unique Users: {unique_users}")
print(f"Number of Unique Organizations: {unique_orgs}")
print(f"Number of Unique Projects: {unique_projects}")


# In[19]:


top_10_orgs = df['org_id'].value_counts().head(10)

# Plot a bar chart to visualize the distribution of the top 10 organizations
plt.figure(figsize=(12, 8))
top_10_orgs.plot(kind='bar')
plt.title('Distribution of Top 10 Organizations')
plt.xlabel('Organization ID')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ### After the carefully looking at the graph i can make a conclusion that * ORG546271885436 is the most occured

# In[20]:


# Plot a bar chart to visualize the distribution of projects
project_counts = df['project_id'].value_counts().head(10)
plt.figure(figsize=(10, 6))
project_counts.plot(kind='bar')
plt.title('Distribution of Projects')
plt.xlabel('Project ID')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ### After the carefully looking at the graph i can make a conclusion that * PRJ256203650640 is most occuried

# In[21]:


# Create a dictionary to store unique users for each org_id
org_users = {}

# Group the dataset by org_id and collect unique user_ids
for org_id, group in df.groupby('org_id'):
    unique_users = group['user_id'].unique()
    org_users[org_id] = unique_users

# Display the unique users for each org_id
for org_id, users in org_users.items():
    print(f"Organization {org_id}: {', '.join(users)}")


# In[22]:


org_user_counts = df['org_id'].value_counts()

# Get the top 10 organizations with the most unique user IDs
top_10_orgs = org_user_counts.head(10)

# Plot a bar chart to visualize the top 10 organizations
plt.figure(figsize=(10, 6))
top_10_orgs.plot(kind='bar')
plt.title('Top 10 Organizations by Unique User Count')
plt.xlabel('Organization ID')
plt.ylabel('Unique User Count')
plt.xticks(rotation=45)
plt.show()


# In[23]:


# Count the number of unique projects for each organization
org_project_counts = df.groupby('org_id')['project_id'].nunique()

# Get the top 10 organizations with the most unique projects
top_10_orgs = org_project_counts.sort_values(ascending=False).head(10)

# Plot a bar chart to visualize the top 10 organizations
plt.figure(figsize=(10, 6))
top_10_orgs.plot(kind='bar')
plt.title('Top 10 Organizations by Unique Project Count')
plt.xlabel('Organization ID')
plt.ylabel('Unique Project Count')
plt.xticks(rotation=45)
plt.show()


# In[24]:


# Calculate event frequency
event_counts = df['event'].value_counts()

# Calculate user engagement (e.g., number of events per user)
user_engagement = df['user_id'].value_counts()

# Print the top event types and user engagement
print("Top Event Types:")
print(event_counts.head(10))

print("\nUser Engagement:")
print(user_engagement.head(10))


# ##### Top Event is: material_profile_material_load  = 10113
# ##### User Engagement: 9594

# ### Time Series Analysis

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is in a DataFrame named 'data'
# Convert 'created_at_time' to a datetime format
#data['created_at_time'] = pd.to_datetime(data['created_at_time'])

# Group data by 'created_at_time' and 'event'
event_counts = df.groupby(['created_at_time', 'event']).size().unstack().fillna(0)

# Create time series line plots for events of interest
events_of_interest = ['project_creation_request_success', 'new_material_added', 'generate_report_success']

plt.figure(figsize=(12, 6))
for event in events_of_interest:
    event_counts[event].rolling('30D').mean().plot(label=event)

plt.title('Event Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Event Frequency (Rolling 30-Day Average)')
plt.legend()
plt.show()


# In[29]:


# Resample data to a daily frequency and count events per day
daily_event_counts = df.resample('D').size()

# Create a time series plot
plt.figure(figsize=(12, 6))
daily_event_counts.plot(title='Daily Event Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Event Count')
plt.show()


# In[32]:


import pandas as pd

# Load data
data = pd.read_csv('final_event.csv')

# Convert 'created_at_time' to datetime
data['created_at_time'] = pd.to_datetime(data['created_at_time'])


# In[33]:


# Group data by month and count events
monthly_event_counts = data.resample('M', on='created_at_time')['event'].count()

# Create a line chart to visualize user engagement over time
plt.figure(figsize=(12, 6))
monthly_event_counts.plot(legend=False)
plt.title('Event Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Event Count')
plt.grid(True)
plt.show()


# In[34]:


# Filter and analyze specific event types
specific_event = 'project_creation_request_success'
event_data = data[data['event'] == specific_event]

# Visualize event frequency for the specific event type
event_counts = event_data.resample('M', on='created_at_time')['event'].count()

plt.figure(figsize=(12, 6))
event_counts.plot(legend=False)
plt.title(f'Event Frequency for {specific_event} Over Time')
plt.xlabel('Date')
plt.ylabel('Event Count')
plt.grid(True)
plt.show()

