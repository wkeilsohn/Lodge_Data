# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

# Personal Pet Peeve
import warnings
warnings.filterwarnings('ignore')

# Get Data
path = os.getcwd()
file = #[REDACTED]
data = os.path.join(path, file)

# Read Data
df = pd.read_csv(data, header=0, index_col=False)
# print(df.head()) # Just make sure the data loaded correctly. 

# Clean and Organize Data

## Drop Extranious Columns
"""
- Some columns in this data set tell the same story in multiple ways.
- Some columns are poorly populated and thus offer little value to the kinds of stats that can be run.
- Some are just irrelevant to the questions that we want to ask.
Removing these columns will make this process faster.
"""
df = df.drop(["suffix","masonic_suffix","nickname","spouse_email","spouse_phone","spouse_deceased_date","bylaws_signed_date","county","country","email","home_phone","mobile_phone",\
	"preferred_contact_method","job_type","job_title","lodge_publications","lodge_solicitations"], axis=1)

## Add Additional Columns
### Create A Lambda function to extract year from date

def get_year(date):
	return date[:4]

### Define Relevant Year Columns

today = datetime.date.today()
this_year = today.year

df['birth_year'] = df['birth_date'].apply(get_year)
df['ea_year'] = df['degree_initiated_ea_date'].astype(str).apply(get_year)
df['fc_year'] = df['degree_passed_fc_date'].astype(str).apply(get_year)
df['mm_year'] = df['degree_raised_mm_date'].astype(str).apply(get_year)
df['est_age'] = df.apply(lambda x: int(this_year) - int(x['birth_year']), axis=1)
df = df.dropna(axis=0, subset=['degree_initiated_ea_date', 'degree_passed_fc_date', 'degree_raised_mm_date']) # This drops 1 memeber who has no additional information.

## Extract Useful Variables
cols = list(df.columns)
nrows = len(df.index)
# print(cols) # Check Transformations

# Run Statistics
stats_df = pd.DataFrame()
mean_age = round(df['est_age'].mean(), 2)
std_age = round(df['est_age'].std(), 2)
age_range = [mean_age - std_age, mean_age, mean_age + std_age]
stats_df['Age_Range'] = age_range
print(stats_df)

'''
There is so much more you could do here, but I think this gets the point accross.
'''

# Create Plots

## Set Seaborn Parameters
sns.set(style="whitegrid")
sns.despine()

### Age Distribution of Lodge
plt.figure(figsize=(10, 10))
plt.plot()
sns.displot(data=df, x="est_age", binwidth=5, edgecolor="0", facecolor="0.5")
plt.title("Members By Age")
plt.ylabel("Number of Members")
plt.xlabel("Memeber Age")
plt.tight_layout()
plt.savefig("members_by_age_bar.png")

### Years Service By Age
plt.figure(figsize=(10, 10))
plt.plot()
sns.regplot(data=df, x="est_age", y="years_service", marker="x", color="0.2", line_kws=dict(color="r"))
plt.title("Members Service By Age")
plt.ylabel("Years of Service")
plt.xlabel("Memeber Age")
plt.savefig("service_by_age_scat.png")

### Years Since Each Degree
plt.figure()
plt.plot()
df['yr_mm'] = df.apply(lambda x: int(this_year) - int(x['mm_year']), axis=1)
df['yr_fc'] = df.apply(lambda x: int(this_year) - int(x['fc_year']), axis=1)
df['yr_ea'] = df.apply(lambda x: int(this_year) - int(x['ea_year']), axis=1)
fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 10))
sns.regplot(ax=axs[0], data=df, x='est_age', y='yr_ea')
sns.regplot(ax=axs[1], data=df, x='est_age', y='yr_fc')
sns.regplot(ax=axs[2], data=df, x='est_age', y='yr_mm')
axs[0].set_title('Entered Apprentice')
axs[1].set_title('Fellow Craft')
axs[2].set_title('Master Mason')
axs[0].set(xlabel=None, ylabel=None)
axs[1].set(xlabel=None, ylabel=None)
axs[2].set(xlabel=None, ylabel=None)
fig.suptitle("Years Since Degree By Memeber Age")
fig.supylabel("Age of Degree")
fig.supxlabel("Memeber Age")
plt.tight_layout()
plt.savefig("age_of_degree_scat.png")

### Type of Membership
plt.figure()
plt.plot()
membership_df = pd.DataFrame(df.groupby("paytype")['member_id'].nunique())
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
axs[0].pie(list(membership_df['member_id']), labels=list(membership_df.index), colors=sns.color_palette('Blues'), autopct='%1.1f%%')
membership_df = membership_df.T
axs[1].table(cellText=membership_df.values, colLabels=membership_df.columns, loc='center')
axs[1].axis('off')
fig.suptitle("Types of Membership")
plt.tight_layout()
plt.savefig("membership_types.png")


### States Memebers Live In
plt.figure()
plt.plot()
state_df = pd.DataFrame(df.groupby("state")['member_id'].nunique())
state_df['State'] = state_df.index
state_df['Is_DMV'] = state_df['State'].apply(lambda x: True if x in ['MD', 'VA', 'DC'] else False)
state_df['USA'] = state_df['State'].apply(lambda x: True if len(x) <= 2 else False)
state_df_dmv = state_df.groupby(['Is_DMV'], as_index=False)['member_id'].sum()
state_df_usa = state_df.groupby(['USA'], as_index=False)['member_id'].sum()
new_state_dic = {"MD": state_df.loc['MD','member_id'], 'VA':state_df.loc['VA','member_id'], 'DC':state_df.loc['DC','member_id'],\
'Non-DMV States':state_df_dmv.loc[0, 'member_id'] - state_df_usa.loc[0, 'member_id'], 'Non-USA':state_df_usa.loc[0, 'member_id']}
new_state_df = pd.DataFrame.from_dict(new_state_dic, orient='index', columns=['Members'])
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
axs[0].pie(list(new_state_df['Members']), labels=list(new_state_df.index), colors=sns.color_palette('pastel'), autopct='%1.1f%%')
new_state_df = new_state_df.T
axs[1].table(cellText=new_state_df.values, colLabels=new_state_df.columns, loc='center')
axs[1].axis('off')
fig.suptitle("Location of Members")
plt.tight_layout()
plt.savefig("where_memebers_live.png") 
