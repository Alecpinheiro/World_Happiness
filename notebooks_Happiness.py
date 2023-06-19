#!/usr/bin/env python
# coding: utf-8

# In[89]:


#installing the needed packages and tools
get_ipython().run_line_magic('pip', 'install seaborn')
get_ipython().run_line_magic('pip', 'install plotly==5.14.1')
get_ipython().run_line_magic('pip', 'install ipykernel')
get_ipython().run_line_magic('pip', 'install nbformat==5.1.2')
get_ipython().run_line_magic('pip', 'install openpyxl')
get_ipython().run_line_magic('pip', 'show nbformat')


# In[90]:


#importing the needed packages
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import geopandas as gpd
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr


# In[91]:


# importing data
df = pd.read_csv('world-happiness-report-2021.csv')
df2 = pd.read_csv('world-happiness-report.csv')
pop = pd.read_csv('2021_population.csv')
hofstede = pd.read_excel('hofstede_full.xlsx')

safety = df.copy()

# renaming columns for easier interpretation and merge later
df = df.rename(columns={'Country name': 'Country'})
df2 = df2.rename(columns={'Country name': 'Country'})
pop = pop.rename(columns={'country': 'Country'})
pop = pop.rename(columns={'2021_last_updated': 'Population in 2021'})
pop = pop.rename(columns={'growth_rate': 'Growth rate'})
df = df.rename(columns={'Ladder score': 'Happiness score'})
df2 = df2.rename(columns={'Life Ladder': 'Happiness score'})

# editing pop column for merge later
pop['Population in 2021'] = pop['Population in 2021'].str.replace(',', '')
pop['Population in 2021'] = pop['Population in 2021'].astype(float)
pop['Growth rate'] = pop['Growth rate'].str.rstrip("%").astype(float)/100


# In[92]:


# cleaning columns
df = df.drop(columns=['Standard error of ladder score', 'upperwhisker', 'lowerwhisker', 'Ladder score in Dystopia', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Dystopia + residual'], axis = 1)

# merging data
df = pd.merge(df, pop[['Country', 'Population in 2021', 'Growth rate']], on='Country', how='inner')
df = pd.merge(df, hofstede[['Country', 'Power distance', 'Individualism', 'Masculinity', 'Uncertainty avoidance', 'Long term orientation', 'Indulgence']], on='Country', how = 'left')


# In[94]:


#creating a bar plot to visualize the top 10 happiest and unhappiest countries in 2021
df_happiest_unhappiest = df[(df.loc[:,"Happiness score"] > 7.2) | (df.loc[:,"Happiness score"] < 3.8)]
df_happiest_unhappiest = df_happiest_unhappiest.sort_values(by='Happiness score', ascending=True)

fig1 = px.bar(df_happiest_unhappiest,
       x = "Happiness score", 
       y = "Country", 
       facet_row_spacing = 0.98,
       color="Happiness score",
       color_continuous_scale='RdBu',
       template = "simple_white",
       height = 800,
       title = "10 Happiest and 10 Unhappiest countries in 2021")
fig1.update_layout(title_x=0.5, title_y=0.91, title_font=dict(size=24))
fig1.update_xaxes(
    tickfont=dict(size=16))
fig1.show()


# In[95]:


#creating a scatter plot to visualize the average happiness score of the 20 happiest countries over time
background = "#fbfbfb"
fig, ax = plt.subplots(1,1, figsize=(10, 5),dpi=150)


# Reduced list as too many to show all at once 
top_list_ = df2.groupby('Country')['Happiness score'].mean().sort_values(ascending=False).reset_index()[:20].sort_values(by='Happiness score',ascending=True)


plot = 1
for country in top_list_['Country']:
    mean = df2[df2['Country'] == country].groupby('Country')['Happiness score'].mean()
    # historic scores
    sns.scatterplot(data=df2[df2['Country'] == country], y=plot, x='Happiness score',color='lightgray',s=50,ax=ax)
    # mean score
    sns.scatterplot(data=df2[df2['Country'] == country], y=plot, x=mean,color='lightsalmon',ec='black',linewidth=1,s=75,ax=ax)
    #2021 score
    sns.scatterplot(data=df[df['Country'] == country], y=plot, x='Happiness score',color='cornflowerblue',ec='black',linewidth=1,s=75,ax=ax)   
    plot += 1


ax.set_yticks(top_list_.index+1)
ax.set_yticklabels(top_list_['Country'][::-1], fontdict={'horizontalalignment': 'right'}, alpha=0.7)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_xlabel("Happiness Index Score",loc='left',color='gray')


for s in ['top','right','bottom','left']:
    ax.spines[s].set_visible(False)
    
Xstart, Xend = ax.get_xlim()
Ystart, Yend = ax.get_ylim()

ax.hlines(y=top_list_.index+1, xmin=Xstart, xmax=Xend, color='gray', alpha=0.5, linewidth=.3, linestyles='--')
ax.set_axisbelow(True)
ax.text(6.25, Yend+3.3, 'Happiness Index Scores through the years', fontsize=17, fontweight='bold', color='#323232')

plt.annotate('2021\nscore', xy=(7.842, 19), xytext=(8.2, 11),
             arrowprops=dict(color="k",facecolor='black',arrowstyle="fancy",connectionstyle="arc3,rad=.3"), fontsize=10,fontfamily='monospace',ha='center', color='cornflowerblue')

plt.annotate('Mean\nscore', xy=(7.6804, 20), xytext=(8.2, 16),
             arrowprops=dict(color="k",facecolor='black',arrowstyle="fancy",connectionstyle="arc3,rad=.5"), fontsize=10,fontfamily='monospace',ha='center', color='lightsalmon')


plt.show()


# In[96]:


#creating a scatter plot to visualize the average happiness score of the 20 unhappiest countries over time
background = "#fbfbfb"
fig, ax = plt.subplots(1,1, figsize=(10, 5),dpi=150)


# Reduced list as too many to show all at once 
top_list_ = df2.groupby('Country')['Happiness score'].mean().sort_values(ascending=True).reset_index()[:20]

plot = 1
for country in top_list_['Country']:
    mean = df2[df2['Country'] == country].groupby('Country')['Happiness score'].mean()
    # historic scores
    sns.scatterplot(data=df2[df2['Country'] == country], y=plot, x='Happiness score',color='lightgray',s=50,ax=ax)
    # mean score
    sns.scatterplot(data=df2[df2['Country'] == country], y=plot, x=mean,color='lightsalmon',ec='black',linewidth=1,s=75,ax=ax)
    #2021 score
    sns.scatterplot(data=df[df['Country'] == country], y=plot, x='Happiness score',color='brown',ec='black',linewidth=1,s=75,ax=ax)   
    plot += 1


ax.set_yticks(top_list_.index+1)
ax.set_yticklabels(top_list_['Country'], fontdict={'horizontalalignment': 'right'}, alpha=0.7)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_xlabel("Happiness Index Score",loc='left',color='gray')


for s in ['top','right','bottom','left']:
    ax.spines[s].set_visible(False)
    
Xstart, Xend = ax.get_xlim()
Ystart, Yend = ax.get_ylim()

ax.hlines(y=top_list_.index+1, xmin=Xstart, xmax=Xend, color='gray', alpha=0.5, linewidth=.3, linestyles='--')
ax.set_axisbelow(True)
ax.text(2.5, Yend+3.3, 'Happiness Index Scores through the years', fontsize=17, fontweight='bold', color='#323232')

plt.annotate('2021\nscore', xy=(5.05, 18), xytext=(5.5, 10),
             arrowprops=dict(color="k",facecolor='black',arrowstyle="fancy",connectionstyle="arc3,rad=.3"), fontsize=10,fontfamily='monospace',ha='center', color='brown')

plt.annotate('Mean\nscore', xy=(3.4, 1), xytext=(4.5, 4),
             arrowprops=dict(color="k",facecolor='black',arrowstyle="fancy",connectionstyle="arc3,rad=.5"), fontsize=10,fontfamily='monospace',ha='center', color='lightsalmon')


plt.show()


# In[99]:


#creating a scatter plot to visualize Happiness score and GDP per capita by Contries with Population by Regions
fig2 = px.scatter(df,
                x = "Logged GDP per capita",
                y = "Happiness score",
                size = "Population in 2021",
                template = "simple_white",
                hover_name = "Country",
                color = "Regional indicator", 
                height=600,
                size_max = 80)
fig2.update_layout(title = "Happiness score and GDP per capita by Contries with Population by Regions")
fig2.show()


# In[100]:


#creating a scatter plot to visualize Happiness score, Freedom and Corruption correlation
fig3 = px.scatter(df,
                x = "Freedom to make life choices",
                y = "Perceptions of corruption",
                size = "Happiness score",
                template = "simple_white",
                hover_name = "Country",
                color = "Regional indicator", 
                height=600,
                size_max = 15)
fig3.update_layout(title = "Happiness score, Freedom and Corruption")
fig3.show()


# In[101]:


#creating a scatter plot to visualize Happiness score and Healthy life expectancy by Regions
continent_score = df.groupby('Regional indicator')['Healthy life expectancy','Logged GDP per capita','Perceptions of corruption','Freedom to make life choices','Happiness score'].mean().reset_index()

fig4 = px.scatter(continent_score,
                x = continent_score['Healthy life expectancy'],
                y = continent_score['Happiness score'],
                size = continent_score['Healthy life expectancy'],
                template = "simple_white",
                hover_name = continent_score['Regional indicator'],
                color = continent_score['Regional indicator'], 
                height=600,
                size_max = 80)
fig4.update_layout(title = "Happiness score and Healthy life expectancy by Regions")
fig4.show()


# In[102]:


#creating a scatter plot to visualize Happiness score and Growth rate by Regions
fig5 = px.scatter(df,
                x = "Growth rate",
                y = "Happiness score",
                size = "Logged GDP per capita",
                template = "simple_white",
                hover_name = "Country",
                color = "Regional indicator", 
                height=600,
                size_max = 20)
fig5.update_layout(title = "Happiness score and Growth rate by Regions")
fig5.show()


# In[103]:


#creating a scatter plot to visualize Perception of freedom and Happiness score correlation
fig6 = px.scatter(df,
                x = "Freedom to make life choices",
                y = "Happiness score",
                size = "Logged GDP per capita",
                template = "simple_white",
                hover_name = "Country",
                color = "Regional indicator", 
                height=600,
                size_max = 20)
fig6.update_layout(title = "Perception of freedom and Happiness score")
fig6.show()


# In[104]:


#creating a choropleth map to visualize the happiness scores of countries in 2021
fig7 = px.choropleth(data_frame=df,
                    locations="Country",
                    locationmode="country names",
                    color="Happiness score",
                    color_continuous_scale='RdBu',
                    title="Happiness Score Map 2021")
fig7.update_layout(margin=dict(l=60, r=60, t=50, b=50))
fig7.update_layout(
    autosize=False,
    width=1200,
    height=600,
    title={
        'y':0.95,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'})
fig7.show()


# In[105]:


#creating an animated choropleth map to compare the ladder scores of countries based on their regional indicators
fig8 = px.choropleth(df.sort_values("Regional indicator"),
                   locations = "Country",
                   color = "Happiness score",
                   locationmode = "country names",
                   animation_frame = "Regional indicator")
fig8.update_layout(title={'text': "Ladder score Comparison by Countries", 
                                           'x':0.5,
                                           'xanchor': 'center',
                                           'font': dict(size=20)})
                  
fig8.show()


# In[106]:


#computing a correlation matrix for the dataframe df and creates a heatmap to visualize the correlation values
correlation_matrix = df.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[107]:


#The bar plot displays the number of countries in each region based on the 'Regional indicator' column

counts = df['Regional indicator'].value_counts()

plt.bar(counts.index, counts)

plt.xticks(rotation = 90)

plt.show();


# In[108]:


#The heatmap shows the correlation between happiness factors, indicating a positive correlation between 
#ladder score and social support, and a negative correlation between ladder score and perceptions of corruption
cols = ['Happiness score', 'Social support', 'Freedom to make life choices', 'Perceptions of corruption']
corr_matrix = df[cols].corr()

sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm');


# In[109]:


#This scatterplot shows the relationship between Generosity, Regional Indicator, and Ladder Score
sns.scatterplot(data = df, x = 'Generosity', y = 'Happiness score', hue = 'Regional indicator')
plt.xlabel('Generosity')
plt.ylabel('Regional Indicator')
plt.title('Generosity vs Regional Indicator')
plt.show();


# In[110]:


#The barplot shows the average happiness score (ladder score) for each region. Western Europe has the highest average score, while Sub-Saharan Africa has the lowest
sns.barplot(data = df, x = 'Happiness score', y = 'Regional indicator', palette = 'viridis')
plt.xlabel('Ladder Score')
plt.ylabel('Region')
plt.title('Happiness Scores by Region')
plt.show();


# In[111]:


df_target = df['Happiness score']

#remove the column Country and Happiness score
df_x = df.drop(['Country', 'Happiness score'], axis = 1)


# In[112]:


# Group by the regional indicator and calculate the mean of each column
mean_values = df_x.groupby('Regional indicator').transform('mean')

# Fill the null values with the respective mean values
df_x = df_x.fillna(mean_values)

df_x.isna().any()


# In[113]:


#get dummies
df_x = pd.get_dummies(df_x, columns=['Regional indicator'])

df_x.head()


# In[114]:


#normalize data to avoid biased 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_x = pd.DataFrame(scaler.fit_transform(df_x), columns = df_x.columns)

df_x.head()


# In[2]:


#deleting any missing value on the data set
df_x = df_x.dropna()

df_x.isna().any()


# In[117]:


#performing Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

pca = PCA()

Coord = pca.fit_transform(df_x)


# In[118]:


# is calculating the eigenvalues of the principal components obtained from PCA and creating a line plot to display these eigenvalues.
print("The eigenvalues are: ",pca.explained_variance_)

plt.plot(range(0,24), pca.explained_variance_)
plt.xlabel('Number of factors')
plt.ylabel('Eigenvalues')
plt.show()


# In[119]:


#computing the explained variance ratios of principal components obtained from PCA and plotting the cumulative sum of these ratios
print("Ratios: ",pca.explained_variance_ratio_)

plt.plot(np.arange(1, 25), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Factor number')
plt.ylabel('Cumsum')
plt.show()


# In[120]:


# Pie chart of the distribution of the share of variance explained by each axis.
L1 = list(pca.explained_variance_ratio_[0:6])
L1.append(sum(pca.explained_variance_ratio_[6:24]))

plt.pie(L1, labels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
                    'Others'], autopct='%1.3f%%', radius=2, labeldistance=1.2)

plt.legend(loc = 'center')
my_circle=plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# In[121]:


#plot the correlation circle 
racine_valeurs_propres = np.sqrt(pca.explained_variance_)
corvar = np.zeros((24, 24))
for k in range(15):
    corvar[:, k] = pca.components_[:, k] * racine_valeurs_propres[k]

# Delimitation 
fig, axes = plt.subplots(figsize=(20, 20))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

# Displaying variables
for j in range(3,15):
    plt.annotate(df_x.columns[j], (corvar[j, 0]*5, corvar[j, 1]*5), color='#091158', fontsize=10)
    plt.arrow(0, 0, corvar[j, 0]*4, corvar[j, 1]*4,
              alpha=0.8, head_width=0.02, color='b')
    
for j in range(2):
    plt.annotate(df_x.columns[j], (corvar[j, 0]*4, corvar[j, 1]*4), color='#091158', fontsize=10)
    plt.arrow(0, 0, corvar[j, 0]*4, corvar[j, 1]*4,
              alpha=0.8, head_width=0.02, color='b')

# Adding Axis
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# Circle and labels
cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
axes.add_artist(cercle)
plt.xlabel('AXIS 1')
plt.ylabel('AXIS 2')
plt.show()


# In[122]:


#creating a pair plot using the seaborn library to visualize the relationships between multiple variables in the dataframe

sns.pairplot(df[['Happiness score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Population in 2021']])

plt.show();


# In[129]:


#spliting the data into training and testing datasets, calculating the percentage of data used for training

X = df_x  
y = df_target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

percentage_training_data = (X_train.shape[0] / df_x.shape[0]) * 100

print("Training data percentage:", percentage_training_data)

if percentage_training_data > 70:
    print("Percentage of data in the training set is above 70%.")
else:
    print("Percentage of data in the training set is not above 70%.")

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Calculate the mean squared error for both training and testing sets
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)


print("MSE train:", train_mse)
print("MSE test:", test_mse)
print("Training R-squared score:", train_r2)
print("Testing R-squared score:", test_r2)


# In[1]:


#creating a scatter plot to visualize the predicted values versus the actual values of the target variable for the testing dataset
plt.scatter(y_test_pred, y_test);


# In[131]:


#performing regression analysis using a decision tree regressor and calculating the R-squared score for both the training and testing datasets

regressor = DecisionTreeRegressor(random_state = 42) 
  
regressor.fit(X_train, y_train)

print(regressor.score(X_train, y_train))

print(regressor.score(X_test, y_test))


# In[132]:


#performing regression analysis using a decision tree algorithm and visualizing the resulting decision tree

regressor = DecisionTreeRegressor(random_state = 42, max_depth = 3) 
  
regressor.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(20, 20))  

plot_tree(regressor, 
          feature_names = df_x.columns, 
          filled = True, 
          rounded = True)

plt.show()

