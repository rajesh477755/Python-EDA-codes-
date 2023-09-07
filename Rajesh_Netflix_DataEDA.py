import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('data cleaning and exploration')
data = pd.read_csv('Netflix_Userbase.csv')
df = pd.DataFrame(data)
print(df)
'''Exploratory data analysis'''
print('Head')
print(df.head())
print("Tail")
print(df.tail())
print("Information")
print(df.info())
print('Shape')
print(df.shape)
print('Data types')
print(df.dtypes)
print('Columns')
print(df.columns)
print('index')
print(df.index)
print('describe')
print(df.describe)
print(df['Age'].describe())
print(df.isnull().sum())
print(df.isnull())
print(df['Country'].mode())
print(df['Monthly Revenue'].mean())
print(df.dropna())
print(df.nunique())
print(df['Subscription Type'].unique())
print(df.rename(columns={'Age': 'Ages'}))
print(df.duplicated())
print(df.groupby('Subscription Type')['Monthly Revenue'].count())
print(df.groupby('Subscription Type')['Monthly Revenue'].max())
#MATPLOTLIB
x = df['Subscription Type']
y = df['Monthly Revenue']
plt.xlabel('Subscription Type')
plt.ylabel('Monthly Revenue')
plt.bar(df['Subscription Type'], df['Monthly Revenue'])
plt.show()
x = df['Subscription Type']
y = df['Monthly Revenue']
plt.xlabel('Subscription Type')
plt.ylabel('Monthly Revenue')
plt.scatter(df['Subscription Type'], df['Monthly Revenue'])
plt.show()
x = df['Subscription Type']
y = df['Monthly Revenue']
plt.xlabel('Subscription Type')
plt.ylabel('Monthly Revenue')
plt.scatter(x[23:78:1], y[23:78:1])
plt.show()
x = df['Subscription Type']
y = df['Monthly Revenue']
plt.xlabel('Subscription Type')
plt.ylabel('Monthly Revenue')
plt.plot(x[23:78:1], y[23:88:1], color='r')
plt.grid(color='g', linestyle=':', linewidth=0.8)
plt.show()
x = df['Subscription Type']
y = df['Monthly Revenue']
z = df['Age']
plt.scatter(x[23:77:1], y[23:77:1])
plt.scatter(x[23:77:1], z[23:77:1])
plt.grid(color='c')
plt.show()
data = df['Monthly Revenue']
keys = df['Age']
plt.title('Between index from 23 to 56 of the startips dataset')
plt.pie(data[23:56:1], labels=keys[23:56:1])
plt.show()
#SEABORN
sns.pairplot(df)
plt.show()
ax = sns.countplot(x='Subscription Type', data=df)
plt.show()
sns.boxplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.displot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.scatterplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.lineplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.relplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.jointplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.catplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.violinplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.show()
sns.heatmap(data=df.corr())
plt.show()
