import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('data cleaning and exploration')
data = pd.read_csv('smartphone_cleaned_v5.csv')
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
print(df['experience'].describe)
print('is null')
print(df.isnull)
print(df.isnull().sum())
print('mean')
print(df[''].mean())
print(df.groupby([''])[''].mean())
print('mode')
print(df[''].mode())
print('median')
print(df[''].median())
print('rename')
print(df.rename(columns={'':""}))
print('groupby')
print(df.groupby('')[''].min())
print('drop null values')
print(df.dropna())
#MATPLOTLIB
x=df[""]
y=df[""]
plt.title('Scatter plot')
plt.xlabel("")
plt.xticks(rotation=45)
plt.ylabel("")
plt.yticks(rotation=20)
plt.scatter(x[],y[],color='')
plt.show()
print('bar graph')
x=df[""]
y=df[""]
plt.title('')
plt.xlabel("")
plt.xticks(rotation=90)
plt.ylabel("")
plt.yticks(rotation=0)
plt.bar(x[],y[],color='')
plt.show()
print('grid plot')
x=df[""]
y=df[""]
plt.title('')
plt.xlabel("")
plt.xticks(rotation=90)
plt.ylabel("")
plt.plot(x[],y[])
plt.grid(color='',linestyle='',linewidth=1)
plt.show()
print('scattered grid plot')
x=df[""]
y=df[""]
z=df[""]
plt.title('')
plt.xticks(rotation=45)
plt.scatter(x[],y[])
plt.scatter(x[],z[])
plt.grid(color='',linestyle='',linewidth=0.5)
plt.show()
data=df['']
keys=df['']
print('pie chart')
plt.title('')
plt.pie(data[],labels=keys[])
plt.show()
#SEABORN

sns.countplot(x='', data=df)
plt.title("countplot")
plt.show()
print('count plot')
sns.pairplot(df)
plt.title("pariplot")
plt.show()
print('pariplot')
sns.boxplot(x='',y='',data=df)
plt.title('box plot')
plt.show()
print('boxplot')
sns.displot(x='',y='',data=df)
plt.title("displot")
plt.show()
print('distribution plot')
sns.scatterplot(x='',y='',data=df)
plt.title("distribution plot")
plt.show()
print("scatter plot")
sns.lineplot(x='',y='',data=df)
plt.title("line plot")
plt.show()
print("line plot")
sns.relplot(x='',y='',data=df)
plt.title("replot")
plt.show()
print("re plot")
sns.jointplot(x='',y='',data=df)
plt.title("joint plot")
plt.show()
print("joint plot")
sns.catplot(x='',y='',data=df)
plt.title("categorical plot")
plt.show()
print("categorical plot")
sns.violinplot(x='',y='',data=df)
plt.title("violin plot")
plt.show()
print("violin plot")