---
title: "Detailed Analysis: Startups vs Mid/Large Size Company"
last_modified_at: 2021-01-24T21:30:02-05:00
categories:
  - Blogs
tags:
  - EDA
  - Python
excerpt: How Startups are doing different from mid/large size company? Salary? Work Opportunity?
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

![Banner](https://miro.medium.com/max/3000/1*NARFfXvPHDwwMs_sai_DOw.jpeg)

You can’t force corporate rules on a startup — or vice versa. Size and complexity affect the basic methodologies used to develop ideas and create revenues, and it is dangerous to ignore the differences.

Smaller companies are organized in a way that stimulates experimentation and risk-taking, while large and complex enterprises are incentivized to maintain the status quo by any means necessary.



<details><summary>code</summary>
<p>

```python
!pip install -q dalex
```

```python
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

from sklearn.ensemble import *
from sklearn.metrics import *
import dalex as dx
import plotly.offline as pyo
pyo.init_notebook_mode()

warnings.filterwarnings('ignore')
pd.set_option("display.precision", 3)
```
</p>
</details>

# Introduction

After first look of the Kaggle survey 2020 dataset, I was curious to know that, is there any difference in data science b/w startups and big companies, who all are prefer to work in these companies?, and what kind of age groups are working in? For my analysis I am going to use `Q20`.

    Q20: What is the size of the company where you are employed?

### Problem Statement: How Startups are doing different from mid/large size company?

# Approach


1.	**Data Preparation:** Started with making company_category column using response of Q20. The values of company_category are Startup, Mid-Size Company and Large Size Company.
2.	**Modelling:** Made Classification model to classify company_category which will be used as feature selection with the use of model feature_importance and break_down approach.
3.	**Aspect Identification:** Using selected feature, I identified aspects on which we will further go down to understand data.
4.	 **Exploratory Analysis:** With the use of different plotting technique, I will to identity pattern which will tell how startups are different from mid/large size companies. 
5.	**Summary Table:** At last I will conclude the difference with the help of difference table.

# Data Loading & Preprocess

Let's Load data and do some preprocessing according to our requirement. I have bucketed Q20 response into three bucket i.e. 

    company_category = {
        "0-49 employees":"Startup",
        "50-249 employees":"Mid Size Company",
        "250-999 employees":"Mid Size Company",
        "1000-9,999 employees":"Large Size Company",
        "10,000 or more employees":"Large Size Company"
    }
    
The total size of dataset, who have given answer to Q20: **11403**


```python
data = pd.read_csv("../input/kaggle-survey-2020/kaggle_survey_2020_responses.csv")
questions = data[0:1].to_numpy().tolist()[0]
column_question_lookup = dict(zip(data.columns.tolist(), questions))
data = data[1:]
data = data[~data['Q20'].isna()]
data.shape
```




    (11403, 355)




```python
def get_columns(q):
    if q in ['Q1', 'Q2', 'Q3']:
        return [q]
    else:
        return [c for c in data.columns if c.find(q)!=-1]
```


```python
# test get_columns
q = "Q12"
get_columns(q)
```




    ['Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_OTHER']


<details><summary>code</summary>
<p>

```python
company_category_lookup = {
    "0-49 employees":"Startup",
    "50-249 employees":"Mid Size Company",
    "250-999 employees":"Mid Size Company",
    "1000-9,999 employees":"Large Size Company",
    "10,000 or more employees":"Large Size Company"
}
data['Company_Category'] = data['Q20'].apply(lambda x: company_category_lookup[x])
```


```python
in_order = [
    "I do not use machine learning methods", "Under 1 year", "1-2 years",
    "2-3 years", "3-4 years", "4-5 years", "5-10 years", "10-20 years",
    "20 or more years"
]

data['Q15'] = pd.Categorical(data['Q15'], categories=in_order, ordered=True)

in_order = [
    "I have never written code", "< 1 years", "1-2 years", "3-5 years",
    "5-10 years", "10-20 years", "20+ years"
]

data['Q6'] = pd.Categorical(data['Q6'], categories=in_order, ordered=True)

in_order = ["0", "1-2", "3-4", "5-9", "10-14", "15-19", "20+"]

data['Q21'] = pd.Categorical(data['Q21'], categories=in_order, ordered=True)

salary_in_order = [
    "$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999",
    "5,000-7,499", "7,500-9,999", "10,000-14,999", "15,000-19,999",
    "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999",
    "50,000-59,999", "60,000-69,999", "70,000-79,999", "80,000-89,999",
    "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999",
    "200,000-249,999", "300,000-500,000", "> $500,000"
]

data['Q24'] = pd.Categorical(data['Q24'],
                             categories=salary_in_order,
                             ordered=True)

in_order = [
    "No formal education past high school",
    "Some college/university study without earning a bachelor’s degree",
    "Bachelor’s degree",
    "Master’s degree",
    "Doctoral degree",
    "Professional degree",
    "I prefer not to answer"
]

data['Q4'] = pd.Categorical(data['Q4'],
                             categories=in_order,
                             ordered=True)
```

</p>
</details>

# Modeling

First, I started with making company category classification model. For data preparation, I converted categorical variable into dummy/indicator variables and then passed into RandomForestClassifier model.

<details><summary>code</summary>
<p>

```python
df = data.drop(columns=["Time from Start to Finish (seconds)", "Q20", "Q21", "Company_Category"])
y_data = data['Company_Category'].values

# Make Dummies
df = pd.get_dummies(df)

# Fill in missing values
df.dropna(axis=1, how='all', inplace=True)
dummy_columns = [c for c in df.columns if len(df[c].unique()) == 2]
non_dummy = [c for c in df.columns if c not in dummy_columns]
df[dummy_columns] = df[dummy_columns].fillna(0)
df[non_dummy] = df[non_dummy].fillna(df[non_dummy].median())

print(f">> Filled NaNs in {len(dummy_columns)} OHE columns with 0")
print(f">> Filled NaNs in {len(non_dummy)} non-OHE columns with median values")

X_data = df.to_numpy()

print(X_data.shape, y_data.shape)

classifier = RandomForestClassifier(n_estimators=100,
                                        criterion='entropy',
                                        random_state=3107)

classifier.fit(X_data, y_data)

y_pred = classifier.predict(X_data)
print('Training Accuracy :', accuracy_score(y_data, y_pred))
```
</p>
</details>


    >> Filled NaNs in 537 OHE columns with 0
    >> Filled NaNs in 0 non-OHE columns with median values
    (11403, 537) (11403,)
    Training Accuracy : 0.9999123037797071


Let's See the feature importance plot.

Feature importance refers to a class of techniques for assigning scores to input features (X_data) to a predictive model(classifier) that indicates the relative importance of each feature when making a prediction.


```python
feat_importances = pd.Series(classifier.feature_importances_,
                             index=list(df.columns))
feat_importances.nlargest(30).plot(
    kind='barh',
    figsize=(10, 20),
    color='#9B5445',
    zorder=2,
    width=0.85,
    fontsize=20
)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3b4d96aa50>




![png](/assets/images/eda/output_16_1.png)


**Importance Features:**
- Q24_$0-999: Salary
- Q22_We have well established ML methods (i.e., models in production for more than 2 years): Employer incorporate machine learning methods
- Q3_India: Location
- Q22_No (we do not use ML methods): Employer incorporate machine learning methods
- Q4_Master’s degree: Highest level of formal education
- Q7_Part_3_SQL: Programming languages do you use on a regular basis
- Q23_Part_1_Analyze and understand data to influence product or business decisions: Activities that make up an important part of your role at work.
- Q22_We are exploring ML methods (and may one day put a model into production)*: Employer incorporate machine learning methods
- Q9_Part_1_Jupyter (JupyterLab, Jupyter Notebooks, etc)*: Integrated development environments (IDE's) do you use on a regular basis
- Q4_Bachelor’s degree*: Highest level of formal education
    
**Important Questions:**

        important_question = [
            'Q1', 'Q3', 'Q4', 'Q5', 'Q7', 'Q9', 'Q10', 'Q12', 'Q14', 'Q17', 'Q22',
            'Q23', 'Q24', 'Q25', 'Q37', 'Q39'
        ]

Let's try to use dalex (moDel Agnostic Language for Exploration and eXplanation) to see the break_down plots. The most commonly asked question when trying to understand a model’s prediction for a single observation is: which variables contribute to this result the most?. For that I have used break_down plot from dalex.


```python
exp = dx.Explainer(classifier, X_data, y_data)

bd_large = exp.predict_parts(df[0:1], type='break_down', label="Large Size Company")
bd_mid = exp.predict_parts(df[2:3], type='break_down', label="Mid Size Company")
bd_startup = exp.predict_parts(df[4:5], type='break_down', label="Startup")

k = 20
imps_large = bd_large.result.variable_name.values[1:k + 1].tolist()
imps_mid = bd_mid.result.variable_name.values[1:k + 1].tolist()
imps_startup = bd_startup.result.variable_name.values[1:k + 1].tolist()
results = pd.DataFrame({
    "Large Size Company": [],
    "Mid Size Company": [],
    "Startup": []
})
for ids in zip(imps_large, imps_mid, imps_startup):

    results = results.append(
        pd.DataFrame({
            "Large Size Company": [list(df.columns)[int(ids[0])]],
            "Mid Size Company": [list(df.columns)[int(ids[1])]],
            "Startup": [list(df.columns)[int(ids[2])]]
        }))
```

    Preparation of a new explainer is initiated
    
      -> data              : numpy.ndarray converted to pandas.DataFrame. Columns are set as string numbers.
      -> data              : 11403 rows 537 cols
      -> target variable   : 11403 values
      -> target variable   : Please note that 'y' is a string array.
      -> target variable   : 'y' should be a numeric or boolean array.
      -> target variable   : Otherwise an Error may occur in calculating residuals or loss.
      -> model_class       : sklearn.ensemble._forest.RandomForestClassifier (default)
      -> label             : Not specified, model's class short name will be used. (default)
      -> predict function  : <function yhat_proba_default at 0x7f3b4f54c3b0> will be used (default)
      -> predict function  : Accepts pandas.DataFrame and numpy.ndarray.
      -> predicted values  : min = 0.0, mean = 0.265, max = 0.91
      -> model type        : classification will be used (default)
      -> residual function : difference between y and yhat (default)
      -> residuals         :  'residual_function' returns an Error when executed:
    unsupported operand type(s) for -: 'str' and 'float'
      -> model_info        : package sklearn
    
    A new explainer has been created!


```python
results.reset_index(drop=True)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Large Size Company</th>
      <th>Mid Size Company</th>
      <th>Startup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Q24_$0-999</td>
      <td>Q24_$0-999</td>
      <td>Q24_$0-999</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q24_100,000-124,999</td>
      <td>Q25_$10,000-$99,999</td>
      <td>Q4_Doctoral degree</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Q29_A_Part_11_Amazon Redshift</td>
      <td>Q29_A_Part_11_Amazon Redshift</td>
      <td>Q5_Research Scientist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q37_Part_4_DataCamp</td>
      <td>Q37_Part_4_DataCamp</td>
      <td>Q24_30,000-39,999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Q19_Part_3_Contextualized embeddings (ELMo, CoVe)</td>
      <td>Q7_Part_10_Bash</td>
      <td>Q39_Part_9_Journal Publications (peer-reviewed...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Q9_Part_8_  Sublime Text</td>
      <td>Q36_Part_9_I do not share my work publicly</td>
      <td>Q36_Part_9_I do not share my work publicly</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Q29_A_Part_12_Amazon Athena</td>
      <td>Q3_India</td>
      <td>Q3_India</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Q3_India</td>
      <td>Q39_Part_11_None</td>
      <td>Q22_We use ML methods for generating insights ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Q1_30-34</td>
      <td>Q1_30-34</td>
      <td>Q37_Part_11_None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Q17_Part_10_Transformer Networks (BERT, gpt-3,...</td>
      <td>Q6_5-10 years</td>
      <td>Q1_35-39</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Q31_A_Part_1_Amazon QuickSight</td>
      <td>Q27_A_Part_1_ Amazon EC2</td>
      <td>Q17_Part_3_Gradient Boosting Machines (xgboost...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Q14_Part_5_ Shiny</td>
      <td>Q31_A_Part_5_Tableau</td>
      <td>Q22_We have well established ML methods (i.e.,...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Q6_5-10 years</td>
      <td>Q37_Part_10_University Courses (resulting in a...</td>
      <td>Q33_A_Part_7_No / None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Q27_A_Part_1_ Amazon EC2</td>
      <td>Q15_3-4 years</td>
      <td>Q1_18-21</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Q29_A_Part_2_PostgresSQL</td>
      <td>Q33_A_Part_7_No / None</td>
      <td>Q7_Part_1_Python</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Q31_A_Part_5_Tableau</td>
      <td>Q1_18-21</td>
      <td>Q8_Python</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Q28_A_Part_1_ Amazon SageMaker</td>
      <td>Q12_Part_3_None</td>
      <td>Q8_R</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Q17_Part_3_Gradient Boosting Machines (xgboost...</td>
      <td>Q26_A_Part_1_ Amazon Web Services (AWS)</td>
      <td>Q16_Part_13_ Tidymodels</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Q18_Part_2_Image segmentation methods (U-Net, ...</td>
      <td>Q9_Part_5_ PyCharm</td>
      <td>Q25_$100,000 or more ($USD)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Q33_A_Part_7_No / None</td>
      <td>Q25_$100,000 or more ($USD)</td>
      <td>Q16_Part_1_  Scikit-learn</td>
    </tr>
  </tbody>
</table>
</div>



Now we got top 20 features which contributed to the prediction pobability. But Still we need to figure out what range of value of these features by company category and how these features related to Company categoies.

# Demographic & Opportunity Analysis

Now, We have list of importance features, let's analyze these features with respect to the company categories. I have divided these features into 5 aspects and these are the following aspects:

1. **Age and Location Aspect:** Which age group prefer working in startups and from where are they from?
2. **Education and Professional Aspect:** How much educated people prefer in working in Startups and Mid/Large Size companies?
3. **Programming Language Aspect:** What programming language/framework they uses on daily basis. 
4. **Work Opportunity Aspect:** What are the work opportunities in these companies?
5. **Salary Aspect:** How much they are paying?


## Age and Location Aspect

Let's try to figure out which age group interested in Startup and which age group prefer established company

<details><summary>code</summary>
<p>

```python
def add_rectangular_patch(ax, xy, w, h, color, alpha=0.4, lw=3, fill=True):
    ax.add_patch(
        Rectangle(xy, w, h, fill=fill, color=color, lw=lw, alpha=alpha))


def add_annotation(ax, text, xy, xytext, facecolor):
    ax.annotate(
        text,
        xy=xy,
        xycoords='data',
        fontsize=16,
        weight='bold',
        xytext=xytext,
        textcoords='axes fraction',
        arrowprops=dict(facecolor=facecolor, shrink=0.05),
        horizontalalignment='right',
        verticalalignment='top',
    )


def add_annotation_v2(ax,
                      text,
                      xy,
                      fontsize,
                      color,
                      weight='bold',
                      verticalalignment='center',
                      horizontalalignment='center'):
    ax.annotate(text,
                xy=xy,
                fontsize=fontsize,
                color=color,
                weight=weight,
                verticalalignment=verticalalignment,
                horizontalalignment=horizontalalignment)
    
def hide_axes(this_ax):
    this_ax.set_frame_on(False)
    this_ax.set_xticks([])
    this_ax.set_yticks([])
    return this_ax
```


```python
df = pd.crosstab([data['Q1']], [data['Company_Category']])
df1 = df.apply(lambda r: r / r.sum(), axis=0)
df2 = df.apply(lambda r: r / r.sum(), axis=1)
df2 = df2.reindex(list(df2.index)[::-1])

heatmap_args = dict(annot_kws={"size": 16},
                    cmap=cm.get_cmap("Greys", 12),
                    cbar=False,
                    annot=True,
                    fmt="d",
                    lw=2,
                    square=False)

f, ax = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(30, 20),
)

# ax [0,0]
hide_axes(ax[0, 0])

# ax[0,1]
df.apply(lambda r: r.sum(), axis=1).plot.barh(ax=ax[0, 1],
                                              fontsize=20,
                                              color='#9B5445')
total = len(data)
for p in ax[0, 1].patches:
    percentage = '{:.1f}%'.format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax[0, 1].annotate(percentage, (x, y))

add_rectangular_patch(ax[0, 1], (0, 0.5),
                      2550,
                      4,
                      'darkgreen',
                      alpha=0.4,
                      lw=3)
add_annotation(ax[0, 1], '67%', (2400, 5), (0.8, 0.65), 'darkgreen')

add_rectangular_patch(ax[0, 1], (0, 1.5), 2550, 2, 'darkred', alpha=0.4, lw=3)
add_annotation(ax[0, 1], '40%', (1000, 3), (0.6, 0.65), 'darkred')

# ax[0,2]
hide_axes(ax[0, 2])

# ax[1,0]
df1.transpose()[list(df1.transpose().columns)[::-1]].plot.bar(ax=ax[1, 0],
                         stacked=True,
                         fontsize=20,
                         colormap=cm.get_cmap("tab20", 20))

ax[1,0].legend(fontsize=20, handlelength=1,labelspacing =0.2, loc='upper right', bbox_to_anchor=(0.5, 0.6))

add_rectangular_patch(ax[1, 0], (-0.5, 0.45),
                      2,
                      0.4,
                      'darkgreen',
                      alpha=0.2,
                      lw=5,
                      fill=True)

add_rectangular_patch(ax[1, 0], (1.5, 0.73),
                      1,
                      0.27,
                      'darkred',
                      alpha=0.2,
                      lw=5,
                      fill=True)

# ax[1,1]
midpoint = (df.values.max() - df.values.min()) / 2
hm = sns.heatmap(df, ax=ax[1, 1], center=midpoint, **heatmap_args)
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=20, rotation=90)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=20, rotation=0)

# ax[1,2]
df2.plot.barh(ax=ax[1, 2], fontsize=20, stacked=True)
ax[1,2].legend(fontsize=20, handlelength=1,labelspacing =0.2, loc=6)
add_rectangular_patch(ax[1, 2], (0, 8.5),
                      1,
                      1.9,
                      'darkgreen',
                      alpha=0.2,
                      lw=5,
                      fill=True)
add_annotation_v2(ax[1, 2],
                  'Learning-First',
                  (0.5, 9.5),
                  fontsize=40,
                  color='white',
                  weight='bold',
                  verticalalignment='center',
                  horizontalalignment='center')

add_rectangular_patch(ax[1, 2], (0, 2.5),
                      1,
                      5.9,
                      'darkred',
                      alpha=0.2,
                      lw=5,
                      fill=True)
add_annotation_v2(ax[1, 2],
                  'Stability-First',
                  (0.5, 6.5),
                  fontsize=40,
                  color='white',
                  weight='bold',
                  verticalalignment='center',
                  horizontalalignment='center')

add_rectangular_patch(ax[1, 2], (0, 0),
                      1,
                      2.5,
                      'darkgreen',
                      alpha=0.2,
                      lw=5,
                      fill=True)
add_annotation_v2(ax[1, 2],
                  "Let's-Do-Startup",
                  (0.5, 1.0),
                  fontsize=40,
                  color='white',
                  weight='bold',
                  verticalalignment='center',
                  horizontalalignment='center')
title = f.suptitle('Learning First or Stability First', fontsize=30)
```

</p>
</details>


![png](/assets/images/eda/output_28_0.png)


🚀Highlights:
1. **67%** of respondents age are b/w 22-40 and **40%** are in 25-34
2. **5 out of 10** in *Large and Mid size company* are of age b/w **25-34**, where as **3 out of 10** in *Startup* has employee of age b/w **18-24**
3. **More than ~50%** of respondents having age **b/w 18-24** are working in *Startup*, whereas **more than ~60%** having age **b/w 25-54** are working in either in *Mid or Large size company*. It feels like in the starting of career they want to learn lots of different things and after 25 they go for stability in life for work-life balance.
4. There is an interesting pattern **after 55**, Looks like people again want to **learn and discover new thing** and want to get rid of corporate culture and go for the *startup*.

<details><summary>code</summary>
<p>

```python
df = pd.crosstab([data['Q3']], [data['Company_Category']])
df = df.reindex(df.sum(axis=1).sort_values().index)
ax = df.plot.barh(
    stacked=True,
    figsize=(15, 15),
    width=0.85,
)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

title = ax.title.set_text('Does India or USA has more respondents')
```

</p>
</details>

![png](/assets/images/eda/output_30_0.png)


🚀Highlights:
1. ~35 of respondents are from India or USA
2. Distrubtion of Company category looks balanced b/w Country wise respondents

⚡Inference:

In the starting (18-24) of carrier people go for Startups to learn and experiment new things, in middle(25-34) they go to establish company for maintaining work life balance because that time they likely to have families n all and in last(50+) they again go for Startups, in this time they likely to have some idea for entrepreneur and they want to implement that.

## Education and Professional Aspect

Now Let's try to figure out how formal education are distrubuted over company category. Do they actualy perfer Master or PhDs or they also consider bachelors.

<details><summary>code</summary>
<p>

```python
df = pd.crosstab([data['Q4']], [data['Company_Category']])
df1 = df.apply(lambda r: r/r.sum(), axis=0)
df2 = df.apply(lambda r: r/r.sum(), axis=1)

f, ax = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(30, 20),
)

hide_axes(ax[0, 0])

df.apply(lambda r: r.sum(), axis=1).plot.barh(ax=ax[0, 1],
                                              fontsize=20,
                                              color='#9B5445')
total = len(data)
for p in ax[0, 1].patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax[0, 1].annotate(percentage, (x, y))

hide_axes(ax[0, 2])

df.plot.bar(ax=ax[1, 0], fontsize=20)
ax[1,0].legend(fontsize=20, handlelength=1,labelspacing =0.2)

df1.transpose().plot.bar(ax=ax[1, 1], fontsize=20, stacked=True, colormap=cm.get_cmap("tab20", 20))
ax[1,1].legend(fontsize=13, handlelength=1, labelspacing =0.2, loc=10)

df2.plot.bar(ax=ax[1, 2], fontsize=20, stacked=True)
ax[1,2].legend(fontsize=20, handlelength=1,labelspacing =0.2)

add_rectangular_patch(ax[1, 2], (-0.5, 0.45),
                      2,
                      0.6,
                      'darkgreen',
                      alpha=0.2,
                      lw=5,
                      fill=True)
title = f.suptitle('Master/PhD or Bachelor is enough?', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_35_0.png)


🚀Highlights:
1. **~45%** respondents have completed **Master's Dregree**.
2. *Startups* has **more bachelors** and **less Master's & PhD's** compare to *Large and Mid Size company*.
3. The respondents who **have'nt completed either high school or bachelors** are mostly work for *Startups*.

<details><summary>code</summary>
<p>

```python
def get_count_dfs(data, col1, col2):
    df = pd.crosstab([data[col1]], [data[col2]])
    df1 = df.apply(lambda r: r / r.sum(), axis=0)
    df2 = df.apply(lambda r: r / r.sum(), axis=1)
    return df, df1, df2

def reindex_df(df, reverse=False):
    if reverse:
        df = df.reindex(list(df.sum(axis=1).sort_values().index)[::-1])
        return df
        
    df = df.reindex(df.sum(axis=1).sort_values().index)    
    return df


main_col = "Company_Category"
by_col = "Q5"
by_col2 = "Q4"

index_cols = ['Software Engineer', 'DBA/Database Engineer', 'Data Engineer', 'Machine Learning Engineer', 'Statistician', 'Data Analyst', 'Data Scientist', 'Research Scientist', 'Business Analyst', 'Product/Project Manager', 'Other']


df, df1, df2 = get_count_dfs(data, by_col, main_col)

df = df.reindex(index_cols)
df1 = df1.reindex(index_cols)
df2 = df2.reindex(index_cols)

df3 = pd.crosstab([data[by_col]], [data[by_col2]])
df3 = df3.reindex(index_cols)

heatmap_args = dict(annot=True,
                    fmt="d",
                    square=False,
                    cmap=cm.get_cmap("Greys", 12),
                    center=90,
                    vmin=0,
                    vmax=500,
                    lw=4,
                    cbar=False)

f, ax = plt.subplots(nrows=2,
                     ncols=3,
                     figsize=(30, 20),
                     gridspec_kw={
                         'height_ratios': [4, 6],
                         'wspace': 0.6,
                         'hspace': 0.6
                     })

# ax[0,0]
df = df.reindex(index_cols[::-1])
df.apply(lambda r: r.sum(), axis=1).plot.barh(ax=ax[0, 0],
                                              fontsize=20,
                                              color='#9B5445')
total = len(data)
for p in ax[0, 0].patches:
    percentage = '{:.1f}%'.format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax[0, 0].annotate(percentage, (x, y))

# ax[0,1]
hm = sns.heatmap(df3, ax=ax[0, 1], annot_kws={"size": 16}, **heatmap_args)
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=20)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=20)

add_rectangular_patch(ax[0, 1], (0, 5),
                      2,
                      2,
                      'yellow',
                      alpha=0.1,
                      lw=5,
                      fill=True)

# ax[0,2]
df3.apply(lambda r: r.sum(), axis=0).plot.bar(ax=ax[0, 2],
                                              fontsize=20,
                                              color='#9B5445')

total = len(data)
for p in ax[0, 2].patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() - 0.5
    y = p.get_y() + p.get_height() + 100
    ax[0, 2].annotate(percentage, (x, y))
    
# ax[1,0]
df = df.reindex(index_cols)
df.plot.bar(ax=ax[1, 0], fontsize=20, width=0.65)
ax[1,0].legend(fontsize=20, handlelength=1,labelspacing =0.2, loc=1)
add_rectangular_patch(ax[1, 0], (2.5, 0),
                      1,
                      600,
                      'darkgreen',
                      alpha=0.2,
                      lw=5,
                      fill=True)

# ax[1,1]
df1.transpose().plot.bar(ax=ax[1, 1], stacked=True,colormap=cm.get_cmap("tab20", 12), fontsize=20, width=0.65)
ax[1,1].legend(fontsize=13, handlelength=1, labelspacing =0.2, loc=1)

# ax[1,2]
df2.plot.bar(ax=ax[1, 2], fontsize=20, stacked=True, width=0.65)
ax[1,2].legend(fontsize=20, handlelength=1,labelspacing =0.2, loc=1)
title = f.suptitle('Data Scientist or Software Engineer or ML Engineer?', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_37_0.png)


🚀Highlights:
1. **~22%** data scientist and **~16%** software developer respondents.
2. **~1.5%** of respondents has **not compeleted thier high school or bachelor's** and working as **Data scientist or Analyst**.
3. *Startups* has **more number of Machine Learning Engineers** compare to Mid or Large Size company.
4. **~40%** of **Research Scientist** are from *Startups*.
5. More **Business Analyst** Profiles are in *Large Size Company*.

## Programming Language Aspect

<details><summary>code</summary>
<p>

```python
df = pd.crosstab([data['Q6']], [data['Company_Category']])
df1 = df.apply(lambda r: r/r.sum(), axis=0)
df2 = df.apply(lambda r: r/r.sum(), axis=1)
df = df.reindex(list(df.index)[::-1] )

df_ = pd.crosstab([data['Q15']], [data['Company_Category']])
df1_ = df_.apply(lambda r: r/r.sum(), axis=0)
df2_ = df_.apply(lambda r: r/r.sum(), axis=1)
df_ = df_.reindex(list(df_.index)[::-1] )



heatmap_args = dict(annot_kws={"size": 16},
                    cmap=cm.get_cmap("Greys", 12),
                    cbar=False,
                    annot=True,
                    fmt="d",
                    lw=2,
                    square=False)

f, ax = plt.subplots(nrows=2,
                     ncols=3,
                     figsize=(30, 20),
                     gridspec_kw={
                         'height_ratios': [5, 5],
                         'wspace': 0.6,
                         'hspace': 0.6
                     })

# ax[0,0]
df1.transpose().plot.bar(ax=ax[0, 0], fontsize=20, stacked=True, width=0.65, colormap=cm.get_cmap("tab20", 20))

# ax[0,1]
midpoint = (df.values.max() - df.values.min()) / 2
hm = sns.heatmap(df, ax=ax[0, 1], center=midpoint, **heatmap_args)
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=20, rotation=90)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=20, rotation=0)

# ax[0,2]
df2.plot.barh(ax=ax[0, 2], fontsize=20, width=0.65, stacked=True)

# ax[1,0]
df1_.transpose().plot.bar(ax=ax[1, 0], fontsize=20, stacked=True, width=0.65, colormap=cm.get_cmap("tab20", 20))

# ax[1,1]
midpoint_ = (df_.values.max() - df_.values.min()) / 2
hm_ = sns.heatmap(df_, ax=ax[1, 1], center=midpoint_, **heatmap_args)
hm_.set_xticklabels(hm_.get_xmajorticklabels(), fontsize=20, rotation=90)
hm_.set_yticklabels(hm_.get_ymajorticklabels(), fontsize=20, rotation=0)

# ax[1,2]
df2_.plot.barh(ax=ax[1, 2], fontsize=20, width=0.65, stacked=True)

title = f.suptitle('Coding experience or ML experience?', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_40_0.png)


🚀Highlights:
1. **~40%** of employee of *Large size company* are of **3-10 years coding experience**.
2. **~60%** of employee of *Startup* are under **5 years of coding experience**.
3. **~50%** of respondents having **0-2 years of coding experience works** in *Startup*.
4. **~30%** of employee of *Large size* company are of **0-1 year of Machine Learning Experience**.
5. **~40%** of employee of *Startup* are of **0-1 year of Machine Learning Experience**.

<details><summary>code</summary>
<p>

```python
def get_df_for_multi_part_question(data, main_col, by_col):
    cols = get_columns(by_col) + [main_col]
    df = data[cols]
    df = (df.set_index(["Company_Category"]).stack().reset_index(name='Value'))
    del df['level_1']
    df.columns = [main_col, by_col]
    df = pd.crosstab([df[by_col]], [df['Company_Category']])
    df = df.reindex(df.sum(axis=1).sort_values().index)
    return df

q7_df = get_df_for_multi_part_question(data, "Company_Category", "Q7")
q9_df = get_df_for_multi_part_question(data, "Company_Category", "Q9")
q14_df = get_df_for_multi_part_question(data, "Company_Category", "Q14")
q16_df = get_df_for_multi_part_question(data, "Company_Category", "Q16")

f, ax = plt.subplots(nrows=2,
                     ncols=2,
                     figsize=(20, 20),
                     gridspec_kw={
                         'height_ratios': [5, 5],
                         'wspace': 0.4,
                         'hspace': 0.1
                     })

# ax[0,0]
(q9_df/data['Company_Category'].value_counts()).plot.barh(ax=ax[0, 0], fontsize=20, width=0.65)

# ax[0,1]
(q7_df/data['Company_Category'].value_counts()).plot.barh(ax=ax[0, 1], fontsize=20, width=0.65)

# ax[1,0]
(q14_df/data['Company_Category'].value_counts()).plot.barh(ax=ax[1, 0], fontsize=20, width=0.65)

# ax[1,1]
(q16_df/data['Company_Category'].value_counts()).plot.barh(ax=ax[1, 1], fontsize=20, width=0.65)

title = f.suptitle('R vs Python & SKLearn or Tensorflow/Keras/Pytorch', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_42_0.png)


🚀Highlights:
1. More *large size company* uses **Jupyter Notebook** comare to Startup & Mid size company.
2. Significant number of *large size company* uses **Notepad++**.
3. **SQL & R** are more used in *Large Size Company*.
4. **Scikit-Learn, Xgboost, LightGBM, Caret, Catboost** are more used in *Large Size Company*.
5. **Tensorflow, Keras, Pytorch** are more used in *Startups*.

## Work Opportunity Aspect

<details><summary>code</summary>
<p>

```python
df = pd.crosstab([data['Q21']], [data['Company_Category']])
df1 = df.apply(lambda r: r/r.sum(), axis=0)
df2 = df.apply(lambda r: r/r.sum(), axis=1)


def hide_axes(this_ax):
    this_ax.set_frame_on(False)
    this_ax.set_xticks([])
    this_ax.set_yticks([])
    return this_ax

f, ax = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(30, 20),
)

hide_axes(ax[0, 0])

df.apply(lambda r: r.sum(), axis=1).plot.barh(ax=ax[0, 1],
                                              fontsize=20,
                                              color='#9B5445')
total = len(data)
for p in ax[0, 1].patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax[0, 1].annotate(percentage, (x, y))

hide_axes(ax[0, 2])

df.plot.bar(ax=ax[1, 0], fontsize=20)
df1.transpose().plot.bar(ax=ax[1, 1], fontsize=20, stacked=True, colormap=cm.get_cmap("tab20", 20))
df2.plot.bar(ax=ax[1, 2], fontsize=20, stacked=True)
title = f.suptitle('len(ML Workforce)?', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_45_0.png)


🚀Highlights:
1. Mostly *startup* has **0-2 people** are responsible for the data science workloads.
2. **20+ People** included in *Large Size Company* for the data science workloads
3. **>50%** of *Mid size company* has **0-4 People** are responsible for the data science workloads.

<details><summary>code</summary>
<p>

```python
df = pd.crosstab([data['Q22']], [data['Company_Category']])
df1 = df.apply(lambda r: r / r.sum(), axis=0)
df2 = df.apply(lambda r: r / r.sum(), axis=1)


def hide_axes(this_ax):
    this_ax.set_frame_on(False)
    this_ax.set_xticks([])
    this_ax.set_yticks([])
    return this_ax


f, ax = plt.subplots(nrows=2,
                     ncols=3,
                     figsize=(30, 20),
                     gridspec_kw={
                         'height_ratios': [5, 5],
                         'wspace': 0.2,
                         'hspace': 0.1
                     })

hide_axes(ax[0, 0])

df = reindex_df(df)

df.apply(lambda r: r.sum(), axis=1).plot.barh(ax=ax[0, 1],
                                              fontsize=20,
                                              color='#9B5445')
total = len(data)
for p in ax[0, 1].patches:
    percentage = '{:.1f}%'.format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax[0, 1].annotate(percentage, (x, y))

hide_axes(ax[0, 2])

df = reindex_df(df, True)
df1 = reindex_df(df1, True)
df2 = reindex_df(df2, True)

df.plot.bar(ax=ax[1, 0], fontsize=20)
df1.transpose().plot.bar(ax=ax[1, 1],
                         fontsize=20,
                         colormap=cm.get_cmap("tab20", 20),
                         stacked=True)
df2.plot.bar(ax=ax[1, 2], fontsize=20, stacked=True)
title = f.suptitle('Do they incorporated Machine Learning?', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_47_0.png)


🚀Highlights:
1. **~30%** *Startups* are exploring ML methods and may one day put a model into production.
2. **~25%** *Large Size company* has have well established ML methods and models in production for more than 2 years.

<details><summary>code</summary>
<p>

```python
main_col = "Company_Category"
by_col = "Q23"
cols = get_columns(by_col) + [main_col]
df = data[cols]
df = (df.set_index(["Company_Category"]).stack().reset_index(name='Value'))
del df['level_1']
df.columns = [main_col, by_col]
df = pd.crosstab([df[by_col]], [df['Company_Category']])

df1 = df.apply(lambda r: r / r.sum(), axis=0)
df2 = df.apply(lambda r: r / r.sum(), axis=1)


def hide_axes(this_ax):
    this_ax.set_frame_on(False)
    this_ax.set_xticks([])
    this_ax.set_yticks([])
    return this_ax


f, ax = plt.subplots(nrows=2,
                     ncols=3,
                     figsize=(30, 20),
                     gridspec_kw={
                         'height_ratios': [5, 5],
                         'wspace': 0.1,
                         'hspace': 0.2
                     })

hide_axes(ax[0, 0])

df = reindex_df(df)

df.apply(lambda r: r.sum(), axis=1).plot.barh(ax=ax[0, 1],
                                              fontsize=20,
                                              color='#9B5445')
total = len(data)
for p in ax[0, 1].patches:
    percentage = '{:.1f}%'.format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax[0, 1].annotate(percentage, (x, y))

hide_axes(ax[0, 2])

df = reindex_df(df, True)
df1 = reindex_df(df1, True)
df2 = reindex_df(df2, True)

df.plot.bar(ax=ax[1, 0], fontsize=20)
(df/data['Company_Category'].value_counts()).plot.bar(ax=ax[1, 1],
             fontsize=20)
df2.plot.bar(ax=ax[1, 2], fontsize=20, stacked=True)
title = f.suptitle('What they doing?', fontsize=30)
```

</p>
</details>

![png](/assets/images/eda/output_49_0.png)


🚀Highlights:
1. **56%** of Companies are Analyzing and understanding data to influence product or business decisions.
2. **40%** of *Large size company* and **30%** of *Startup* are building prototypes to explore applying machine learning to new areas. 

## Salary Aspect

Let's us see how salary varies with the companay size. 

Starting with job role. To calculate salary part, I used reponse of `Q24` and took upper bound as thier salary for simlicity and NaN repaced with the mean value. Now with the use of groupy function of pandas, I able to calculate salary by job role and company category.

<details><summary>code</summary>
<p>

```python
salary_in_order = [
    "$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999",
    "5,000-7,499", "7,500-9,999", "10,000-14,999", "15,000-19,999",
    "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999",
    "50,000-59,999", "60,000-69,999", "70,000-79,999", "80,000-89,999",
    "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999",
    "200,000-249,999", "300,000-500,000", "> $500,000", "nan"
]

## Put NaN with mean

salary_in_value = [
    999, 1999, 2999, 3999, 4999, 7499, 9999, 14999, 19999, 24999, 29999, 39999,
    49999, 59999, 69999, 79999, 89999, 99999, 124999, 149999, 199999, 249999, 500000, 1000000, 46910
]

salary_lookup = dict(zip(salary_in_order, salary_in_value))

data['Q24_new'] = data['Q24'].astype(str)
data['Q24_new'] = data['Q24_new'].apply(lambda x: salary_lookup[x])
```


```python
def add_annotation(ax, text, xy, xytext, facecolor):
    ax.annotate(
        text,
        xy=xy,
        xycoords='data',
        fontsize=16,
        weight=None,
        xytext=xytext,
        textcoords='axes fraction',
        arrowprops=dict(facecolor=facecolor, shrink=0.05),
        horizontalalignment='right',
        verticalalignment='top',
    )
```


```python
df = data[['Company_Category','Q5', 'Q24_new']].groupby(['Company_Category','Q5']).describe()
df = df['Q24_new']
(df.style
.background_gradient(subset=['mean']))
```

</p>
</details>


<style  type="text/css" >
#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col1{
            background-color:  #89b1d4;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col1{
            background-color:  #056dac;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col1{
            background-color:  #a5bddb;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col1{
            background-color:  #1278b4;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col1{
            background-color:  #023c5f;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col1{
            background-color:  #045c90;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col1{
            background-color:  #1e80b8;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col1{
            background-color:  #023858;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col1{
            background-color:  #023a5b;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col1{
            background-color:  #4a98c5;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col1{
            background-color:  #045f95;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col1,#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col1{
            background-color:  #9ab8d8;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col1{
            background-color:  #034871;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col1{
            background-color:  #dedcec;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col1{
            background-color:  #65a3cb;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col1{
            background-color:  #0c74b2;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col1{
            background-color:  #78abd0;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col1{
            background-color:  #71a8ce;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col1{
            background-color:  #0569a5;
            color:  #f1f1f1;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col1,#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col1{
            background-color:  #b8c6e0;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col1{
            background-color:  #c9cee4;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col1{
            background-color:  #bcc7e1;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col1{
            background-color:  #e1dfed;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col1{
            background-color:  #fff7fb;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col1{
            background-color:  #a4bcda;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col1{
            background-color:  #faf2f8;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col1{
            background-color:  #c1cae2;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col1{
            background-color:  #75a9cf;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col1{
            background-color:  #b1c2de;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col1{
            background-color:  #e9e5f1;
            color:  #000000;
        }#T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col1{
            background-color:  #fef6fb;
            color:  #000000;
        }</style><table id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202" ><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>        <th class="col_heading level0 col1" >mean</th>        <th class="col_heading level0 col2" >std</th>        <th class="col_heading level0 col3" >min</th>        <th class="col_heading level0 col4" >25%</th>        <th class="col_heading level0 col5" >50%</th>        <th class="col_heading level0 col6" >75%</th>        <th class="col_heading level0 col7" >max</th>    </tr>    <tr>        <th class="index_name level0" >Company_Category</th>        <th class="index_name level1" >Q5</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level0_row0" class="row_heading level0 row0" rowspan=11>Large Size Company</th>
                        <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row0" class="row_heading level1 row0" >Business Analyst</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col0" class="data row0 col0" >317.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col1" class="data row0 col1" >48365.735</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col2" class="data row0 col2" >54009.599</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col3" class="data row0 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col4" class="data row0 col4" >9999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col5" class="data row0 col5" >39999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col6" class="data row0 col6" >69999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row0_col7" class="data row0 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row1" class="row_heading level1 row1" >DBA/Database Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col0" class="data row1 col0" >51.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col1" class="data row1 col1" >66438.451</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col2" class="data row1 col2" >140567.291</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col3" class="data row1 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col4" class="data row1 col4" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col5" class="data row1 col5" >39999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col6" class="data row1 col6" >69999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row1_col7" class="data row1 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row2" class="row_heading level1 row2" >Data Analyst</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col0" class="data row2 col0" >452.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col1" class="data row2 col1" >44258.460</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col2" class="data row2 col2" >50615.318</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col3" class="data row2 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col4" class="data row2 col4" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col5" class="data row2 col5" >24999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col6" class="data row2 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row2_col7" class="data row2 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row3" class="row_heading level1 row3" >Data Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col0" class="data row3 col0" >173.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col1" class="data row3 col1" >63913.451</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col2" class="data row3 col2" >89405.007</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col3" class="data row3 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col4" class="data row3 col4" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col5" class="data row3 col5" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col6" class="data row3 col6" >79999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row3_col7" class="data row3 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row4" class="row_heading level1 row4" >Data Scientist</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col0" class="data row4 col0" >978.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col1" class="data row4 col1" >79224.119</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col2" class="data row4 col2" >95688.852</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col3" class="data row4 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col4" class="data row4 col4" >19999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col5" class="data row4 col5" >54999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col6" class="data row4 col6" >124999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row4_col7" class="data row4 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row5" class="row_heading level1 row5" >Machine Learning Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col0" class="data row5 col0" >241.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col1" class="data row5 col1" >72179.232</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col2" class="data row5 col2" >115729.684</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col3" class="data row5 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col4" class="data row5 col4" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col5" class="data row5 col5" >39999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col6" class="data row5 col6" >89999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row5_col7" class="data row5 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row6" class="row_heading level1 row6" >Other</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col0" class="data row6 col0" >627.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col1" class="data row6 col1" >61932.820</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col2" class="data row6 col2" >91976.007</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col3" class="data row6 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col4" class="data row6 col4" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col5" class="data row6 col5" >39999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col6" class="data row6 col6" >79999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row6_col7" class="data row6 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row7" class="row_heading level1 row7" >Product/Project Manager</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col0" class="data row7 col0" >247.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col1" class="data row7 col1" >80123.745</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col2" class="data row7 col2" >74590.717</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col3" class="data row7 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col4" class="data row7 col4" >29999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col5" class="data row7 col5" >69999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col6" class="data row7 col6" >124999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row7_col7" class="data row7 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row8" class="row_heading level1 row8" >Research Scientist</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col0" class="data row8 col0" >340.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col1" class="data row8 col1" >79595.779</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col2" class="data row8 col2" >146005.220</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col3" class="data row8 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col4" class="data row8 col4" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col5" class="data row8 col5" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col6" class="data row8 col6" >82499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row8_col7" class="data row8 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row9" class="row_heading level1 row9" >Software Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col0" class="data row9 col0" >675.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col1" class="data row9 col1" >56162.613</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col2" class="data row9 col2" >111186.941</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col3" class="data row9 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col4" class="data row9 col4" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col5" class="data row9 col5" >24999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col6" class="data row9 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row9_col7" class="data row9 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row10" class="row_heading level1 row10" >Statistician</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col0" class="data row10 col0" >71.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col1" class="data row10 col1" >71181.634</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col2" class="data row10 col2" >81723.700</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col3" class="data row10 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col4" class="data row10 col4" >12499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col5" class="data row10 col5" >49999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col6" class="data row10 col6" >99999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row10_col7" class="data row10 col7" >500000.000</td>
            </tr>
            <tr>
                        <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level0_row11" class="row_heading level0 row11" rowspan=11>Mid Size Company</th>
                        <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row11" class="row_heading level1 row11" >Business Analyst</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col0" class="data row11 col0" >172.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col1" class="data row11 col1" >45950.942</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col2" class="data row11 col2" >85187.530</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col3" class="data row11 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col4" class="data row11 col4" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col5" class="data row11 col5" >24999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col6" class="data row11 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row11_col7" class="data row11 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row12" class="row_heading level1 row12" >DBA/Database Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col0" class="data row12 col0" >28.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col1" class="data row12 col1" >76641.893</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col2" class="data row12 col2" >187652.780</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col3" class="data row12 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col4" class="data row12 col4" >3749.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col5" class="data row12 col5" >24999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col6" class="data row12 col6" >67499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row12_col7" class="data row12 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row13" class="row_heading level1 row13" >Data Analyst</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col0" class="data row13 col0" >391.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col1" class="data row13 col1" >33407.350</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col2" class="data row13 col2" >38283.207</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col3" class="data row13 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col4" class="data row13 col4" >2999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col5" class="data row13 col5" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col6" class="data row13 col6" >49999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row13_col7" class="data row13 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row14" class="row_heading level1 row14" >Data Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col0" class="data row14 col0" >120.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col1" class="data row14 col1" >52999.467</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col2" class="data row14 col2" >65660.127</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col3" class="data row14 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col4" class="data row14 col4" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col5" class="data row14 col5" >29999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col6" class="data row14 col6" >79999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row14_col7" class="data row14 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row15" class="row_heading level1 row15" >Data Scientist</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col0" class="data row15 col0" >595.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col1" class="data row15 col1" >64784.402</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col2" class="data row15 col2" >92530.380</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col3" class="data row15 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col4" class="data row15 col4" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col5" class="data row15 col5" >39999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col6" class="data row15 col6" >89999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row15_col7" class="data row15 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row16" class="row_heading level1 row16" >Machine Learning Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col0" class="data row16 col0" >223.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col1" class="data row16 col1" >50853.206</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col2" class="data row16 col2" >65934.683</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col3" class="data row16 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col4" class="data row16 col4" >3999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col5" class="data row16 col5" >29999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col6" class="data row16 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row16_col7" class="data row16 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row17" class="row_heading level1 row17" >Other</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col0" class="data row17 col0" >457.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col1" class="data row17 col1" >51604.140</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col2" class="data row17 col2" >94100.363</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col3" class="data row17 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col4" class="data row17 col4" >3999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col5" class="data row17 col5" >19999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col6" class="data row17 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row17_col7" class="data row17 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row18" class="row_heading level1 row18" >Product/Project Manager</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col0" class="data row18 col0" >181.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col1" class="data row18 col1" >67920.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col2" class="data row18 col2" >95610.307</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col3" class="data row18 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col4" class="data row18 col4" >9999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col5" class="data row18 col5" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col6" class="data row18 col6" >89999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row18_col7" class="data row18 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row19" class="row_heading level1 row19" >Research Scientist</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col0" class="data row19 col0" >338.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col1" class="data row19 col1" >41034.728</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col2" class="data row19 col2" >50834.413</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col3" class="data row19 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col4" class="data row19 col4" >2999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col5" class="data row19 col5" >19999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col6" class="data row19 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row19_col7" class="data row19 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row20" class="row_heading level1 row20" >Software Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col0" class="data row20 col0" >439.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col1" class="data row20 col1" >45799.708</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col2" class="data row20 col2" >91669.747</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col3" class="data row20 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col4" class="data row20 col4" >3999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col5" class="data row20 col5" >24999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col6" class="data row20 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row20_col7" class="data row20 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row21" class="row_heading level1 row21" >Statistician</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col0" class="data row21 col0" >79.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col1" class="data row21 col1" >38173.962</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col2" class="data row21 col2" >45869.644</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col3" class="data row21 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col4" class="data row21 col4" >3999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col5" class="data row21 col5" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col6" class="data row21 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row21_col7" class="data row21 col7" >199999.000</td>
            </tr>
            <tr>
                        <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level0_row22" class="row_heading level0 row22" rowspan=11>Startup</th>
                        <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row22" class="row_heading level1 row22" >Business Analyst</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col0" class="data row22 col0" >223.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col1" class="data row22 col1" >40505.682</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col2" class="data row22 col2" >89929.708</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col3" class="data row22 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col4" class="data row22 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col5" class="data row22 col5" >9999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col6" class="data row22 col6" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row22_col7" class="data row22 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row23" class="row_heading level1 row23" >DBA/Database Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col0" class="data row23 col0" >36.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col1" class="data row23 col1" >32688.500</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col2" class="data row23 col2" >41397.226</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col3" class="data row23 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col4" class="data row23 col4" >1999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col5" class="data row23 col5" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col6" class="data row23 col6" >49999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row23_col7" class="data row23 col7" >149999.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row24" class="row_heading level1 row24" >Data Analyst</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col0" class="data row24 col0" >492.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col1" class="data row24 col1" >22609.541</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col2" class="data row24 col2" >55265.584</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col3" class="data row24 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col4" class="data row24 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col5" class="data row24 col5" >2999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col6" class="data row24 col6" >29999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row24_col7" class="data row24 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row25" class="row_heading level1 row25" >Data Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col0" class="data row25 col0" >96.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col1" class="data row25 col1" >44424.812</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col2" class="data row25 col2" >72132.595</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col3" class="data row25 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col4" class="data row25 col4" >1749.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col5" class="data row25 col5" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col6" class="data row25 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row25_col7" class="data row25 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row26" class="row_heading level1 row26" >Data Scientist</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col0" class="data row26 col0" >937.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col1" class="data row26 col1" >41081.572</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col2" class="data row26 col2" >97318.933</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col3" class="data row26 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col4" class="data row26 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col5" class="data row26 col5" >7499.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col6" class="data row26 col6" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row26_col7" class="data row26 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row27" class="row_heading level1 row27" >Machine Learning Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col0" class="data row27 col0" >528.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col1" class="data row27 col1" >24833.379</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col2" class="data row27 col2" >47422.643</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col3" class="data row27 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col4" class="data row27 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col5" class="data row27 col5" >2999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col6" class="data row27 col6" >39999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row27_col7" class="data row27 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row28" class="row_heading level1 row28" >Other</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col0" class="data row28 col0" >522.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col1" class="data row28 col1" >39493.249</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col2" class="data row28 col2" >86954.872</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col3" class="data row28 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col4" class="data row28 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col5" class="data row28 col5" >14999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col6" class="data row28 col6" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row28_col7" class="data row28 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row29" class="row_heading level1 row29" >Product/Project Manager</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col0" class="data row29 col0" >198.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col1" class="data row29 col1" >51205.283</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col2" class="data row29 col2" >89628.317</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col3" class="data row29 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col4" class="data row29 col4" >2999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col5" class="data row29 col5" >24999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col6" class="data row29 col6" >59999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row29_col7" class="data row29 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row30" class="row_heading level1 row30" >Research Scientist</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col0" class="data row30 col0" >410.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col1" class="data row30 col1" >42202.039</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col2" class="data row30 col2" >108386.977</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col3" class="data row30 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col4" class="data row30 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col5" class="data row30 col5" >9999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col6" class="data row30 col6" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row30_col7" class="data row30 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row31" class="row_heading level1 row31" >Software Engineer</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col0" class="data row31 col0" >650.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col1" class="data row31 col1" >30492.631</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col2" class="data row31 col2" >43956.411</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col3" class="data row31 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col4" class="data row31 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col5" class="data row31 col5" >9999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col6" class="data row31 col6" >46910.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row31_col7" class="data row31 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202level1_row32" class="row_heading level1 row32" >Statistician</th>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col0" class="data row32 col0" >116.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col1" class="data row32 col1" >23055.991</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col2" class="data row32 col2" >59313.111</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col3" class="data row32 col3" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col4" class="data row32 col4" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col5" class="data row32 col5" >999.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col6" class="data row32 col6" >26249.000</td>
                        <td id="T_b34da3a0_5e33_11eb_b4d5_0242ac130202row32_col7" class="data row32 col7" >500000.000</td>
            </tr>
    </tbody></table>



🚀Highlights:
1. Average Salary in *Large Size Company* are **Research Scientist(77741) > Data Scientist(77737) > Machine Learning Engineer(73560) > Statistician(71181) > Data Engineer(63187) > Data Analyst (44106)**

2. Average Salary in *Mid Size Company* are **Data Scientist(64432) > Data Engineer(53419) > Machine Learning Engineer(51064) > Research Scientist(40375) > Statistician(38650) > Data Analyst(33369)**

3. Average Salary in *Startup* are **Data Scientist(41170) > Research Scientist(41550) > Data Engineer(39629) > Machine Learning Engineer(24921) > Statistician(23247) > Data Analyst(22645)**

4. **Product/Project Manager** get more money

<details><summary>code</summary>
<p>

```python
%matplotlib inline
%config InlineBackend.figure_format='retina'

index_cols = ['Software Engineer', 'DBA/Database Engineer', 'Data Engineer', 'Machine Learning Engineer', 'Statistician', 'Data Analyst', 'Data Scientist', 'Research Scientist', 'Business Analyst', 'Product/Project Manager', 'Other']


data['Q5'] = pd.Categorical(data['Q5'], categories=index_cols, ordered=True)

df = pd.crosstab([data['Q24'], data['Company_Category'], data['Q5']],
                 []).reset_index()
df = df.rename(columns={'__dummy__': 'size'})

df1 = pd.crosstab([data['Company_Category'], data['Q5']],
                 []).reset_index()
df1 = df1.rename(columns={'__dummy__': 'total_size'})

df = df.merge(df1, how='inner', on=['Company_Category', 'Q5'])
df['percentage'] = df['size']/df['total_size']

palette = sns.color_palette("tab20", len(data['Q5'].unique()))
lp = sns.relplot(
    data=df,
    x="Q24",
    y="percentage",
    hue="Q5",
    col="Company_Category",
    kind="scatter",
    height=5,
    aspect=.75,
    palette=palette,
    facet_kws=dict(sharex=False),
)
lp.set_xticklabels(fontsize=10, rotation=90, step=2)
```

</p>
</details>

![png](/assets/images/eda/output_57_1.png)


🚀Highlights:
1. **~55% staticians** of *Startup* has salary **b/w 0-999**. 
2. **~40% Data scientist, Analyst and machine learning developer** of *Startups* has salary **b/w 0-999**.
4. Overall *Startups* give **less money to staticians, Data scientist, Analyst and machine learning developers** compare to Large & Mid Size company


Now Let's see how salary varies with the gender categories.

<details><summary>code</summary>
<p>

```python
data_sub = data[data['Q2'].isin(['Man','Woman'])]
df = data_sub[['Company_Category','Q2', 'Q24_new']].groupby(['Company_Category','Q2']).describe()
df = df['Q24_new']
(df.style
.background_gradient(subset=['mean']))
```

</p>
</details>


<style  type="text/css" >
#T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col1{
            background-color:  #023858;
            color:  #f1f1f1;
        }#T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col1{
            background-color:  #2d8abd;
            color:  #000000;
        }#T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col1{
            background-color:  #2c89bd;
            color:  #000000;
        }#T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col1{
            background-color:  #96b6d7;
            color:  #000000;
        }#T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col1{
            background-color:  #bdc8e1;
            color:  #000000;
        }#T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col1{
            background-color:  #fff7fb;
            color:  #000000;
        }</style><table id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202" ><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>        <th class="col_heading level0 col1" >mean</th>        <th class="col_heading level0 col2" >std</th>        <th class="col_heading level0 col3" >min</th>        <th class="col_heading level0 col4" >25%</th>        <th class="col_heading level0 col5" >50%</th>        <th class="col_heading level0 col6" >75%</th>        <th class="col_heading level0 col7" >max</th>    </tr>    <tr>        <th class="index_name level0" >Company_Category</th>        <th class="index_name level1" >Q2</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level0_row0" class="row_heading level0 row0" rowspan=2>Large Size Company</th>
                        <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level1_row0" class="row_heading level1 row0" >Man</th>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col0" class="data row0 col0" >3493.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col1" class="data row0 col1" >67140.745</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col2" class="data row0 col2" >97001.144</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col3" class="data row0 col3" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col4" class="data row0 col4" >14999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col5" class="data row0 col5" >46910.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col6" class="data row0 col6" >89999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row0_col7" class="data row0 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level1_row1" class="row_heading level1 row1" >Woman</th>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col0" class="data row1 col0" >608.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col1" class="data row1 col1" >51888.600</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col2" class="data row1 col2" >80491.320</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col3" class="data row1 col3" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col4" class="data row1 col4" >7499.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col5" class="data row1 col5" >29999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col6" class="data row1 col6" >69999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row1_col7" class="data row1 col7" >1000000.000</td>
            </tr>
            <tr>
                        <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level0_row2" class="row_heading level0 row2" rowspan=2>Mid Size Company</th>
                        <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level1_row2" class="row_heading level1 row2" >Man</th>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col0" class="data row2 col0" >2440.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col1" class="data row2 col1" >51975.483</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col2" class="data row2 col2" >82024.324</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col3" class="data row2 col3" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col4" class="data row2 col4" >7499.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col5" class="data row2 col5" >29999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col6" class="data row2 col6" >69999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row2_col7" class="data row2 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level1_row3" class="row_heading level1 row3" >Woman</th>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col0" class="data row3 col0" >532.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col1" class="data row3 col1" >41772.118</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col2" class="data row3 col2" >80058.746</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col3" class="data row3 col3" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col4" class="data row3 col4" >1999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col5" class="data row3 col5" >14999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col6" class="data row3 col6" >49999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row3_col7" class="data row3 col7" >1000000.000</td>
            </tr>
            <tr>
                        <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level0_row4" class="row_heading level0 row4" rowspan=2>Startup</th>
                        <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level1_row4" class="row_heading level1 row4" >Man</th>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col0" class="data row4 col0" >3439.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col1" class="data row4 col1" >37113.458</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col2" class="data row4 col2" >83085.053</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col3" class="data row4 col3" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col4" class="data row4 col4" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col5" class="data row4 col5" >9999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col6" class="data row4 col6" >46910.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row4_col7" class="data row4 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202level1_row5" class="row_heading level1 row5" >Woman</th>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col0" class="data row5 col0" >706.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col1" class="data row5 col1" >23793.449</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col2" class="data row5 col2" >51965.554</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col3" class="data row5 col3" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col4" class="data row5 col4" >999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col5" class="data row5 col5" >1999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col6" class="data row5 col6" >39999.000</td>
                        <td id="T_b56dd1d2_5e33_11eb_b4d5_0242ac130202row5_col7" class="data row5 col7" >1000000.000</td>
            </tr>
    </tbody></table>


<details><summary>code</summary>
<p>

```python
data_sub = data[data['Q2'].isin(['Man','Woman'])]

df = pd.crosstab([data_sub['Q24'], data_sub['Company_Category'], data_sub['Q2']],
                 []).reset_index()
df = df.rename(columns={'__dummy__': 'size'})

df1 = pd.crosstab([data_sub['Company_Category'], data_sub['Q2']],
                 []).reset_index()
df1 = df1.rename(columns={'__dummy__': 'total_size'})

df = df.merge(df1, how='inner', on=['Company_Category', 'Q2'])
df['percentage'] = df['size']/df['total_size']

palette = sns.color_palette("Paired", len(data_sub['Q2'].unique()))
lp = sns.relplot(
    data=df,
    x="Q24",
    y="percentage",
    hue="Q2",
    col="Company_Category",
    kind="line",
    height=5,
    aspect=.75,
    palette=palette,
    facet_kws=dict(sharex=False),
)

lp.set_xticklabels(fontsize=10, rotation=90, step=2)
```

</p>
</details>

![png](/assets/images/eda/output_61_1.png)


🚀Highlights:
1. The average salary of a **Man is greater than average salary of woman**.
2. On Average Man earns **22%** more than Woman in *Large Size Company* where as in *Startups* difference is **35%**

Now Let's see how salary varies with the highest education taken by respondent.

<details><summary>code</summary>
<p>

```python
df = data[['Company_Category','Q4', 'Q24_new']].groupby(['Company_Category','Q4']).describe()
df = df['Q24_new']
(df.style
.background_gradient(subset=['mean']))
```

</p>
</details>


<style  type="text/css" >
#T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col1{
            background-color:  #cdd0e5;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col1{
            background-color:  #4e9ac6;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col1{
            background-color:  #b1c2de;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col1{
            background-color:  #3d93c2;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col1{
            background-color:  #023858;
            color:  #f1f1f1;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col1{
            background-color:  #83afd3;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col1{
            background-color:  #2a88bc;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col1,#T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col1{
            background-color:  #e2dfee;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col1{
            background-color:  #c0c9e2;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col1{
            background-color:  #84b0d3;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col1{
            background-color:  #529bc7;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col1{
            background-color:  #d9d8ea;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col1{
            background-color:  #fff7fb;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col1{
            background-color:  #f5eef6;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col1{
            background-color:  #f2ecf5;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col1{
            background-color:  #eee8f3;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col1{
            background-color:  #e5e1ef;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col1{
            background-color:  #93b5d6;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col1{
            background-color:  #eee9f3;
            color:  #000000;
        }#T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col1{
            background-color:  #f7f0f7;
            color:  #000000;
        }</style><table id="T_b7590598_5e33_11eb_b4d5_0242ac130202" ><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>        <th class="col_heading level0 col1" >mean</th>        <th class="col_heading level0 col2" >std</th>        <th class="col_heading level0 col3" >min</th>        <th class="col_heading level0 col4" >25%</th>        <th class="col_heading level0 col5" >50%</th>        <th class="col_heading level0 col6" >75%</th>        <th class="col_heading level0 col7" >max</th>    </tr>    <tr>        <th class="index_name level0" >Company_Category</th>        <th class="index_name level1" >Q4</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level0_row0" class="row_heading level0 row0" rowspan=7>Large Size Company</th>
                        <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row0" class="row_heading level1 row0" >No formal education past high school</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col0" class="data row0 col0" >27.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col1" class="data row0 col1" >41655.778</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col2" class="data row0 col2" >31099.823</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col3" class="data row0 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col4" class="data row0 col4" >19999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col5" class="data row0 col5" >29999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col6" class="data row0 col6" >64999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row0_col7" class="data row0 col7" >99999.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row1" class="row_heading level1 row1" >Some college/university study without earning a bachelor’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col0" class="data row1 col0" >82.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col1" class="data row1 col1" >64647.110</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col2" class="data row1 col2" >72915.418</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col3" class="data row1 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col4" class="data row1 col4" >14999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col5" class="data row1 col5" >46910.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col6" class="data row1 col6" >97499.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row1_col7" class="data row1 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row2" class="row_heading level1 row2" >Bachelor’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col0" class="data row2 col0" >1164.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col1" class="data row2 col1" >47513.050</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col2" class="data row2 col2" >71769.927</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col3" class="data row2 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col4" class="data row2 col4" >7499.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col5" class="data row2 col5" >24999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col6" class="data row2 col6" >59999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row2_col7" class="data row2 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row3" class="row_heading level1 row3" >Master’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col0" class="data row3 col0" >2024.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col1" class="data row3 col1" >67249.131</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col2" class="data row3 col2" >93095.598</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col3" class="data row3 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col4" class="data row3 col4" >14999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col5" class="data row3 col5" >46910.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col6" class="data row3 col6" >89999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row3_col7" class="data row3 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row4" class="row_heading level1 row4" >Doctoral degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col0" class="data row4 col0" >651.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col1" class="data row4 col1" >95438.639</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col2" class="data row4 col2" >130908.668</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col3" class="data row4 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col4" class="data row4 col4" >24999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col5" class="data row4 col5" >59999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col6" class="data row4 col6" >124999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row4_col7" class="data row4 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row5" class="row_heading level1 row5" >Professional degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col0" class="data row5 col0" >173.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col1" class="data row5 col1" >56355.150</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col2" class="data row5 col2" >107037.022</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col3" class="data row5 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col4" class="data row5 col4" >7499.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col5" class="data row5 col5" >24999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col6" class="data row5 col6" >49999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row5_col7" class="data row5 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row6" class="row_heading level1 row6" >I prefer not to answer</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col0" class="data row6 col0" >51.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col1" class="data row6 col1" >70274.627</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col2" class="data row6 col2" >194255.147</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col3" class="data row6 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col4" class="data row6 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col5" class="data row6 col5" >14999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col6" class="data row6 col6" >49999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row6_col7" class="data row6 col7" >1000000.000</td>
            </tr>
            <tr>
                        <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level0_row7" class="row_heading level0 row7" rowspan=7>Mid Size Company</th>
                        <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row7" class="row_heading level1 row7" >No formal education past high school</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col0" class="data row7 col0" >29.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col1" class="data row7 col1" >34909.724</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col2" class="data row7 col2" >36946.337</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col3" class="data row7 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col4" class="data row7 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col5" class="data row7 col5" >19999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col6" class="data row7 col6" >49999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row7_col7" class="data row7 col7" >124999.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row8" class="row_heading level1 row8" >Some college/university study without earning a bachelor’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col0" class="data row8 col0" >109.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col1" class="data row8 col1" >44324.385</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col2" class="data row8 col2" >64384.962</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col3" class="data row8 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col4" class="data row8 col4" >2999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col5" class="data row8 col5" >24999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col6" class="data row8 col6" >59999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row8_col7" class="data row8 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row9" class="row_heading level1 row9" >Bachelor’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col0" class="data row9 col0" >762.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col1" class="data row9 col1" >35174.307</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col2" class="data row9 col2" >43905.655</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col3" class="data row9 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col4" class="data row9 col4" >3999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col5" class="data row9 col5" >14999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col6" class="data row9 col6" >49999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row9_col7" class="data row9 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row10" class="row_heading level1 row10" >Master’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col0" class="data row10 col0" >1371.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col1" class="data row10 col1" >56075.032</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col2" class="data row10 col2" >92992.454</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col3" class="data row10 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col4" class="data row10 col4" >7499.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col5" class="data row10 col5" >29999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col6" class="data row10 col6" >69999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row10_col7" class="data row10 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row11" class="row_heading level1 row11" >Doctoral degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col0" class="data row11 col0" >568.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col1" class="data row11 col1" >63939.639</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col2" class="data row11 col2" >99034.190</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col3" class="data row11 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col4" class="data row11 col4" >4999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col5" class="data row11 col5" >39999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col6" class="data row11 col6" >79999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row11_col7" class="data row11 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row12" class="row_heading level1 row12" >Professional degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col0" class="data row12 col0" >142.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col1" class="data row12 col1" >38251.655</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col2" class="data row12 col2" >47191.591</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col3" class="data row12 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col4" class="data row12 col4" >2999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col5" class="data row12 col5" >14999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col6" class="data row12 col6" >59999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row12_col7" class="data row12 col7" >199999.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row13" class="row_heading level1 row13" >I prefer not to answer</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col0" class="data row13 col0" >42.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col1" class="data row13 col1" >22666.976</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col2" class="data row13 col2" >26273.561</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col3" class="data row13 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col4" class="data row13 col4" >1999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col5" class="data row13 col5" >12499.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col6" class="data row13 col6" >45182.250</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row13_col7" class="data row13 col7" >124999.000</td>
            </tr>
            <tr>
                        <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level0_row14" class="row_heading level0 row14" rowspan=7>Startup</th>
                        <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row14" class="row_heading level1 row14" >No formal education past high school</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col0" class="data row14 col0" >57.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col1" class="data row14 col1" >27503.088</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col2" class="data row14 col2" >44475.788</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col3" class="data row14 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col4" class="data row14 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col5" class="data row14 col5" >3999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col6" class="data row14 col6" >39999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row14_col7" class="data row14 col7" >199999.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row15" class="row_heading level1 row15" >Some college/university study without earning a bachelor’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col0" class="data row15 col0" >227.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col1" class="data row15 col1" >28691.454</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col2" class="data row15 col2" >52773.586</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col3" class="data row15 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col4" class="data row15 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col5" class="data row15 col5" >2999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col6" class="data row15 col6" >46910.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row15_col7" class="data row15 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row16" class="row_heading level1 row16" >Bachelor’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col0" class="data row16 col0" >1334.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col1" class="data row16 col1" >31005.221</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col2" class="data row16 col2" >92367.311</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col3" class="data row16 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col4" class="data row16 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col5" class="data row16 col5" >2999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col6" class="data row16 col6" >39999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row16_col7" class="data row16 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row17" class="row_heading level1 row17" >Master’s degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col0" class="data row17 col0" >1743.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col1" class="data row17 col1" >34155.404</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col2" class="data row17 col2" >55711.309</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col3" class="data row17 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col4" class="data row17 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col5" class="data row17 col5" >9999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col6" class="data row17 col6" >46910.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row17_col7" class="data row17 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row18" class="row_heading level1 row18" >Doctoral degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col0" class="data row18 col0" >573.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col1" class="data row18 col1" >53573.141</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col2" class="data row18 col2" >115999.994</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col3" class="data row18 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col4" class="data row18 col4" >1999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col5" class="data row18 col5" >19999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col6" class="data row18 col6" >59999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row18_col7" class="data row18 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row19" class="row_heading level1 row19" >Professional degree</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col0" class="data row19 col0" >185.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col1" class="data row19 col1" >30825.957</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col2" class="data row19 col2" >44857.348</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col3" class="data row19 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col4" class="data row19 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col5" class="data row19 col5" >9999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col6" class="data row19 col6" >46910.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row19_col7" class="data row19 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_b7590598_5e33_11eb_b4d5_0242ac130202level1_row20" class="row_heading level1 row20" >I prefer not to answer</th>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col0" class="data row20 col0" >89.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col1" class="data row20 col1" >26749.056</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col2" class="data row20 col2" >63057.663</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col3" class="data row20 col3" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col4" class="data row20 col4" >999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col5" class="data row20 col5" >2999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col6" class="data row20 col6" >29999.000</td>
                        <td id="T_b7590598_5e33_11eb_b4d5_0242ac130202row20_col7" class="data row20 col7" >500000.000</td>
            </tr>
    </tbody></table>


<details><summary>code</summary>
<p>

```python
df = pd.crosstab([data['Q24'], data['Company_Category'], data['Q4']],
                 []).reset_index()
df = df.rename(columns={'__dummy__': 'size'})

df1 = pd.crosstab([data['Company_Category'], data['Q4']],
                 []).reset_index()
df1 = df1.rename(columns={'__dummy__': 'total_size'})

df = df.merge(df1, how='inner', on=['Company_Category', 'Q4'])
df['percentage'] = df['size']/df['total_size']

palette = sns.color_palette("Paired", len(data['Q4'].unique()))
lp = sns.relplot(
    data=df,
    x="Q24",
    y="percentage",
    hue="Q4",
    col="Company_Category",
    kind="scatter",
    height=5,
    aspect=.75,
    palette=palette,
    facet_kws=dict(sharex=False),
)

lp.set_xticklabels(fontsize=10, rotation=90, step=2)
```

</p>
</details>

![png](/assets/images/eda/output_65_1.png)


🚀Highlights:
1. The average salary of a **Doctoral degree is greater**.
2. There is **very small** difference in **avg Salary of masters and bachelors** in *Startups*, where as **large difference** in *Large and Mid Size Company*.
3. Avg salary for **Professional degree** holder in *Startups* is **less than bachelors** where as it is more in *Large and Mid Size Company*.

<details><summary>code</summary>
<p>

```python
df = data[['Company_Category','Q6', 'Q24_new']].groupby(['Company_Category','Q6']).describe()
df = df['Q24_new']
(df.style
.background_gradient(subset=['mean']))
```

</p>
</details>


<style  type="text/css" >
#T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col1{
            background-color:  #c5cce3;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col1{
            background-color:  #dad9ea;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col1{
            background-color:  #dcdaeb;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col1{
            background-color:  #b3c3de;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col1{
            background-color:  #509ac6;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col1{
            background-color:  #045f95;
            color:  #f1f1f1;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col1{
            background-color:  #023858;
            color:  #f1f1f1;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col1{
            background-color:  #dddbec;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col1{
            background-color:  #ebe6f2;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col1{
            background-color:  #eee8f3;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col1{
            background-color:  #d6d6e9;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col1{
            background-color:  #9cb9d9;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col1{
            background-color:  #569dc8;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col1{
            background-color:  #045b8e;
            color:  #f1f1f1;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col1{
            background-color:  #fdf5fa;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col1{
            background-color:  #fff7fb;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col1{
            background-color:  #fbf4f9;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col1{
            background-color:  #efe9f3;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col1{
            background-color:  #d5d5e8;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col1{
            background-color:  #7dacd1;
            color:  #000000;
        }#T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col1{
            background-color:  #2685bb;
            color:  #000000;
        }</style><table id="T_b8f75012_5e33_11eb_b4d5_0242ac130202" ><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>        <th class="col_heading level0 col1" >mean</th>        <th class="col_heading level0 col2" >std</th>        <th class="col_heading level0 col3" >min</th>        <th class="col_heading level0 col4" >25%</th>        <th class="col_heading level0 col5" >50%</th>        <th class="col_heading level0 col6" >75%</th>        <th class="col_heading level0 col7" >max</th>    </tr>    <tr>        <th class="index_name level0" >Company_Category</th>        <th class="index_name level1" >Q6</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level0_row0" class="row_heading level0 row0" rowspan=7>Large Size Company</th>
                        <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row0" class="row_heading level1 row0" >I have never written code</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col0" class="data row0 col0" >222.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col1" class="data row0 col1" >45005.248</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col2" class="data row0 col2" >98834.930</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col3" class="data row0 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col4" class="data row0 col4" >7499.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col5" class="data row0 col5" >24999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col6" class="data row0 col6" >46910.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row0_col7" class="data row0 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row1" class="row_heading level1 row1" >< 1 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col0" class="data row1 col0" >384.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col1" class="data row1 col1" >37846.539</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col2" class="data row1 col2" >56622.404</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col3" class="data row1 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col4" class="data row1 col4" >3999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col5" class="data row1 col5" >14999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col6" class="data row1 col6" >49999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row1_col7" class="data row1 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row2" class="row_heading level1 row2" >1-2 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col0" class="data row2 col0" >604.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col1" class="data row2 col1" >37158.487</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col2" class="data row2 col2" >45042.416</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col3" class="data row2 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col4" class="data row2 col4" >4999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col5" class="data row2 col5" >14999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col6" class="data row2 col6" >49999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row2_col7" class="data row2 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row3" class="row_heading level1 row3" >3-5 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col0" class="data row3 col0" >968.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col1" class="data row3 col1" >50016.803</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col2" class="data row3 col2" >79062.026</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col3" class="data row3 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col4" class="data row3 col4" >7499.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col5" class="data row3 col5" >29999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col6" class="data row3 col6" >69999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row3_col7" class="data row3 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row4" class="row_heading level1 row4" >5-10 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col0" class="data row4 col0" >854.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col1" class="data row4 col1" >71460.430</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col2" class="data row4 col2" >94127.307</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col3" class="data row4 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col4" class="data row4 col4" >19999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col5" class="data row4 col5" >49999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col6" class="data row4 col6" >99999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row4_col7" class="data row4 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row5" class="row_heading level1 row5" >10-20 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col0" class="data row5 col0" >660.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col1" class="data row5 col1" >96566.029</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col2" class="data row5 col2" >129544.549</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col3" class="data row5 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col4" class="data row5 col4" >29999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col5" class="data row5 col5" >69999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col6" class="data row5 col6" >124999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row5_col7" class="data row5 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row6" class="row_heading level1 row6" >20+ years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col0" class="data row6 col0" >480.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col1" class="data row6 col1" >110754.558</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col2" class="data row6 col2" >128806.812</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col3" class="data row6 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col4" class="data row6 col4" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col5" class="data row6 col5" >79999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col6" class="data row6 col6" >149999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row6_col7" class="data row6 col7" >1000000.000</td>
            </tr>
            <tr>
                        <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level0_row7" class="row_heading level0 row7" rowspan=7>Mid Size Company</th>
                        <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row7" class="row_heading level1 row7" >I have never written code</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col0" class="data row7 col0" >162.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col1" class="data row7 col1" >37011.617</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col2" class="data row7 col2" >92715.013</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col3" class="data row7 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col4" class="data row7 col4" >1999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col5" class="data row7 col5" >14999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col6" class="data row7 col6" >46910.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row7_col7" class="data row7 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row8" class="row_heading level1 row8" >< 1 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col0" class="data row8 col0" >373.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col1" class="data row8 col1" >31155.094</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col2" class="data row8 col2" >83376.832</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col3" class="data row8 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col4" class="data row8 col4" >1999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col5" class="data row8 col5" >7499.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col6" class="data row8 col6" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row8_col7" class="data row8 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row9" class="row_heading level1 row9" >1-2 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col0" class="data row9 col0" >487.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col1" class="data row9 col1" >29824.735</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col2" class="data row9 col2" >71755.159</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col3" class="data row9 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col4" class="data row9 col4" >2999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col5" class="data row9 col5" >9999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col6" class="data row9 col6" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row9_col7" class="data row9 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row10" class="row_heading level1 row10" >3-5 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col0" class="data row10 col0" >691.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col1" class="data row10 col1" >39623.211</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col2" class="data row10 col2" >45096.085</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col3" class="data row10 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col4" class="data row10 col4" >4999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col5" class="data row10 col5" >24999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col6" class="data row10 col6" >59999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row10_col7" class="data row10 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row11" class="row_heading level1 row11" >5-10 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col0" class="data row11 col0" >557.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col1" class="data row11 col1" >55832.594</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col2" class="data row11 col2" >79480.477</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col3" class="data row11 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col4" class="data row11 col4" >9999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col5" class="data row11 col5" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col6" class="data row11 col6" >69999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row11_col7" class="data row11 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row12" class="row_heading level1 row12" >10-20 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col0" class="data row12 col0" >450.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col1" class="data row12 col1" >70391.400</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col2" class="data row12 col2" >80720.265</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col3" class="data row12 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col4" class="data row12 col4" >19999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col5" class="data row12 col5" >49999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col6" class="data row12 col6" >89999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row12_col7" class="data row12 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row13" class="row_heading level1 row13" >20+ years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col0" class="data row13 col0" >303.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col1" class="data row13 col1" >98782.152</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col2" class="data row13 col2" >120978.358</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col3" class="data row13 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col4" class="data row13 col4" >34999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col5" class="data row13 col5" >79999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col6" class="data row13 col6" >124999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row13_col7" class="data row13 col7" >1000000.000</td>
            </tr>
            <tr>
                        <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level0_row14" class="row_heading level0 row14" rowspan=7>Startup</th>
                        <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row14" class="row_heading level1 row14" >I have never written code</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col0" class="data row14 col0" >273.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col1" class="data row14 col1" >20271.674</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col2" class="data row14 col2" >32638.859</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col3" class="data row14 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col4" class="data row14 col4" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col5" class="data row14 col5" >1999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col6" class="data row14 col6" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row14_col7" class="data row14 col7" >199999.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row15" class="row_heading level1 row15" >< 1 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col0" class="data row15 col0" >649.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col1" class="data row15 col1" >19139.891</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col2" class="data row15 col2" >53885.739</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col3" class="data row15 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col4" class="data row15 col4" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col5" class="data row15 col5" >1999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col6" class="data row15 col6" >24999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row15_col7" class="data row15 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row16" class="row_heading level1 row16" >1-2 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col0" class="data row16 col0" >945.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col1" class="data row16 col1" >21358.325</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col2" class="data row16 col2" >64684.753</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col3" class="data row16 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col4" class="data row16 col4" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col5" class="data row16 col5" >1999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col6" class="data row16 col6" >24999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row16_col7" class="data row16 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row17" class="row_heading level1 row17" >3-5 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col0" class="data row17 col0" >932.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col1" class="data row17 col1" >29021.101</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col2" class="data row17 col2" >81392.880</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col3" class="data row17 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col4" class="data row17 col4" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col5" class="data row17 col5" >4999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col6" class="data row17 col6" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row17_col7" class="data row17 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row18" class="row_heading level1 row18" >5-10 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col0" class="data row18 col0" >576.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col1" class="data row18 col1" >40083.932</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col2" class="data row18 col2" >54646.262</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col3" class="data row18 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col4" class="data row18 col4" >1999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col5" class="data row18 col5" >19999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col6" class="data row18 col6" >49999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row18_col7" class="data row18 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row19" class="row_heading level1 row19" >10-20 years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col0" class="data row19 col0" >419.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col1" class="data row19 col1" >62854.248</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col2" class="data row19 col2" >95953.473</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col3" class="data row19 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col4" class="data row19 col4" >6249.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col5" class="data row19 col5" >39999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col6" class="data row19 col6" >89999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row19_col7" class="data row19 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_b8f75012_5e33_11eb_b4d5_0242ac130202level1_row20" class="row_heading level1 row20" >20+ years</th>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col0" class="data row20 col0" >414.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col1" class="data row20 col1" >80057.572</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col2" class="data row20 col2" >127480.974</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col3" class="data row20 col3" >999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col4" class="data row20 col4" >5624.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col5" class="data row20 col5" >46910.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col6" class="data row20 col6" >99999.000</td>
                        <td id="T_b8f75012_5e33_11eb_b4d5_0242ac130202row20_col7" class="data row20 col7" >1000000.000</td>
            </tr>
    </tbody></table>


<details><summary>code</summary>
<p>

```python
# Q6
df = pd.crosstab([data['Q24'], data['Company_Category'], data['Q6']],
                 []).reset_index()
df = df.rename(columns={'__dummy__': 'size'})

df1 = pd.crosstab([data['Company_Category'], data['Q6']],
                 []).reset_index()
df1 = df1.rename(columns={'__dummy__': 'total_size'})

df = df.merge(df1, how='inner', on=['Company_Category', 'Q6'])
df['percentage'] = df['size']/df['total_size']

palette = sns.color_palette("Paired", len(data['Q6'].unique()))
lp = sns.relplot(
    data=df,
    x="Q24",
    y="percentage",
    hue="Q6",
    col="Company_Category",
    kind="scatter",
    height=5,
    aspect=.75,
    palette=palette,
    facet_kws=dict(sharex=False),
)

lp.set_xticklabels(fontsize=10, rotation=90, step=2)
```

</p>
</details>


![png](/assets/images/eda/output_68_1.png)


🚀Highlights:
1. In All, avg Salary increses with the year of coding experience.
2. Avg salary in Startup is less than Mid or Large Size company.

<details><summary>code</summary>
<p>

```python
df = data[['Company_Category','Q15', 'Q24_new']].groupby(['Company_Category','Q15']).describe()
df = df['Q24_new']
(df.style
.background_gradient(subset=['mean']))
```

</p>
</details>



<style  type="text/css" >
#T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col1,#T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col1{
            background-color:  #cacee5;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col1{
            background-color:  #e8e4f0;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col1{
            background-color:  #dedcec;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col1{
            background-color:  #b9c6e0;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col1{
            background-color:  #a2bcda;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col1{
            background-color:  #76aad0;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col1,#T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col1{
            background-color:  #0771b1;
            color:  #f1f1f1;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col1{
            background-color:  #023858;
            color:  #f1f1f1;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col1{
            background-color:  #eee8f3;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col1,#T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col1{
            background-color:  #f2ecf5;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col1{
            background-color:  #cdd0e5;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col1{
            background-color:  #adc1dd;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col1{
            background-color:  #3b92c1;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col1{
            background-color:  #5a9ec9;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col1,#T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col1{
            background-color:  #046096;
            color:  #f1f1f1;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col1{
            background-color:  #faf3f9;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col1,#T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col1{
            background-color:  #fff7fb;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col1{
            background-color:  #ede7f2;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col1{
            background-color:  #d1d2e6;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col1{
            background-color:  #c9cee4;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col1{
            background-color:  #88b1d4;
            color:  #000000;
        }#T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col1{
            background-color:  #056ead;
            color:  #f1f1f1;
        }</style><table id="T_ba940d84_5e33_11eb_b4d5_0242ac130202" ><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>        <th class="col_heading level0 col1" >mean</th>        <th class="col_heading level0 col2" >std</th>        <th class="col_heading level0 col3" >min</th>        <th class="col_heading level0 col4" >25%</th>        <th class="col_heading level0 col5" >50%</th>        <th class="col_heading level0 col6" >75%</th>        <th class="col_heading level0 col7" >max</th>    </tr>    <tr>        <th class="index_name level0" >Company_Category</th>        <th class="index_name level1" >Q15</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level0_row0" class="row_heading level0 row0" rowspan=9>Large Size Company</th>
                        <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row0" class="row_heading level1 row0" >I do not use machine learning methods</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col0" class="data row0 col0" >427.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col1" class="data row0 col1" >59222.862</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col2" class="data row0 col2" >99546.378</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col3" class="data row0 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col4" class="data row0 col4" >7499.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col5" class="data row0 col5" >39999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col6" class="data row0 col6" >79999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row0_col7" class="data row0 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row1" class="row_heading level1 row1" >Under 1 year</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col0" class="data row1 col0" >1004.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col1" class="data row1 col1" >43221.308</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col2" class="data row1 col2" >64925.946</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col3" class="data row1 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col4" class="data row1 col4" >6874.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col5" class="data row1 col5" >19999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col6" class="data row1 col6" >59999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row1_col7" class="data row1 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row2" class="row_heading level1 row2" >1-2 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col0" class="data row2 col0" >802.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col1" class="data row2 col1" >49030.254</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col2" class="data row2 col2" >57838.133</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col3" class="data row2 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col4" class="data row2 col4" >7499.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col5" class="data row2 col5" >29999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col6" class="data row2 col6" >69999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row2_col7" class="data row2 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row3" class="row_heading level1 row3" >2-3 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col0" class="data row3 col0" >502.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col1" class="data row3 col1" >65434.530</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col2" class="data row3 col2" >96476.630</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col3" class="data row3 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col4" class="data row3 col4" >14999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col5" class="data row3 col5" >46910.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col6" class="data row3 col6" >79999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row3_col7" class="data row3 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row4" class="row_heading level1 row4" >3-4 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col0" class="data row4 col0" >346.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col1" class="data row4 col1" >73905.309</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col2" class="data row4 col2" >112471.381</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col3" class="data row4 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col4" class="data row4 col4" >19999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col5" class="data row4 col5" >48454.500</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col6" class="data row4 col6" >89999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row4_col7" class="data row4 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row5" class="row_heading level1 row5" >4-5 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col0" class="data row5 col0" >307.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col1" class="data row5 col1" >88104.368</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col2" class="data row5 col2" >81829.149</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col3" class="data row5 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col4" class="data row5 col4" >29999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col5" class="data row5 col5" >69999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col6" class="data row5 col6" >124999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row5_col7" class="data row5 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row6" class="row_heading level1 row6" >5-10 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col0" class="data row6 col0" >376.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col1" class="data row6 col1" >120033.197</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col2" class="data row6 col2" >143621.043</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col3" class="data row6 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col4" class="data row6 col4" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col5" class="data row6 col5" >79999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col6" class="data row6 col6" >149999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row6_col7" class="data row6 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row7" class="row_heading level1 row7" >10-20 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col0" class="data row7 col0" >114.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col1" class="data row7 col1" >120106.342</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col2" class="data row7 col2" >126344.556</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col3" class="data row7 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col4" class="data row7 col4" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col5" class="data row7 col5" >89999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col6" class="data row7 col6" >149999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row7_col7" class="data row7 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row8" class="row_heading level1 row8" >20 or more years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col0" class="data row8 col0" >72.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col1" class="data row8 col1" >153047.222</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col2" class="data row8 col2" >207354.624</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col3" class="data row8 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col4" class="data row8 col4" >46910.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col5" class="data row8 col5" >89999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col6" class="data row8 col6" >199999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row8_col7" class="data row8 col7" >1000000.000</td>
            </tr>
            <tr>
                        <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level0_row9" class="row_heading level0 row9" rowspan=9>Mid Size Company</th>
                        <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row9" class="row_heading level1 row9" >I do not use machine learning methods</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col0" class="data row9 col0" >337.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col1" class="data row9 col1" >39642.252</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col2" class="data row9 col2" >66896.277</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col3" class="data row9 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col4" class="data row9 col4" >3999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col5" class="data row9 col5" >19999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col6" class="data row9 col6" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row9_col7" class="data row9 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row10" class="row_heading level1 row10" >Under 1 year</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col0" class="data row10 col0" >804.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col1" class="data row10 col1" >35483.867</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col2" class="data row10 col2" >65699.467</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col3" class="data row10 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col4" class="data row10 col4" >2999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col5" class="data row10 col5" >14999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col6" class="data row10 col6" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row10_col7" class="data row10 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row11" class="row_heading level1 row11" >1-2 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col0" class="data row11 col0" >596.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col1" class="data row11 col1" >35959.883</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col2" class="data row11 col2" >43153.827</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col3" class="data row11 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col4" class="data row11 col4" >3999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col5" class="data row11 col5" >19999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col6" class="data row11 col6" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row11_col7" class="data row11 col7" >249999.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row12" class="row_heading level1 row12" >2-3 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col0" class="data row12 col0" >373.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col1" class="data row12 col1" >57896.954</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col2" class="data row12 col2" >103845.938</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col3" class="data row12 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col4" class="data row12 col4" >7499.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col5" class="data row12 col5" >39999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col6" class="data row12 col6" >69999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row12_col7" class="data row12 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row13" class="row_heading level1 row13" >3-4 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col0" class="data row13 col0" >228.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col1" class="data row13 col1" >59134.039</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col2" class="data row13 col2" >61185.089</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col3" class="data row13 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col4" class="data row13 col4" >14999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col5" class="data row13 col5" >46910.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col6" class="data row13 col6" >79999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row13_col7" class="data row13 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row14" class="row_heading level1 row14" >4-5 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col0" class="data row14 col0" >210.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col1" class="data row14 col1" >70306.410</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col2" class="data row14 col2" >66302.248</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col3" class="data row14 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col4" class="data row14 col4" >19999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col5" class="data row14 col5" >54999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col6" class="data row14 col6" >89999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row14_col7" class="data row14 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row15" class="row_heading level1 row15" >5-10 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col0" class="data row15 col0" >206.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col1" class="data row15 col1" >103465.748</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col2" class="data row15 col2" >149086.275</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col3" class="data row15 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col4" class="data row15 col4" >26249.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col5" class="data row15 col5" >69999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col6" class="data row15 col6" >124999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row15_col7" class="data row15 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row16" class="row_heading level1 row16" >10-20 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col0" class="data row16 col0" >66.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col1" class="data row16 col1" >95655.409</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col2" class="data row16 col2" >82203.755</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col3" class="data row16 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col4" class="data row16 col4" >29999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col5" class="data row16 col5" >89999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col6" class="data row16 col6" >124999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row16_col7" class="data row16 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row17" class="row_heading level1 row17" >20 or more years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col0" class="data row17 col0" >41.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col1" class="data row17 col1" >132702.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col2" class="data row17 col2" >96074.938</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col3" class="data row17 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col4" class="data row17 col4" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col5" class="data row17 col5" >124999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col6" class="data row17 col6" >199999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row17_col7" class="data row17 col7" >500000.000</td>
            </tr>
            <tr>
                        <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level0_row18" class="row_heading level0 row18" rowspan=9>Startup</th>
                        <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row18" class="row_heading level1 row18" >I do not use machine learning methods</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col0" class="data row18 col0" >428.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col1" class="data row18 col1" >28860.953</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col2" class="data row18 col2" >48139.774</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col3" class="data row18 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col4" class="data row18 col4" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col5" class="data row18 col5" >9999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col6" class="data row18 col6" >46910.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row18_col7" class="data row18 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row19" class="row_heading level1 row19" >Under 1 year</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col0" class="data row19 col0" >1425.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col1" class="data row19 col1" >24700.697</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col2" class="data row19 col2" >77560.292</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col3" class="data row19 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col4" class="data row19 col4" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col5" class="data row19 col5" >1999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col6" class="data row19 col6" >29999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row19_col7" class="data row19 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row20" class="row_heading level1 row20" >1-2 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col0" class="data row20 col0" >933.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col1" class="data row20 col1" >25164.996</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col2" class="data row20 col2" >51420.469</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col3" class="data row20 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col4" class="data row20 col4" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col5" class="data row20 col5" >3999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col6" class="data row20 col6" >39999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row20_col7" class="data row20 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row21" class="row_heading level1 row21" >2-3 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col0" class="data row21 col0" >431.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col1" class="data row21 col1" >40433.441</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col2" class="data row21 col2" >83894.328</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col3" class="data row21 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col4" class="data row21 col4" >1999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col5" class="data row21 col5" >14999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col6" class="data row21 col6" >49999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row21_col7" class="data row21 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row22" class="row_heading level1 row22" >3-4 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col0" class="data row22 col0" >232.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col1" class="data row22 col1" >56312.991</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col2" class="data row22 col2" >105434.274</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col3" class="data row22 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col4" class="data row22 col4" >9374.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col5" class="data row22 col5" >29999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col6" class="data row22 col6" >69999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row22_col7" class="data row22 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row23" class="row_heading level1 row23" >4-5 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col0" class="data row23 col0" >192.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col1" class="data row23 col1" >59477.708</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col2" class="data row23 col2" >67260.104</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col3" class="data row23 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col4" class="data row23 col4" >7499.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col5" class="data row23 col5" >39999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col6" class="data row23 col6" >89999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row23_col7" class="data row23 col7" >500000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row24" class="row_heading level1 row24" >5-10 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col0" class="data row24 col0" >184.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col1" class="data row24 col1" >82851.424</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col2" class="data row24 col2" >105673.430</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col3" class="data row24 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col4" class="data row24 col4" >19999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col5" class="data row24 col5" >54999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col6" class="data row24 col6" >106249.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row24_col7" class="data row24 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row25" class="row_heading level1 row25" >10-20 years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col0" class="data row25 col0" >54.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col1" class="data row25 col1" >122049.667</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col2" class="data row25 col2" >158252.020</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col3" class="data row25 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col4" class="data row25 col4" >46910.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col5" class="data row25 col5" >79999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col6" class="data row25 col6" >124999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row25_col7" class="data row25 col7" >1000000.000</td>
            </tr>
            <tr>
                                <th id="T_ba940d84_5e33_11eb_b4d5_0242ac130202level1_row26" class="row_heading level1 row26" >20 or more years</th>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col0" class="data row26 col0" >56.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col1" class="data row26 col1" >132942.304</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col2" class="data row26 col2" >196411.495</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col3" class="data row26 col3" >999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col4" class="data row26 col4" >18749.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col5" class="data row26 col5" >69999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col6" class="data row26 col6" >199999.000</td>
                        <td id="T_ba940d84_5e33_11eb_b4d5_0242ac130202row26_col7" class="data row26 col7" >1000000.000</td>
            </tr>
    </tbody></table>


<details><summary>code</summary>
<p>

```python
# Q15
df = pd.crosstab([data['Q24'], data['Company_Category'], data['Q15']],
                 []).reset_index()
df = df.rename(columns={'__dummy__': 'size'})

df1 = pd.crosstab([data['Company_Category'], data['Q15']],
                 []).reset_index()
df1 = df1.rename(columns={'__dummy__': 'total_size'})

df = df.merge(df1, how='inner', on=['Company_Category', 'Q15'])
df['percentage'] = df['size']/df['total_size']

palette = sns.color_palette("Paired", len(df['Q15'].unique()))
lp = sns.relplot(
    data=df,
    x="Q24",
    y="percentage",
    hue="Q15",
    col="Company_Category",
    kind="scatter",
    height=5,
    aspect=.75,
    palette=palette,
    facet_kws=dict(sharex=False),
)

lp.set_xticklabels(fontsize=10, rotation=90, step=2)
```

</p>
</details>

![png](/assets/images/eda/output_71_1.png)


🚀Highlights:
1. In All, avg Salary **increses with the year of machine learning experience**.
2. In All, avg Salary of **machine learning experience is higher than coding experience**.

# Summary

| Aspect | Large/Mid Size Company | Startup |
| --- | --- | --- |
| Age | 5 out of 10 has age under 25-35 year and After 55 year, people don’t want to do job in Large or Mid-Size company. | 3 out of 10 has age under 18-24 years. After 55 year, people don’t want to do job for Start-ups. |
| Location | ~40% are from India & USA.| ~30% are from India & USA|
| Education | Has more Mater’s & PhD’s | Has more Bachelors.|
|Job Role |Has more Business Analysts. | Has more Machine learning engineer and Research Scientist.|
|Coding Experience|	~40% of 3-10 year of coding experience.|	~60 of 5 years of coding experience.
|Machine learning Experience	|~30% of 0-1 year of machine learning experience. |	~40% of 0-1 year of machine learning experience.|
|Programming Language & Packages|	SQL & R are more used in Large Size Company. Scikit-Learn, Xgboost, LightGBM, Caret, Catboost are more use in Large Size company.|	DL framework i.e. TensorFlow, Keras, Pytorch are more use in Startups.|
|Incorporated Machine Learning | ~25% of them have well established ML methods and models in production for more than 2 years.|~30% of them are exploring ML methods and may one day put a model into production|
|Opportunities|	~40% of them are building prototypes to explore applying machine learning to new areas.	|~30% of them are building prototypes to explore applying machine learning to new areas.
|Salary by Job Role|	Research Scientist & Data Scientist getting more salary compare to other profiles, avg. salary is $75000-80000. Order is like this: Research Scientist(77741) > Data Scientist(77737) > Machine Learning Engineer(73560) > Statistician(71181) > Data Engineer(63187) > Data Analyst (44106)	|Research Scientist & Data Scientist getting more salary compare to other profiles, avg. salary is $40000-45000. \n Order is like this: Data Scientist(41170) > Research Scientist(41550) > Data Engineer(39629) > Machine Learning Engineer(24921) > Statistician(23247) > Data Analyst(22645)
|Salary by Gender|Man is greater than average salary of woman. Difference in Man vs Woman salary is about 22%.	|Man is greater than average salary of woman. Difference in Man vs Woman salary is about 35%.|
|Salary by Education|Avg. Salary of Doctoral degree is greater, whereas large difference in avg. salary of master and bachelors.|Avg. Salary of Doctoral degree is greater, Whereas very small difference in avg. salary of master and bachelors.|
|Salary by ML Experience |Avg. Salary increases with ml experience.|	Avg. Salary increases with ml experience.|

# References

1. moDel Agnostic Language for Exploration and eXplanation: https://github.com/ModelOriented/DALEX
2. Line plots on multiple facets: https://seaborn.pydata.org/examples/faceted_lineplot.html
3. Color: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
4. Annotations: https://matplotlib.org/3.3.3/tutorials/text/annotations.html
5. Combining two subplots using subplots and GridSpec: https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_and_subplots.htm