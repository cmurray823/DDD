#!/usr/bin/env python
# coding: utf-8

# In[537]:


import pandas as pd     
import os 
data = pd.read_csv('Unins_data.csv') 
import numpy as np
import patsy                           
import statsmodels.api as sm           
import statsmodels.formula.api as smf 


# In[539]:


random_subset=data.sample(n=100000)


# In[540]:


random_subset=random_subset.set_index('age')
random_subset.drop(['Less than 1 year old', '90 (90+ in 1980 and 1990)'], inplace=True)
random_subset=random_subset.reset_index()
random_subset['age']=random_subset['age'].astype(int)
random_subset=random_subset[random_subset['age']<65]


# In[541]:


data=random_subset.drop(['hiufpgbase', 'hiufpginc'], axis=1)
data=data.dropna()


# In[542]:


dummy=pd.get_dummies(data['hinscaid'])
data=pd.concat([data, dummy], axis=1)
data=data.drop(['No insurance through Medicaid'], axis=1)
data.rename(columns={'Has insurance through Medicaid': 'Have_caid'}, inplace=True)

dummy=pd.get_dummies(data['hinsemp'])
data=pd.concat([data, dummy], axis=1)
data=data.drop(['No insurance through employer/union'], axis=1)
data.rename(columns={'Has insurance through employer/union': 'Have_emp'}, inplace=True)

dummy=pd.get_dummies(data['hcovany'])
data=pd.concat([data, dummy], axis=1)
data=data.drop(['No health insurance coverage'], axis=1)
data.rename(columns={'With health insurance coverage': 'Have_any'}, inplace=True)

pur=pd.get_dummies(data['hinspur'])
data=pd.concat([data, pur], axis=1)
data=data.drop(['No insurance purchased directly'], axis=1)
data.rename(columns={'Has insurance purchased directly': 'Have_pur'}, inplace=True)

pur=pd.get_dummies(data['hcovpriv'])
data=pd.concat([data, pur], axis=1)
data=data.drop(['Without private health insurance coverage'], axis=1)
data.rename(columns={'With private health insurance coverage': 'Have_priv'}, inplace=True)

pur=pd.get_dummies(data['hinsva'])
data=pd.concat([data, pur], axis=1)
data=data.drop(['No insurance through VA'], axis=1)
data.rename(columns={'Has insurance through VA': 'Have_va'}, inplace=True)

pur=pd.get_dummies(data['hinstri'])
data=pd.concat([data, pur], axis=1)
data=data.drop(['No insurance through TRICARE'], axis=1)
data.rename(columns={'Has insurance through TRICARE': 'Have_tri'}, inplace=True)

pur=pd.get_dummies(data['hinsihs'])
data=pd.concat([data, pur], axis=1)
data=data.drop(['No insurance through Indian Health Service'], axis=1)
data.rename(columns={'Has insurance through Indian Health Service': 'Have_ihs'}, inplace=True)


# In[543]:


data['pov_bins']=pd.cut(x=data['poverty'], bins=[0, 138, 400, 501])
dummy=pd.get_dummies(data['pov_bins'])
data=pd.concat([data, dummy], axis=1)


# In[544]:


data.head()


# In[545]:


data.columns=['age','year' ,'sample', 'serial', 'cbserial', 'hhwt', 'statefip', 'countyfip', 'puma', 'gq', 'pernum', 'perwt', 'famsize', 'sex', 'marst', 'race','raced', 'hispan', 'hispand', 'hcovany', 'hcovpriv', 'hinsemp', 'hinspur', 'hinstri', 'hcovpub', 'hinscaid', 'hinsva', 'hinsihs', 'empstat', 'empstatd', 'occ', 'occ1990', 'occ2010', 'inctot', 'ftotinc', 'incwage', 'incwelfr', 'poverty', 'hwsei', 'Have_caid', 'Have_emp', 'Have_any', 'Have_pur', 'Have_priv', 'Have_va', 'Have_tri', 'Have_ihs', 'pov_bins', 'low', 'mid', 'high']



# In[546]:


data['other']=data[['Have_va','Have_tri','Have_ihs']].sum(axis=1)


# In[547]:


expand=[]

for state in data['statefip']:
        if state=='Alaska':
            expand.append(0)
        elif state=='Florida':
            expand.append(0)
        elif state=='Georgia':
            expand.append(0)
        elif state=='Idaho':
            expand.append(0)
        elif state=='Indiana':
            expand.append(0)
        elif state=='Kansas':
            expand.append(0)
        elif state=='Louisiana':
            expand.append(0)
        elif state=='Maine':
            expand.append(0)
        elif state=='Michigan':
            expand.append(0)
        elif state=='Mississippi':
            expand.append(0)
        elif state=='Alabama':
            expand.append(0)
        elif state=='Missouri':
            expand.append(0)
        elif state=='Montana':
            expand.append(0)
        elif state=='Nebraska':
            expand.append(0)
        elif state=='New Hampshire':
            expand.append(0)
        elif state=='North Carolina':
            expand.append(0)
        elif state=='Oklahoma':
            expand.append(0)
        elif state=='Pennsylvania':
            expand.append(0)
        elif state=='South Carolina':
            expand.append(0)
        elif state=='South Dakota':
            expand.append(0)
        elif state=='Tennessee':
            expand.append(0)
        elif state=='Utah':
            expand.append(0)
        elif state=='Virginia':
            expand.append(0)
        elif state=='Wisconsin':
            expand.append(0)
        elif state=='Wyoming':
            expand.append(0)
        else:
            expand.append(1)
data['expand']=expand


# In[548]:


yr=[]
for year in data['year']:
        if year==2011:
            yr.append(0)
        elif year==2012:
            yr.append(0)
        elif year==2013:
            yr.append(0)
        else:
            yr.append(1)
data['y14']=yr


# In[549]:


dummy=pd.get_dummies(data['hcovany'])
data=pd.concat([data, dummy], axis=1)
data=data.drop(['With health insurance coverage'], axis=1)
data.rename(columns={'No health insurance coverage': 'Unins'}, inplace=True)


# In[550]:


data


# In[551]:


pre=data[data['year']==2013]


# In[552]:


pre['statun'] = pre.groupby(['statefip'])['Unins'].transform('mean') 


# In[553]:


states=pre.copy()


# In[554]:


states.sort_values('statefip', inplace = True) 
states.drop_duplicates(subset ="statefip", 
                     keep = 'first', inplace = True)


# In[555]:


states["statefip"] = states["statefip"].astype('category')
states["statefip_cat"] = states["statefip"].cat.codes


# In[556]:


data["statefip"] = data["statefip"].astype('category')
data["statefip_cat"] = data["statefip"].cat.codes


# In[557]:


states.index = range(len(states)) 


# In[558]:


states=states.set_index('statefip_cat')


# In[559]:


data=data.reset_index()


# In[560]:


state=[]
for sta in data['statefip_cat']:
    val=states['statun'].at[sta]
    state.append(val)


# In[561]:


data['statun']=state


# In[562]:


data


# In[620]:


mean_s=data['statun'].mean()


# In[621]:


res_any = smf.ols('Have_any ~ y14*statun + y14*expand + y14*statun*expand', data).fit()
res_priv = smf.ols('Have_priv ~ y14*statun + y14*expand + y14*statun*expand', data).fit()
res_emp = smf.ols('Have_emp ~ y14*statun + y14*expand + y14*statun*expand', data).fit()
res_pur = smf.ols('Have_pur ~ y14*statun + y14*expand + y14*statun*expand', data).fit()
res_caid = smf.ols('Have_caid ~ y14*statun + y14*expand + y14*statun*expand', data).fit()
res_oth = smf.ols('other ~ y14*statun + y14*expand + y14*statun*expand', data).fit()
#print(res.summary())

coef = res_any.params.to_frame()
coef.rename(columns={0: "Any Insurance" }, inplace = True)
coef['Any Private']=res_priv.params.to_frame()
coef['Employer Sponsored']=res_emp.params.to_frame()
coef['Individually Purchased']=res_pur.params.to_frame()
coef['Medicaid']=res_caid.params.to_frame()
coef['Other']=res_oth.params.to_frame()
coef.drop(['Intercept'])


# In[622]:


coef


# In[623]:


data["puma"] = data["puma"].astype('category')
data['puma_cat'] = data["puma"].cat.codes


# In[624]:


cop=data.copy()


# In[625]:


#cop.to_excel('cop.xlsx')


# In[626]:


cop.sort_values(['puma_cat','statefip_cat'],axis=0,ascending=True,inplace=True) 
cop.index = range(len(cop))
cop['pumun']=0
state=1


# In[627]:


for i in range(len(cop)):
    if (cop['statefip_cat'].at[i]==0) & (cop['statefip_cat'].at[i]==0):
        cop['pumun'].at[i] = state
    else:
        if (cop['statefip_cat'].at[i]==cop['statefip_cat'].at[i-1])&(cop['puma_cat'].at[i]==cop['puma_cat'].at[i-1]):
            cop['pumun'].at[i] = state
        else:
            state = state + 1
            cop['pumun'].at[i] = state


# In[628]:


ys=cop[cop['year']==2013]


# In[629]:


ys


# In[630]:


ys['up'] = ys.groupby('pumun')['Unins'].transform('mean')


# In[631]:


ys.sort_values('pumun', inplace = True) 
ys.drop_duplicates(subset ="pumun", 
                     keep = 'first', inplace = True)


# In[632]:


ys


# In[633]:


pumadf=cop[cop['year']>2011]


# In[634]:


ys=ys.set_index('pumun')


# In[635]:


#copy.index=copy.index.astype(int)
puma=[]
for pu in pumadf['pumun']:
    if pu not in ys.index:
        puma.append(0)
    else:
        val=ys['up'].at[pu]
        puma.append(val)

pumadf['unsp_2']=puma


# In[636]:


pumadf


# In[637]:


mean_p=pumadf['unsp_2'].mean()


# In[638]:


res_any = smf.ols('Have_any ~ y14*unsp_2 + y14*expand + y14*unsp_2*expand', pumadf).fit()
res_priv = smf.ols('Have_priv ~ y14*unsp_2 + y14*expand + y14*unsp_2*expand', pumadf).fit()
res_emp = smf.ols('Have_emp ~ y14*unsp_2 + y14*expand + y14*unsp_2*expand', pumadf).fit()
res_pur = smf.ols('Have_pur ~ y14*unsp_2 + y14*expand + y14*unsp_2*expand', pumadf).fit()
res_caid = smf.ols('Have_caid ~ y14*unsp_2 + y14*expand + y14*unsp_2*expand', pumadf).fit()
res_oth = smf.ols('other ~ y14*unsp_2 + y14*expand + y14*unsp_2*expand', pumadf).fit()
#print(res.summary())

coef = res_any.params.to_frame()
coef.rename(columns={0: "Any Insurance" }, inplace = True)
coef['Any Private']=res_priv.params.to_frame()
coef['Employer Sponsored']=res_emp.params.to_frame()
coef['Individually Purchased']=res_pur.params.to_frame()
coef['Medicaid']=res_caid.params.to_frame()
coef['Other']=res_oth.params.to_frame()
coef=coef.drop(['Intercept', 'y14', 'unsp_2', 'expand', 'unsp_2:expand'])


# In[639]:


coef=coef.transpose()
coef['ACA without Medicaid Expansion']=coef['y14:unsp_2']*mean_p
coef['Medicaid Expansion']=coef['y14:unsp_2:expand']*mean_p


# In[640]:


#2010-2012, ACS sample, respondents from 2010 and 2011 correspond to census 2000 based pumas, 2012 correspond to 
#2010 based puma, so drop 2010 and 2011


# In[641]:


median=pumadf['unsp_2'].median()


# In[642]:


median


# In[643]:


med=[]
for i in pumadf.index:
    val=median
    med.append(val)

pumadf['median']=med


# In[644]:


pumadf=pumadf.reset_index()


# In[645]:


pumadf['above_med']=0
for i in range(len(pumadf)):
    if (pumadf['unsp_2'].at[i]>pumadf['median'].at[i]):
        pumadf['above_med'].at[i] = 1
    else:
        pumadf['above_med'].at[i] = 0


# In[646]:


pumadf


# In[647]:


abovemed=pumadf[pumadf['above_med']==1]
belowmed=pumadf[pumadf['above_med']==0]


# In[648]:


res = smf.ols('Have_caid ~ y14 + y14*expand', abovemed).fit()
print(res.summary())


# In[649]:


res = smf.ols('Have_caid ~ y14 + y14*expand', belowmed).fit()
print(res.summary())


# In[650]:


abovemed['dddmean'] = abovemed.groupby('expand')['Have_caid'].transform('mean')


# In[651]:


belowmed['dddmean'] = belowmed.groupby('expand')['Have_caid'].transform('mean')


# In[652]:


abovemed


# In[653]:


belowmed.sort_values('dddmean', inplace = True) 
mean_b=belowmed['dddmean'].unique()
mean_b=mean_b.tolist()


# In[654]:


abovemed.sort_values('dddmean', inplace = True) 
mean_a=abovemed['dddmean'].unique()
mean_a=mean_a.tolist()


# In[655]:


import matplotlib.pyplot as plt
import numpy as np


# In[656]:


objects = ('No_Expand', 'Expand')
y_pos = np.arange(len(objects))
rate_a = mean_a
rate_b = mean_b

width=.5
fig, ax = plt.subplots(1,2, figsize=(10,5))
plt.subplots_adjust(wspace=0.2)
ax[0].bar(y_pos - width/14, rate_a, width, color=['red', 'blue'])
ax[0].set_ylabel('Medicaid Rate', fontsize=15)
ax[0].set_title('Above Median Uninsured Rate', fontsize=15)
ax[0].set_xticks(y_pos)
ax[0].set_xticklabels(objects)
ax[0].set_ylim(0, 0.18)

width=0.5
ax[1].bar(y_pos - width/14, rate_b, width, color=['red', 'blue'])
ax[1].set_title('Below Median Uninsured Rate', fontsize=15)
ax[1].set_xticks(y_pos)
ax[1].set_xticklabels(objects)
ax[1].set_ylim(0, 0.18)
ax[1].set_yticklabels([])



# In[663]:


columns=['Have_caid', 'Have_any', 'Have_priv', 'Have_emp', 'Have_pur', 'other']

for i in columns:
    abovemed['dddmean'] = abovemed.groupby('expand')[i].transform('mean')
    belowmed['dddmean'] = belowmed.groupby('expand')[i].transform('mean')

    belowmed.sort_values('dddmean', inplace = True) 
    mean_b=belowmed['dddmean'].unique()
    mean_b=mean_b.tolist()

    abovemed.sort_values('dddmean', inplace = True) 
    mean_a=abovemed['dddmean'].unique()
    mean_a=mean_a.tolist()

    objects = ('No_Expand', 'Expand')
    y_pos = np.arange(len(objects))
    rate_a = mean_a
    rate_b = mean_b

    width=.5
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(wspace=0.2)
    ax[0].bar(y_pos - width/14, rate_a, width, color=['red', 'blue'])
    ax[0].set_ylabel(i, fontsize=15)
    ax[0].set_title('Above Median Uninsured Rate', fontsize=15)
    ax[0].set_xticks(y_pos)
    ax[0].set_xticklabels(objects)
    ax[0].set_ylim(0, mean_a[0]+.1)

    width=0.5
    ax[1].bar(y_pos - width/14, rate_b, width, color=['red', 'blue'])
    ax[1].set_title('Below Median Uninsured Rate', fontsize=15)
    ax[1].set_xticks(y_pos)
    ax[1].set_xticklabels(objects)
    ax[1].set_ylim(0, mean_b[0]+.1)
    ax[1].set_yticklabels([])


# In[472]:


data["countyfip"] = data["countyfip"].astype('category')
data['county_cat'] = data["countyfip"].cat.codes


# In[473]:


cop=data.copy()
cop.sort_values(['county_cat','statefip_cat'],axis=0,ascending=True,inplace=True) 
cop.index = range(len(cop))
cop['countun']=0
state=1


# In[474]:


for i in range(len(cop)):
    if (cop['statefip_cat'].at[i]==0) & (cop['statefip_cat'].at[i]==0):
        cop['countun'].at[i] = state
    else:
        if (cop['statefip_cat'].at[i]==cop['statefip_cat'].at[i-1])&(cop['county_cat'].at[i]==cop['county_cat'].at[i-1]):
            cop['countun'].at[i] = state
        else:
            state = state + 1
            cop['countun'].at[i] = state


# In[475]:


ys=cop[cop['year']==2013]


# In[476]:


ys['uc'] = ys.groupby('countun')['Unins'].transform('mean')


# In[477]:


ys.sort_values('countun', inplace = True) 
ys.drop_duplicates(subset ="countun", 
                     keep = 'first', inplace = True)
ys=ys.set_index('countun')


# In[478]:


ys


# In[479]:


county=[]
for count in cop['countun']:
    if count not in ys.index:
        county.append(0)
    else:
        val=ys['uc'].at[count]
        county.append(val)

cop['unsc']=county


# In[480]:


cop['unsc'].mean()


# In[481]:


cop


# In[482]:


res_any = smf.ols('Have_any ~ y14*unsc + y14*expand + unsc*y14*expand', cop).fit()
res_priv = smf.ols('Have_priv ~ y14*unsc + y14*expand + unsc*y14*expand', cop).fit()
res_emp = smf.ols('Have_emp ~ y14*unsc + y14*expand + unsc*y14*expand', cop).fit()
res_pur = smf.ols('Have_pur ~ y14*unsc + y14*expand + unsc*y14*expand', cop).fit()
res_caid = smf.ols('Have_caid ~ y14*unsc + y14*expand + unsc*y14*expand', cop).fit()
res_oth = smf.ols('other ~ y14*unsc + y14*expand + unsc*y14*expand', cop).fit()
#print(res.summary())

coef = res_any.params.to_frame()
coef.rename(columns={coe.columns[0]: "Any Insurance" }, inplace = True)
coef['Any Private']=res_priv.params.to_frame()
coef['Employer Sponsored']=res_emp.params.to_frame()
coef['Individually Purchased']=res_pur.params.to_frame()
coef['Medicaid']=res_caid.params.to_frame()
coef['Other']=res_oth.params.to_frame()

