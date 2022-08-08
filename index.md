# Predicting Mortgage Default and Prepayment via Cox Proportional Hazards Models


I worked on 2 similar projects that predict mortgage default and prepayment.


## Project 1: Exploratory Data Analysis and Default/Prepayment Models for Unison's REA (Python)

Tiancheng Xu

May 2022

### Summary

Working with historical Freddie Mac single-family loan level data, I tried to understand the default/prepayment behaviors, their distribution across different orgination years or different ages, and their relationship with other factors, using matplotlib in Python. I looked into the implications of these behaviors on the loan-lending or house-coinvesting business. I also build a very preliminary CoxPH model, fitted it with data and applied k-fold cross-validation with concordance-index. I made an in-depth analysis on how Unison should leverage these information to raise profit and to avoid risks in its line of business - home co-investing.

Some of the Python packages used: numpy, pandas, matplotlib, lifeline, pymysql.

Future improvement needed: cross-compare more models (with different factors), and use different Ks for k-fold cross-validation to fine-tune the model; expand the analysis to other housing types, including apartments and condos, to obtain a more exhaustive overview of the housing market.

---

### Primer
Unison's REA (Rental Equity Agreement) issues a homeowner a certain amount of cash (a percentage of house value) in year 0, but instead of asking for monthly repayment, it asks for the pricipal and a certain percentage in the "value change" of the house in year 30 (or at the time of termination of the agreement, whichever comes earlier).


```
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
```

```
# Connect to the MySQL database and query the data
conn = pymysql.connect(
    host='xxxxx',
    port=int(3306),
    user='xxxxx',
    passwd='xxxxx',
    db='xxxxx',
    charset='utf8mb4')

df = pd.read_sql_query("xxxxx", conn)
```

```
# Create column for the year-part of the first_pmt_date
df['year'] = df['first_pmt_date'].str[0:4]
df.head()
```

```
# Count number of loans, grouped by year and status
counts = df.groupby(['year','status']).count()['loan_seq_num']
counts = counts.unstack(level=1)
counts = counts.fillna(0)

# Calculate the percentages of each count
perc = counts.copy()
perc['sum'] = perc.sum(axis=1)
totals = perc['sum']
perc['Alive'] = round(perc['Alive']/perc['sum'],4)
perc['Default'] = round(perc['Default']/perc['sum'],4)
perc['Prepay'] = round(perc['Prepay']/perc['sum'],4)
perc = pd.concat([perc['Alive'], perc['Default'], perc['Prepay']])

years = counts.index
```

```
# Plot Count/Percentage of Each Mortgage Status by Origination Cohort
fig, ax = plt.subplots(figsize=(25, 30))
colors = ['paleturquoise', 'plum', 'lightsalmon']
ax.bar(years, counts['Alive'], bottom=None, color=colors[0], label='Alive')
ax.bar(years, counts['Default'], bottom=counts['Alive'], color=colors[1], label='Default')
ax.bar(years, counts['Prepay'], bottom=counts['Default']+counts['Alive'], color=colors[2], label='Prepay')

ax.set_facecolor('#f0f4fc')

# Print total count for each stacked bar
for i, total in enumerate(totals):
  ax.text(totals.index[i], total + 5, round(total), ha='center', weight='bold')

# Print count and percentage for each sub-bar
for i, bar in enumerate(ax.patches):
    if bar.get_height() == 0:
        next
    else:
          ax.text(
              bar.get_x() + bar.get_width() / 2,
              bar.get_height()/2 + bar.get_y() - 10,
              str(round(bar.get_height())) + '\n('+ str(round(perc[i]*100,2)) + '%)',
              ha='center',
              color='black',
              size=10
  )

ax.legend(fontsize=20)
ax.set_title('Count/Percentage of Each Mortgage Status by Origination Cohort', color='#2c3cdc',fontsize=30, weight='bold')
plt.xlabel('Origination Cohort', fontsize=20)
plt.xticks(fontsize=10)
plt.ylabel('Count', fontsize=20)
plt.yticks(fontsize=10)

plt.savefig('Status ct by Orig Year.png')
```

<img width="660" alt="image" src="https://user-images.githubusercontent.com/63265930/172239302-f3b3d011-7e10-45f7-909f-4b39f30c569a.png">

There’s an obvious trend of decrease in the percentage of prepayment for loans that originated from 1999 to 2017.

Default percentage sees a major increase for loans originated in the mid-late 2000s (before the 2008 financial crisis), but improves critically for those originated in the 2010s.

We still need more information to determine if the above-mentioned trends were caused by the market (then vs. now), or merely by the ages of the loans (e.g. newer loans tend to have lower probability of prepayment and default).


```
# Count number of loans, grouped by age and status
counts2 = df.groupby(['age','status']).count()['loan_seq_num']
counts2 = counts2.unstack(level=1)
counts2 = counts2.fillna(0)
counts2
       
ages = counts2.index
```

```
# Plot Count/Percentage of Each Mortgage Status by Age
fig, ax = plt.subplots(figsize=(50, 40))
colors = ['paleturquoise', 'plum', 'lightsalmon']
ax.bar(ages, counts2['Alive'], bottom=None, color=colors[0], label='Alive')
ax.bar(ages, counts2['Default'], bottom=counts2['Alive'], color=colors[1], label='Default')
ax.bar(ages, counts2['Prepay'], bottom=counts2['Default']+counts2['Alive'], color=colors[2], label='Prepay')

ax.set_facecolor('#f0f4fc')

ax.legend(fontsize=40)
ax.set_title('Count/Percentage of Each Mortgage Status by Age', color='#2c3cdc', fontsize=50, weight='bold')
plt.xlabel('Age', fontsize=30)
plt.xticks(fontsize=20)
plt.ylabel('Count', fontsize=30)
plt.yticks(fontsize=20)
# It is a very crowded graph so the count values are not shown for better eligibility

plt.savefig('Status ct by Age.png')
```

<img width="662" alt="image" src="https://user-images.githubusercontent.com/63265930/172239420-9948a44e-ae00-44d0-9d06-65fe4be7d620.png">

From the given data alone, older loans (over 140-month of age) tend to have lower percentage of prepayment. The percentage of default stays relatively stable across different ages.

Again, the analysis could be biased because the sample size for older loans is relatively small compared to younger ones.

If I were to build prepayment/default models, I would use Cox Proportional Hazard models and train the models on factors including age of loans, credit scores, area codes, orig_dti, etc.

Empirical data needed for a turnover model: relationship between house price increase and willingness to sell, typical length of ownership for houses, etc. For instance, we could use A/B testing to discover the causality between price increase and the decision to sell.


```
# Quick plot on default & prepayment percentages for analysis
counts2['sum'] = counts2.sum(axis=1)
counts2['def_perc'] = round(counts2['Default']/counts2['sum'],4)
counts2['prp_perc'] = round(counts2['Prepay']/counts2['sum'],4)

fig, ax = plt.subplots()
ax.plot(counts2['def_perc'], color='r', label='Default Perc')
ax.plot(counts2['prp_perc'], color='b', label='Prepay Perc')
ax.legend()
```

<img width="377" alt="image" src="https://user-images.githubusercontent.com/63265930/172239484-a756a534-34c5-4b68-93a7-faa78509b70b.png">

```
# Unconditional probabilities

# Exclude loans that are still alive but are younger than 60 months
# Because if they will default/prepay in first 5 years remains unknown
df5 = df[(df['age']>60) | (df['status']!='Alive')]

# That a mortgage has defaulted in the first 5 years
year5_def = len(df5[(df5['status']=='Default') & (df5['age']<=60)]) / len(df5)
round(year5_def,4)

# That a mortgage has prepaid in the first 5 years
year5_prp = len(df5[(df5['status']=='Prepay') & (df5['age']<=60)]) / len(df5)
round(year5_prp,4)

# That a mortgage remains alive after the first 5 years
year5_alv = len(df5[df5['age']>60]) / len(df5)
round(year5_alv,4)

# Sum check
year5_def + year5_prp + year5_alv
```

Different Mortgage Behaviors and Impacts on REA

Prepayments potentially mean that homeowners are likely to plan on, if not already, sell the houses earlier, thus terminating the REA earlier. For Unison, it could potentially mean that we are losing out on future increases in house prices (or avoiding future price drops). Prepayments can also be signals that homeowners have a lot of faith in the housing market compared to traditional investments and decide to put their money into the equity of the houses before the mortgage term.

Defaults, on the other hand, mean that the homeowners are having trouble with their cash flows, and it also poses risks to the REA as the lender may also not be able to repay the principals on REA at the end of the co-investment term.

Increases interest rates makes home purchases more expensive, and reduces the demand in the market, which decreases the turnover rate on REA. A bad housing market has the same effect on REA. But when the market is good and home prices are appreciating, people tend to sell their houses more often, thus increasing the turnover on REA.



```
# Build a simple CoxPH default model
# Data preparation
df['default'] = np.where(df['status']=='Default', 1, 0)
df['flag'] = np.where(df['first_time_ho_flag']=='Y', 1, 0)

# Select factors
data = df[['default', 'year', 'age', 'flag', 'msa_code', 'credit_score', 'num_units', 'orig_dti', 
           'orig_cltv', 'orig_upb', 'orig_ir', 'num_borrowers']]
data = data.dropna()
data = data[(data['orig_dti']!='') & (data['orig_dti']!='   ')]

# Build model, fit the model and k-fold-cross-validate it with concordance-index (k=5 by default)
cph = CoxPHFitter()
np.mean(k_fold_cross_validation(cph, data, duration_col='age', event_col='default', scoring_method="concordance_index"))
```

### Default Model
I built a CoxPH model to predict the default behaviors of the loans.
Factors picked: 'year', 'age', 'flag', 'msa_code', 'credit_score', 'num_units', 'orig_dti', 'orig_cltv', 'orig_upb', 'orig_ir', ' num_borrowers'.
Used the built-in k_fold_cross_validation function within the Lifelines package to cross validate the model, and obtained a concordance-index of about 0.84, which signals that the model is relatively accurate (1 being a perfect c-index and 0.5 being a random guess).
Could try other combinations of factors and compare different models by their AICs to find the best one; could also increase the k for the k-fold cross validation so that the model would be trained on a larger dataset.

Default will hurt the market price of the mortgage in general. When a homeowner defaults on mortgage and when the lender decides to move on to foreclosure, the property can potentially be sold below market price. To compensate for the extra risks that we take on, we can ask for a lowered initial co-investment (from Unison) and  an increased investor percentage. Right now, for a typical Unison customer, the investor percentage is 4x the percentage of the co-investment. We can potentially raise the multiplier to 5x or 6x.

### Final Thoughts
Fluctuation in the mortgage market and the overall economy decides the demand on the housing market, which greatly affect Unison’s business model, revenue and risks. Customers’ mortgage behaviors, on the other hand, serve as road-signs for the end-of-term price behaviors of the houses, which eventually translate to Unison’s profit. 

Loan-specific metrics like credit scores, loan-to-value, debt-to-income, as well as market metrics like House Price Index and 30-year mortgage rates are all good indicators for investment risks. We can focus on providing co-investment opportunities to low-risk individuals/families and adjust the number of ongoing projects according to market and general economic health.

Two ways to enhance investment returns: 1. to implement a stronger screening process according to borrowers’ personal background – if they have any default history, if they have a high credit score, are they married and have kids, etc., 2. to identify geographic areas where housing prices have the biggest upward potentials and the market is robust and liquid. 


---
---

## Project 2: Predicting Mortgage Default and Prepayment via Cox Proportional Hazards Models (R)

Tiancheng Xu, Ford Danielsen, Rishabh Kumar, Sundar Ryali, Van Xu, Evelyn Zhang

Apr 2022


### Summary
We performed ETL on 3M+ data from mortgage loans, mortgage rates, national unemployment rates, House Price Indices and more; using survival package in R, we trained multiple coxph models and determined the best choice of variables through cross-validationa and comparison of AICs; we predicted default/prepayment; we visualized cumulative probabilities using ggplot2 to report the default/prepayment likelihood of a given loan.

---

### Default Model

<img width="450" alt="image" src="https://user-images.githubusercontent.com/63265930/172236492-74d2f774-0a36-4912-be26-4c0fc8e585e5.png">


Schoenfeld Residuals of some selected variables

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172236646-0ebfe143-5f80-46a9-bbe8-751db34ddc55.png">

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172236685-0e41a8bb-b294-4942-ba87-83c51b5737fd.png">



### Prepay Model

<img width="450" alt="image" src="https://user-images.githubusercontent.com/63265930/172236994-7c98caec-790e-4e0f-a804-b89e4400978a.png">


Schoenfeld Residuals of some selected variables

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172237065-e1e00b87-d917-4e15-99eb-d0dd0403c96f.png">

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172237121-85f4e60b-dcb2-4a73-9efa-55ba2be9f900.png">



### Prediction

After building the models, we also tried to predict the default / prepayment status on an imaginary loan. See the R code below for details.

Cumulative Default Probability of the model

<img width="461" alt="image" src="https://user-images.githubusercontent.com/63265930/172236810-be29d09f-0207-4558-a8f1-60f9e51e8087.png">

Cumulative Prepayment Probability of the model

<img width="449" alt="image" src="https://user-images.githubusercontent.com/63265930/172237252-2de9dcff-cddb-49b8-847d-faa0eb7df70d.png">




R Code:

```
library(ggplot2)
library(survival)
library(data.table)

# Data Cleaning
load('TeamAssignment3_cdata_Q3.rda')

Data_C[, MSA := as.numeric(MSA)]
Data_C <- Data_C[MSA %in% ue_msa$MSA,]
Data_C <- Data_C[MSA %in% hpi_msa$MSA,]
Data_C <- Data_C[!is.na(Data_C$CSCORE_B),]
Data_C <- Data_C[!is.na(Data_C$ORIG_VAL),]
Data_C <- Data_C[!is.na(Data_C$ORIG_AMT),]
Data_C <- Data_C[!is.na(Data_C$ORIG_RT),]

Data_C[, ORIG_DTE := as.numeric(format(Data_C$ORIG_DTE, "%Y%m"))]
Data_C <- merge(Data_C, rates, by.x = 'ORIG_DTE', by.y = 'yearmon', all.x = TRUE)
Data_C[, spread := ORIG_RT - rate]

Data_C <- merge(Data_C, hpi_msa, by.x = c('ORIG_DTE', 'MSA'), by.y = c('yearmon', 'MSA'), all.x = TRUE)
Data_C[, hpi0 := hpi]

Data_C <- Data_C[, c("LOAN_ID","OLTV","CSCORE_B","spread","ORIG_VAL","hpi0","MSA","ORIG_RT","NUM_BO","PURPOSE","PROP_TYP","OCC_STAT","DTI","FTHB_FLG")]

load('TeamAssignment3_pdata_Q3.rda')

setnames(Data_P, "Monthly.Rpt.Prd", "yearmon")
Data_P[, yearmon := as.numeric(format(Data_P$yearmon, "%Y%m"))]
Data_P <- Data_P[LOAN_ID %in% Data_C$LOAN_ID,]

data3 <- merge(Data_P, Data_C, by='LOAN_ID')
setorderv(data3, c('LOAN_ID', 'yearmon'))
data3$status <- ifelse(data3$Zero.Bal.Code %in% c("02","03","09","15"),"default",
                       ifelse(data3$Zero.Bal.Code %in% c("01"),"prepaid","censored"))

data3 <- merge(data3, rates, by='yearmon')
data3[, cvr := ORIG_RT / rate]

data3 <- merge(data3, ue_msa, by = c('yearmon', 'MSA'))

data3 <- merge(data3, hpi_msa, by=c('yearmon', 'MSA'))

data3[, val := ORIG_VAL * hpi / hpi0]
data3[, pneq := pnorm(log(LAST_UPB/val)/(100*spi))]

data3[, start := Loan.Age]
data3[, end   := start+1]


# Use coxph to estimate a default model

# start with CSCORE_B and pneq, add other covariates one by one
cox.v1 <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq,
                data=data3, ties="efron")

cox.OLTV <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                    OLTV, data=data3, ties="efron")

cox.spread <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                      OLTV + spread, data=data3, ties="efron")

cox.ue <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                  OLTV + spread + ue, data=data3, ties="efron")

cox.cvr <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                   OLTV + spread + ue + cvr, data=data3, ties="efron")

cox.NUM_BO <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                      OLTV + spread + ue + cvr + NUM_BO, data=data3, ties="efron")

cox.PURPOSE <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                       OLTV + spread + ue + cvr + NUM_BO + PURPOSE, data=data3, ties="efron")

cox.PROP_TYP <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                        OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                        PROP_TYP, data=data3, ties="efron")

cox.OCC_STAT <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                        OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                        PROP_TYP + OCC_STAT, data=data3, ties="efron")

cox.DTI <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq + 
                   OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                   PROP_TYP + OCC_STAT + DTI, data=data3, ties="efron")

cox.FTHB_FLG <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq +
                        OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                        PROP_TYP + OCC_STAT + DTI + FTHB_FLG, data=data3, ties="efron")

# compare AIC of models
AIC.default <- rbind(AIC(cox.v1), AIC(cox.OLTV), AIC(cox.spread), AIC(cox.ue), 
                     AIC(cox.cvr), AIC(cox.NUM_BO), AIC(cox.PURPOSE), AIC(cox.PROP_TYP),
                     AIC(cox.OCC_STAT), AIC(cox.DTI), AIC(cox.FTHB_FLG))
rownames(AIC.default) <- c('v1', 'OLTV', 'spread', 'ue', 'cvr', 'NUM_BO', 'PURPOSE', 'PROP_TYP', 'OCC_STAT', 'DTI', 'FTHB_FLG')
colnames(AIC.default) <- 'AIC'
AIC.default


# final default model
cox.default <- coxph(formula = Surv(start,end,status=="default") ~ CSCORE_B + pneq +
                       OLTV + spread + PURPOSE + PROP_TYP + OCC_STAT + DTI, data=data3, ties="efron")
cox.default
AIC(cox.default)

# bh
bh1 <- basehaz(cox.default,centered=FALSE)
bh1$surv <- exp(-bh1$hazard)
ggplot(bh1,aes(x=time,y=surv)) +
  geom_line() +
  xlab("Time in months") + 
  ylab("Survival function") +
  ylim(c(0,1)) +
  ggtitle("Coxph Baseline Survival Function: Model 1")

zph1 <- cox.zph(cox.default)
plot(zph1)


# Use coxph to estimate a prepayment model

# start with CSCORE_B and pneq, add other covariates one by one
cox.v2 <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq,
                data=data3, ties="efron")

cox.OLTV.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                    OLTV, data=data3, ties="efron")

cox.spread.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                      OLTV + spread, data=data3, ties="efron")

cox.ue.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                  OLTV + spread + ue, data=data3, ties="efron")

cox.cvr.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                   OLTV + spread + ue + cvr, data=data3, ties="efron")

cox.NUM_BO.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                      OLTV + spread + ue + cvr + NUM_BO, data=data3, ties="efron")

cox.PURPOSE.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                       OLTV + spread + ue + cvr + NUM_BO + PURPOSE, data=data3, ties="efron")

cox.PROP_TYP.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                        OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                        PROP_TYP, data=data3, ties="efron")

cox.OCC_STAT.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                        OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                        PROP_TYP + OCC_STAT, data=data3, ties="efron")

cox.DTI.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq + 
                   OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                   PROP_TYP + OCC_STAT + DTI, data=data3, ties="efron")

cox.FTHB_FLG.p <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq +
                        OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                        PROP_TYP + OCC_STAT + DTI + FTHB_FLG, data=data3, ties="efron")

# compare AIC of models
AIC.prepay <- rbind(AIC(cox.v2), AIC(cox.OLTV.p), AIC(cox.spread.p), AIC(cox.ue.p), 
                     AIC(cox.cvr.p), AIC(cox.NUM_BO.p), AIC(cox.PURPOSE.p), AIC(cox.PROP_TYP.p),
                     AIC(cox.OCC_STAT.p), AIC(cox.DTI.p), AIC(cox.FTHB_FLG.p))
rownames(AIC.prepay) <- c('v2', 'OLTV', 'spread', 'ue', 'cvr', 'NUM_BO', 'PURPOSE', 'PROP_TYP', 'OCC_STAT', 'DTI', 'FTHB_FLG')
colnames(AIC.prepay) <- 'AIC'
AIC.prepay


# final prepayment model
cox.prepay1 <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq +
                          OLTV + spread + ue + cvr + NUM_BO + PURPOSE + 
                          PROP_TYP + OCC_STAT + DTI + FTHB_FLG, data=data3, ties="efron")
cox.prepay1
AIC(cox.prepay1)

cox.prepay <- coxph(formula = Surv(start,end,status=="prepaid") ~ CSCORE_B + pneq +
                       OLTV + spread + ue + cvr + PURPOSE + 
                       PROP_TYP + OCC_STAT + DTI + FTHB_FLG, data=data3, ties="efron")
cox.prepay
AIC(cox.prepay)

bh2 <- basehaz(cox.prepay,centered=FALSE)
bh2$surv <- exp(-bh2$hazard)
ggplot(bh2,aes(x=time,y=surv)) +
  geom_line() +
  xlab("Time in months") + 
  ylab("Survival function") +
  ylim(c(0,1)) +
  ggtitle("Coxph Baseline Survival Function: Model 2")

zph2 <- cox.zph(cox.prepay)
plot(zph2)


# Predict Default / Prepayment with a made-up loan

ndata <- data.frame(end = c(1:60))
ndata$start = ndata$end-1
ndata$CSCORE_B <- 720
ndata$pneq <- 0
ndata$spread <- median(data3$spread)
ndata$OCC_STAT <- median(data3$OCC_STAT)
ndata$PURPOSE <- median(data3$PURPOSE)
ndata$PROP_TYP <- median(data3$PROP_TYP)
ndata$LOAN_ID <- 1111111
ndata$status <- 'censored'
ndata$ue <- median(data3$ue)
ndata$cvr <- median(data3$cvr)
ndata$OLTV <- median(data3$OLTV)
ndata$DTI <- median(data3$DTI, na.rm = TRUE)
ndata$FTHB_FLG <- median(data3$FTHB_FLG, na.rm = TRUE)
ndata$NUM_BO <- median(data3$NUM_BO)


# Default model prediction
h <- predict(cox.default, newdata=ndata, type="expected")
df <- data.frame(t=c(1:60),h=h)
df$H <- cumsum(df$h)
df$surv<- exp(-df$H)
df$cumdef<- 1-df$surv
df
ggplot(df,aes(x=t,y=cumdef)) +
  geom_line(color="blue") +
  xlab("Time (months)") +
  ylab("Cumulative Default Probability") +
  xlim(c(0,60)) +
  ylim(c(0,.005)) +
  scale_color_discrete(name = "Cox Default model") +
  ggtitle("Cumulative Default Probability: Cox Default")


# Prepayment model prediction
h <- predict(cox.prepay, newdata=ndata, type="expected")
df <- data.frame(t=c(1:60),h=h)
df$H <- cumsum(df$h)
df$surv<- exp(-df$H)
df$cumdef<- 1-df$surv

ggplot(df, aes(x=t, y=cumdef)) + 
  geom_line(color="blue") +
  xlab("Time (months)") +
  ylab("Cumulative Prepayment Probability") +
  xlim(c(0,60)) +
  ylim(c(0,1)) +
  scale_color_discrete(name = "Cox Prepay model") +
  ggtitle("Cumulative Prepayment Probability: Cox Prepay")
```
