# Predicting Mortgage Default and Prepayment via Cox Proportional Hazards Models


I worked on 2 similar projects that predict mortgage default and prepayment.


## Project 1: Predicting Mortgage Default and Prepayment via Cox Proportional Hazards Models (R)

Tiancheng Xu, Ford Danielsen, Rishabh Kumar, Sundar Ryali, Van Xu, Evelyn Zhang

Apr 2022


### Summary
We performed ETL on 3M+ data from mortgage loans, mortgage rates, national unemployment rates and House Price Indices; using survival package in R, we trained multiple coxph models and determined the best choice of variables through cross-validationa and comparison of AICs; we predicted default/prepayment; we visualized cumulative probabilities using ggplot2 to report the default/prepayment likelihood of a given loan.


### Default Model

<img width="359" alt="image" src="https://user-images.githubusercontent.com/63265930/172236492-74d2f774-0a36-4912-be26-4c0fc8e585e5.png">


Schoenfeld Residuals of some selected variables

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172236646-0ebfe143-5f80-46a9-bbe8-751db34ddc55.png">

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172236685-0e41a8bb-b294-4942-ba87-83c51b5737fd.png">


Cumulative Default Probability of the model

<img width="461" alt="image" src="https://user-images.githubusercontent.com/63265930/172236810-be29d09f-0207-4558-a8f1-60f9e51e8087.png">



### Prepay Model

<img width="405" alt="image" src="https://user-images.githubusercontent.com/63265930/172236994-7c98caec-790e-4e0f-a804-b89e4400978a.png">


Schoenfeld Residuals of some selected variables

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172237065-e1e00b87-d917-4e15-99eb-d0dd0403c96f.png">

<img width="325" alt="image" src="https://user-images.githubusercontent.com/63265930/172237121-85f4e60b-dcb2-4a73-9efa-55ba2be9f900.png">


Cumulative Prepayment Probability of the model

<img width="449" alt="image" src="https://user-images.githubusercontent.com/63265930/172237252-2de9dcff-cddb-49b8-847d-faa0eb7df70d.png">



### Prediction

After building the models, we also tried to predict the default / prepayment status on an imaginary loan. See the R code below for details.



R Code:

```
library(ggplot2)
library(survival)

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

Project 2: Exploratory Data Analysis and Default/Prepayment Models for Unison (Python)
Using lifeline package in Python, trained multiple coxph models and compared the best choice of variables; applied k-fold cross-validation; predicted default/prepayment; visualized mortgage default/prepayment distribution with matplotlib.
