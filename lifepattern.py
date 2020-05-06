from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

kmf.fit(durations = churn_data['tenure'], event_observed = churn_data['Churn - Yes'] )

"""
 Lifelines is SCiKit_Learn friendly , Built on top of Pandas
    Only focus is survival analysis
    handles left, right and interval censored data
    Estimating Hazard Rates
    Defining personal Surviving Models
    Compare two or more survival functions
         lifelines.statistics.logrank_test()

  Kaplan Meier is Linear Regression. 
  Survival regression in additional to traditional linear regression, is used to 
  explain relationship between the survival or person and 
  characteristics.         

"""