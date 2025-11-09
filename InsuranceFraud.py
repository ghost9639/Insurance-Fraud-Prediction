"""
My investigation into an insurance fraud database.

Windows and Linux users do not need to be concerned with the rest of this docline, for Mac users:

I used Emacs Plus (30.2) on Mac (Sequoia) using Brew Python (3.14.00) and brew pip (25.2), which meant that any python libraries had to be installed to a virtual environment. I have included by .venv folder in this repo for other mac users, but if you are securely minded, then you can run the folloing commands in your terminal:

cd ~/Downloads/
python3 -m venv .venv
source .venv/bin/activate
python3 InsuranceFraud.py

These will let python know where to look for functional libraries.
You can open terminal by pressing "CMD-Space" and typing in "terminal", "terminal.app" will flash up in autocomplete, if you hit "Enter", it will open terminal.
If running "python3 --version" in terminal doesn't return anything, you can enter the following command into your terminal:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python3

Just be sure to run:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall.sh)"

in your terminal once you are done using this file, as Apple Support often refuse to fix any MacBook that has homebrew installed.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy.stats import shapiro
import statsmodels.api as sm


df = pd.read_csv("data/insurance_claims.csv")

print(df)

df = df.replace(["NaN", "nan", "NULL", "None", "?", ""], np.nan) # tell pandas where the NaNs are

df = df.drop(columns='_c39')

# Checking for missing data
print("This is our set of missing-data")
print((df
 .isna()                        # if this is missing assign 0 or 1 
 .mean()                        # take the mean (this is essentially a binomial bern p)
 .mul(100)                      # put the proportion into % terms
 .pipe(lambda ser: ser[ser != 0]) # pick specifically missing data
 )) # checking for missing numerical data %

# We see that the only remaining missing data is in the authorities_contacted dataset (9.1%)


print("This is the set of non-missing data.")
print((df
 .isna()                        # if this is missing assign 0 or 1 
 .mean()                        # take the mean (this is essentially a binomial bern p)
 .mul(100)                      # put the proportion into % terms
 .pipe(lambda ser: ser[ser == 0]) # pick specifically nonmissing data
 )) # checking for missing numerical data %

print("Most of our dataset is complete, aside from specific accident information")


# Making policy length variable using pandas
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'].astype('string')) # some specific error in reading
df['policyDays'] = (df['incident_date'] - df['policy_bind_date']).dt.days

df['policyDays']                # complete and useful, more granular thann policy months



print(df.dtypes)                # smaller types

print(df.columns)
# Interesting columns:
# months as a customer, age, policy bind date, policy state, insured sex, insured education level, insured occupation, insured hubbies, incident type, collision type, incident severity, propert damage, witnesses, claim amount, frad reported

# dataviz
def scatterplot(df, RefCol1, RefCol2):
	plt.figure(figsize=(8,6))
	sns.scatterplot(data=df,x=RefCol1,y=RefCol2)
	plt.title('Title')
	plt.tight_layout()
	plt.show()


print("Let's try and understsand the distribution of these claims")
plt.boxplot(df['policyDays'], orientation='horizontal')
plt.title('Distribution of Policy Days')
plt.show()

plt.violinplot(df['total_claim_amount'], orientation='horizontal')
plt.title('Distribution of total claim amount')
plt.show()

print("The distribution of the claims could come down to entirely different accident types")

def histogram_atypes(df):
    SVCclaims = df.loc[df['incident_type']=='Single Vehicle Collision', 'total_claim_amount']
    VTclaims = df.loc[df['incident_type']=='Vehicle Theft', 'total_claim_amount']
    MvCclaims = df.loc[df['incident_type']=='Multi-vehicle Collision', 'total_claim_amount']
    PCclaims = df.loc[df['incident_type']=='Parked Car', 'total_claim_amount']
    plt.figure(figsize=(10,6))

    plt.hist(
        [SVCclaims, VTclaims, MvCclaims, PCclaims],
        bins = 20,
        edgecolor = 'black',
        stacked = True,
        label = ['Single Vehicle Collision', 'Vehicle Theft', 'Multi-vehicle Collision', 'Parked Car'],
        color = ['skyblue', 'salmon', 'green', 'yellow'],
        alpha = 0.3
    )
    plt.xlabel('Claims amount for Collision Type')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

histogram_atypes(df)
print("We can now see that the two clusters of claims are parked car and vehicle thefts at the low end, and single vehicle collisions and multi-vehicle collisions at the high end.")

print("We can also test our visual inclination to believe single vehicle collision and multi vehicle collisions are similarly distributed under the normal distribution.")

def plot_cdf(df):
    SVCclaims = df.loc[df['incident_type']=='Single Vehicle Collision', 'total_claim_amount']
    MvCclaims = df.loc[df['incident_type']=='Multi-vehicle Collision', 'total_claim_amount']
    
    plt.figure(figsize=(8,6))
    
    sns.ecdfplot(data = SVCclaims, label = "Single Vehicle Collision claims")
    sns.ecdfplot(data = MvCclaims, label = "Multi-vehicle Collision claims")
    
    plt.xlabel("Experimental CDF graphs")
    plt.ylabel(f"CDF of {SVCclaims.name} and {MvCclaims.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

plot_cdf(df)


def run_ks(df):
    SVCclaims = df.loc[df['incident_type'] == 'Single Vehicle Collision', 'total_claim_amount']
    MvCclaims = df.loc[df['incident_type'] == 'Multi-vehicle Collision', 'total_claim_amount']

    ks, p_value = stats.ks_2samp(SVCclaims, MvCclaims)

    if (p_value < 0.01):
        print("Evidence at 99% CI to reject same distribution")
    elif (p_value < 0.05):
        print("Evidence at 95% CI to reject same distribution")
    else:
        print("No statistically significant evidence to reject different sampling distributions")

    print(f"KS statistic = {ks}, p-value = {p_value}")

    print()
    if (stats.shapiro(SVCclaims).pvalue < 0.05):
        print("Evidence at 95% CI that Single Vehicle Accident claims are normally distributed")
    else:
        print("No statistical evidence that Single Vehicle Accident claims are normally distributed")
    if (stats.shapiro(MvCclaims).pvalue < 0.05):
        print("Evidence at 95% CI that multi-vehicle accident claims are normally distributed")
    else:
        print("No statistical evidence multi-vehicle accident claims are normally distirbuted")

        
run_ks(df)    
print("They don't seem to come from the same distributions, and while there is statistical evidence that single vehicle claim sizes are normally distributed, there is no such evidence for multi-vehicle claim sizes being normally distributed.")
swilkresults = stats.shapiro(df['total_claim_amount'])

def f_testing(df):
    SVCclaims = df.loc[df['incident_type'] == 'Single Vehicle Collision', 'total_claim_amount']
    MvCclaims = df.loc[df['incident_type'] == 'Multi-vehicle Collision', 'total_claim_amount']
    
    print()
    print()
    fstat = np.var(SVCclaims, ddof = 1) / np.var(MvCclaims, ddof = 1)

    print(f"Dof 1:{len(SVCclaims)-1}")
    print(f"Dof 2:{len(MvCclaims)-1}")
    print(f"f-stat: {fstat}")
    print(f"p-val: {stats.f.cdf(fstat, (len(SVCclaims)-1), (len(MvCclaims)-1))}")

    if stats.f.cdf(fstat, (len(SVCclaims)-1), (len(MvCclaims)-1)) < 0.05:
        print("Evidence at 95% CI that distributions have different standard deviance")
        
    
f_testing(df)

def plot_cdf(df):
    VTclaims = df.loc[df['incident_type']=='Vehicle Theft', 'total_claim_amount']
    PCclaims = df.loc[df['incident_type']=='Parked Car', 'total_claim_amount']
    
    plt.figure(figsize=(8,6))
    
    sns.ecdfplot(data = VTclaims, label = "Vehicle Theft claims amounts ($)")
    sns.ecdfplot(data = PCclaims, label = "Parked Car claims amounts ($))")
    
    plt.xlabel("Experimental CDF graphs")
    plt.ylabel(f"CDF of {PCclaims.name} and {PCclaims.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

plot_cdf(df)


def run_ks(df):
    VTclaims = df.loc[df['incident_type']=='Vehicle Theft', 'total_claim_amount']
    PCclaims = df.loc[df['incident_type']=='Parked Car', 'total_claim_amount']
    
    ks, p_value = stats.ks_2samp(VTclaims, PCclaims)

    print(f"p value is: {p_value}")
    print(f"Shapiro-Wilk test on Vehicle Thefts claims sizes p value was: {stats.shapiro(VTclaims).pvalue}")
    print(f"Shapiro-Wilk test on Parked Cars claims sizes p value was: {stats.shapiro(PCclaims).pvalue}")

    if (p_value < 0.01):
        print("Evidence at 99% CI to reject same distribution")
    elif (p_value < 0.05):
        print("Evidence at 95% CI to reject same distribution")
    else:
        print("No statistically significant evidence to reject different sampling distributions")

    print(f"KS statistic = {ks}, p-value = {p_value}")

    print()
    if (stats.shapiro(VTclaims).pvalue < 0.05):
        print("Evidence at 95% CI that Single Vehicle Accident claims are normally distributed")
    else:
        print("No statistical evidence that Single Vehicle Accident claims are normally distributed")
    if (stats.shapiro(PCclaims).pvalue < 0.05):
        print("Evidence at 95% CI that multi-vehicle accident claims are normally distributed")
    else:
        print("No statistical evidence multi-vehicle accident claims are normally distirbuted")

        
run_ks(df)    


df['fraud_reported'].describe() # only 247 counts of Y, care must be taken in train-test split

sns.scatterplot(
    data=df,
    x=df.index,  
    y='total_claim_amount',
    hue='fraud_reported',  
    palette={'N': 'blue', 'Y': 'red'},  
    alpha=0.7
)

plt.xlabel("Index")
plt.ylabel("Total Claim Amount")
plt.title("Total Claim Amount by Fraud Reported")
plt.legend(title="Fraud Reported")
plt.show()


# scatterplot(df, 'age', 'total_claim_amount')

df.get(["insured_sex"])
print(df.select_dtypes('category').columns)

# need to codify these for GLM

print("We can check how many dummies we'll have to make to include any particular categorical variable.")
print(df['policy_state'].unique()) # data is only from Ohio, Indiana, and Illinois, won't be too many dummies
print(df.select_dtypes('category').columns)
print(df['insured_education_level'].unique()) # high number of options, but groupable
print(df['insured_hobbies'].unique())         # too many unique hobbies to group
print(df['insured_relationship'].unique())
print(df['fraud_reported'].value_counts())

df['incident_date'] = pd.to_datetime(df['incident_date'].astype('string'))
print(df['incident_date'].describe())

print(df['age'].describe())
plt.boxplot(df['age'], orientation='horizontal')
plt.title("Distribution of claimant ages")
plt.xlabel("Age")
plt.show()

# Need to group education and relationship

PostGrad = ['MD', 'PhD', 'Masters']
UnderGrad = ['Associate', 'College', 'JD']
HighSchool = ['High School']


df['Coded Education Level'] = np.select(
    [
        df['insured_education_level'].isin(PostGrad),
        df['insured_education_level'].isin(UnderGrad),
        df['insured_education_level'].isin(HighSchool)
    ],
    ['PostGrad', 'UnderGrad', 'HighSchool'],
    default='Other'
)

df['married_status'] = df['insured_relationship'].map({
    'husband': 1,
    'wife': 1,
    'other-relative': 0,
    'own-child': 0,
    'unmarried': 0,
    'not-in-family': 0
})

print(df['Coded Education Level'].unique()) # This is reasonable for a dummy variable
print(df['married_status'].unique())

print(df['insured_sex'].describe())

# pandas can generate dummies for us to regress with
df_GLM = pd.get_dummies(df, columns=["insured_sex", "policy_state", "Coded Education Level", "fraud_reported", "incident_type", "incident_severity"], drop_first = True)

print(df_GLM.columns)
print(df_GLM)

print("This gives us the potential regressands: total_claim_amount and fraud_reported_Y.")
print("And potential regressors: age, insured_sex, policy_state, Coded Education Level, policyDays, incident_type, witnesses, and any of the other entries in the set.")
print("We can use the weight of evidence credit risk ranking to assess which parameters could be relevant to predicting fraud.")

# WoE method

def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

ivs = iv_woe(df_GLM, 'fraud_reported_Y')

# incident_date, incident_type, collision type, auth contact, total claim amount, model relevant
# including incident date in a linear model would likely incorrectly model stochastic time trends linearly, authorities contacted has 9.1% missing data, collision type is almost 20% missing

# key target: fraud_reported_Y
# Good regressors: incident type, incident severity, total claim amount, policyDays  

sns.scatterplot(
    data=df,
    x=df.index,  
    y='policyDays',
    hue='fraud_reported',  
    palette={'N': 'blue', 'Y': 'red'},  
    alpha=0.7  
)

plt.xlabel("Index")
plt.ylabel("Policy Days")
plt.title("Policy Days by Fraud Reported")
plt.legend(title="Fraud Reported")
plt.show()                      # basically random too


# The plan for the categorical data is to somehow colour by fraud rate (binned means?)
severity_order = ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']

df_GLM['incident_severity'] = pd.Categorical (
    df['incident_severity'],
    categories = severity_order,
    ordered=True
)

fraud_rat = (df_GLM.groupby('incident_severity')['fraud_reported_Y'].mean().reset_index())
severity_counts = df_GLM['incident_severity'].value_counts().reset_index()

severityViz = severity_counts.merge(fraud_rat, on = 'incident_severity')

sns.barplot(
    data = severityViz,
    x = 'incident_severity',
    y = 'count',
    hue='fraud_reported_Y',
    palette='coolwarm'
)
plt.xlabel("Index")
plt.ylabel("Incident Severity")
plt.title("Incident Severity by Fraud Reported")
plt.show()                      # Strong overincidence of fraud in major incidents, uncertain if this is because they are more robustly studied

# incident type visualisation
incident_type_order = ['Parked Car', 'Single Vehicle Collision',  'Multi-vehicle Collision', 'Vehicle Theft']

df_GLM['incident_type'] = pd.Categorical (
    df['incident_type'],
    categories = incident_type_order,
    ordered = True
)

typefraud_rat = (df_GLM.groupby('incident_type')['fraud_reported_Y'].mean().reset_index())
typeCounts = df_GLM['incident_type'].value_counts().reset_index()

typeViz = typeCounts.merge(typefraud_rat, on = 'incident_type')

sns.barplot(
    data = typeViz,
    x = 'incident_type',
    y = 'count',
    hue = 'fraud_reported_Y',
    palette = 'coolwarm'
)
plt.xlabel("Index")
plt.ylabel("Incident Type")
plt.title("Fraud Reported by Incident Type")
plt.show()                      # Appears to generally be more fraud in crashes, especially single vehicle collisions

# Actual Regressions

# key target: fraud_reported_Y
# Good regressors: incident type, incident severity, total claim amount, policyDays  

# Starting with GLM, likely logit model

X = df_GLM[['incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft', 'incident_severity_Minor Damage', 'incident_severity_Total Loss', 'incident_severity_Trivial Damage', 'total_claim_amount', 'policyDays']]
Y = df_GLM['fraud_reported_Y']

Y = np.reshape(Y, (1000, ))

clf = LogisticRegression(random_state = 10, max_iter = 1000).fit(X, Y)
clf.score(X, Y)


coef = clf.coef_[0]
inter = clf.intercept_[0]
featureNames = clf.feature_names_in_

coef_df = pd.DataFrame ({
    "Regressor": featureNames,
    "Coefficients": coef
})

coef_df["Odds Ratio"] = np.exp(coef_df['Coefficients'])

print(coef_df)

# We can also call on statsmodels for pvalues since sklearn doesn't seem to provide them by default

X_sm = X.apply(pd.to_numeric, errors = "coerce")
X_sm = sm.add_constant(X_sm)

Y = df_GLM['fraud_reported_Y']
Y_sm = Y.apply(pd.to_numeric, errors = "coerce")

X_sm = X_sm.astype({col: int for col in X_sm.select_dtypes('bool').columns})
Y_sm = Y_sm.astype(int)

smlogit = sm.Logit(Y_sm, X_sm)

result = smlogit.fit()
print(result.summary())

# So incident severity is our only statistically significant factor
# the omitted variable here is severe damage, which implies that severe damage is a strong predictor of fraud. The smallest negative effect size of the included variables is Total Loss, which implies this is the second most extreme.

# Next we'll be using gradient boosting (using the xgboost library)
# it is statistical malfeasance to select gradient boosting models using WoE (since gboosted relations are non-deterministically selected), so we will use a premade theoretical model

# key target: fraud_reported_Y
# Good regressors: incident severity, total claim amount, policyDays, Coded Education Level, insured sex

X = df_GLM[['incident_severity_Minor Damage', 'incident_severity_Total Loss', 'incident_severity_Trivial Damage', 'total_claim_amount', 'policyDays', 'Coded Education Level_PostGrad', 'Coded Education Level_UnderGrad', 'insured_sex_MALE']]
X = X.astype({col: int for col in X.select_dtypes('bool').columns})

Y = Y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=10)
model = xgb.XGBRegressor (
    objective = 'reg:squarederror',
    n_estimators=100,
    learning_rate = 0.1,
    max_depth = 6,
    random_state = 10,
    stratify = Y
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error (y_test, y_pred)

r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]

adjR2 = 1 - (1 - r2) * ((n-1) / (n - p - 1))

print(f"Mean Squared Error: {mse}")
print(f"Adjusted R^2: {adjR2}") # terrible

booster = model.get_booster()

importance_types = ['weight', 'gain']
importance_dfs = []

for imp_type in importance_types:
	scores = booster.get_score(importance_type = imp_type)
	df = pd.DataFrame.from_dict(scores, orient='index', columns = [imp_type])
	importance_dfs.append(df)
	
feature_importance = pd.concat(importance_dfs, axis=1).fillna(0)
feature_importance = feature_importance.sort_values(by='gain',ascending=False)
feature_importance.index.name = 'feature'
feature_importance = feature_importance.reset_index()

plt.figure(figsize=(6,8))
top_n = 10


def plot_weight():
	sns.barplot(
	data=feature_importance.nlargest(top_n, 'weight'),
	y='feature', x='weight',palette = 'viridis'
	)
	plt.title(f"Top {top_n} Features by Weight")
	plt.xlabel("Weight (Frequency in Trees)")
	plt.ylabel("Feature")
	plt.tight_layout()
	plt.show()

plot_weight()

def plot_gain():
	sns.barplot(
	data=feature_importance.nlargest(top_n, 'gain'),
	y='feature', x='gain',palette = 'viridis'
	)
	plt.title(f"Top {top_n} Features by Gain")
	plt.xlabel("Gain (Frequency in Trees)")
	plt.ylabel("Feature")
	plt.tight_layout()
	plt.show()



plot_gain()

# while pseudo-r2 in logit models and adjusted R2 in gradient boosting models don't really mean what we're used to wrt linear regression R2, the logit model seems to better explain errors
# There are interesting comparisons between their insights, in the logit model, the only significant variables were the incident severity dummies, whereas in the tree model, while incident types provided strong gain, the highest weight was actually allocated to the total claim amount, which implies that while the models used seem to allocate high important to the severity of the insurance claim, the size of the monetary compensation should also be considered. In all likelihood, these are two collider variables, and the tree model was just better able to catch it.



# scatterplot(df, 'policy_number', 'policyDays') # no clearly visible trend in days between policy taken out and claims, dataset should be reflective of the insured population as a whole
# scatterplot(df, 'policyDays','total_claim_amount') # unique clustering but no evidence of systematic error between claim amounts and elapsed time on policy held


# let's list potential regressions:
# 'total_claim_amount', 'fraud_reported'

# let's list potential regressors:
# policy_state can't be done since 51 dummies interferes with effect sizes
# 'age', 'policy_deductible', 'policy_annual_premium', 'insured_sex', 'insured_education_level', 'incident_type', 'number_of_vehicles_involved', 'police_report_available'




# Linear Regression
X = df.get(['age', ''])
Y = df.get(['fraud_reported'])








