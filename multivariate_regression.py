import pandas as pd
import numpy as np

# Read the Excel's data in pandas DataFrame 
df = pd.read_excel('CigaretteData.xlsx')
# Transform the columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')
# Select the columns corresponding to the variables x(1), x(2) and x(3)
X = df[['x^(2)', 'x^(3)', 'x^(1)']].copy()
# Add a column of 1 for the constant term
X['constant'] = 1
# Convert the DataFrame to a numpy matrix
X = X.values
# Get the y vector
y = df['y'].values

#Calculate the β slope vector
XT = X.T
beta = np.linalg.inv(XT @ X) @ XT @ y
print(beta)

# resultat: [ 0.96257386 -2.63166111 -0.13048185  3.20219002]
# Calculer le vecteur de prédiction y*
# Calculate the y* prediction vector
y_star = X @ beta

print(y_star)

# resultat: 
# [14.38268907 15.6710899  26.39260753  9.01848077  5.97261646 14.78793719
#   9.53881179 12.51765828 16.111168   14.74466532 13.60565048 15.24700336
#   9.08358663 11.97617472  9.80679411  3.72020663 16.13019195 12.5453055
#  15.75955201  6.30965793 14.37013799  8.49571539  9.53800296 15.02511274
#  12.44918328]

# Calculate the total sum of squares
SST = np.sum((y - np.mean(y))**2)

# Calculate the sum of squares of the regression
SSR = np.sum((y_star - np.mean(y))**2)

# Calculate the sum of squares of the errors
SSE = SST - SSR

# Calculate the degrees of freedom
n = len(y)  # number of observations
p = len(beta)  # number of predictors
df_model = p - 1  # degrees of freedom of the model
df_resid = n - p  # degrees of freedom of the residuals

# Calculate the F-statistic
F = (SSR / df_model) / (SSE / df_resid)

# Build the ANOVA table
anova_table = pd.DataFrame({
    'Source of Variation': ['Regression', 'Residual', 'Total'],
    'Degrees of Freedom': [df_model, df_resid, n-1],
    'Sum of Squares': [SSR, SSE, SST],
    'Mean Square': [SSR/df_model, SSE/df_resid, np.nan],
    'F-statistic': [F, np.nan, np.nan]
})

print(anova_table)

#resultat 
# Source of Variation  Degrees of Freedom  Sum of Squares  Mean Square  F-statistic
# 0          Regression                   3      495.257814   165.085938    78.983834
# 1            Residual                  21       43.892586     2.090123          NaN
# 2               Total                  24      539.150400          NaN          NaN

# Calculate the determination coefficient R²
R_squared = SSR / SST

print(R_squared)
#result: 0.9185893479475149

# Create a DataFrame with the predictors
df_predictors = df[['x^(2)', 'x^(3)', 'x^(1)']]

# Calculate the correlation matrix
corr_matrix = df_predictors.corr()

print(corr_matrix)
#result:
#           x^(2)     x^(3)     x^(1)
# x^(2)  1.000000  0.976608  0.490765
# x^(3)  0.976608  1.000000  0.500183
# x^(1)  0.490765  0.500183  1.000000

# Create a new matrix X with only two predictors
X_reduced = df[['x^(3)', 'x^(1)']].copy()
X_reduced['constant'] = 1
X_reduced = X_reduced.values

# Calculate the new slope vector β
XT_reduced = X_reduced.T
beta_reduced = np.linalg.inv(XT_reduced @ X_reduced) @ XT_reduced @ y

# Calculate the new y* prediction vector
y_star_reduced = X_reduced @ beta_reduced

# Calculate the new sum of squares
SST = np.sum((y - np.mean(y))**2)
SSR_reduced = np.sum((y_star_reduced - np.mean(y))**2)
SSE_reduced = SST - SSR_reduced

# Calculate the new degrees of freedom
n = len(y)
p_reduced = len(beta_reduced)
df_model_reduced = p_reduced - 1
df_resid_reduced = n - p_reduced

# Calculate the new F-statistic
F_reduced = (SSR_reduced / df_model_reduced) / (SSE_reduced / df_resid_reduced)

# Build the new ANOVA table
anova_table_reduced = pd.DataFrame({
    'Source of Variation': ['Regression', 'Residual', 'Total'],
    'Degrees of Freedom': [df_model_reduced, df_resid_reduced, n-1],
    'Sum of Squares': [SSR_reduced, SSE_reduced, SST],
    'Mean Square': [SSR_reduced/df_model_reduced, SSE_reduced/df_resid_reduced, np.nan],
    'F-statistic': [F_reduced, np.nan, np.nan]
})

print(beta_reduced)
print(anova_table_reduced)

#resultat:
# [12.3881157   0.05882551  1.61397795]
#   Source of Variation  Degrees of Freedom  Sum of Squares  Mean Square  F-statistic
# 0          Regression                   2      462.256393   231.128196    66.127654
# 1            Residual                  22       76.894007     3.495182          NaN
# 2               Total                  24      539.150400          NaN          NaN
