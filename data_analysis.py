import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('adult.csv')
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

print("=" * 70)
print("DATA OVERVIEW")
print("=" * 70)
print(f"Total records: {len(df)}")
print(f"Total features: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())

print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)
pd.set_option('display.max_columns', None)
print(df.head())

print("\n" + "=" * 70)
print("DEMOGRAPHIC DISTRIBUTION")
print("=" * 70)

print("\nRace:")
print(df['race'].value_counts())

print("\nGender:")
print(df['gender'].value_counts())

print("\n" + "=" * 70)
print("INCOME BY DEMOGRAPHIC")
print("=" * 70)

# FIXED: Get the actual column names from crosstab
race_income = pd.crosstab(df['race'], df['income'], normalize='index') * 100
print("\nIncome by Race (% earning >50K):")
# The column name is ' >50K' with a space
if ' >50K' in race_income.columns:
    print(race_income[[' >50K']].sort_values(' >50K', ascending=False))
else:
    # If no space, try without
    print(race_income[['>50K']].sort_values('>50K', ascending=False))

gender_income = pd.crosstab(df['gender'], df['income'], normalize='index') * 100
print("\nIncome by Gender (% earning >50K):")
if ' >50K' in gender_income.columns:
    print(gender_income[[' >50K']])
else:
    print(gender_income[['>50K']])

print("\n" + "=" * 70)
print("INCOME BY RACE AND GENDER COMBINED")
print("=" * 70)

combined = pd.crosstab([df['race'], df['gender']], df['income'], normalize='index') * 100
print("\nCombined breakdown (% earning >50K):")
if ' >50K' in combined.columns:
    print(combined[[' >50K']].sort_values(' >50K', ascending=False))
else:
    print(combined[['>50K']].sort_values('>50K', ascending=False))

print("\n" + "=" * 70)
print("EDUCATION LEVEL BY RACE")
print("=" * 70)

edu_by_race = pd.crosstab(df['race'], df['education'])
edu_by_race['Total'] = edu_by_race.sum(axis=1)
print(edu_by_race)

print("\n" + "=" * 70)
print("OCCUPATION DISTRIBUTION BY RACE")
print("=" * 70)

occupation_by_race = pd.crosstab(df['race'], df['occupation'])
print(occupation_by_race)

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\nAge Statistics by Income:")
print(df.groupby('income')['age'].describe())

print("\nHours per Week by Income:")
print(df.groupby('income')['hours-per-week'].describe())

print("\nEducation Level by Income (% earning >50K):")
edu_income = pd.crosstab(df['education'], df['income'], normalize='index') * 100
if ' >50K' in edu_income.columns:
    print(edu_income[[' >50K']].sort_values(' >50K', ascending=False).head(10))
else:
    print(edu_income[['>50K']].sort_values('>50K', ascending=False).head(10))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
