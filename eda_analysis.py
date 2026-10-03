"""
Week 2 Task: Exploratory Data Analysis and Visualization
Dataset: Titanic passenger data (cleaned version produced in Week 1)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

df = pd.read_csv('titanic_cleaned.csv')
print("Loaded cleaned dataset:", df.shape)
print(df.dtypes)

# Human-readable labels used across charts
df['Survived_Label'] = df['survived'].map({0: 'Did not survive', 1: 'Survived'})
df['Pclass_Label'] = df['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
df['Sex_Label'] = df['sex'].str.capitalize()
df['Embarked_Label'] = df['embarked'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})

# ------------------------------------------------------------------
# 1. BASIC STATISTICS
# ------------------------------------------------------------------
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS - NUMERIC COLUMNS")
print(df[['age', 'fare', 'family_size', 'sibsp', 'parch']].describe().round(2))

print("\n" + "="*70)
print("OVERALL SURVIVAL RATE")
overall_rate = df['survived'].mean()
print(f"{overall_rate:.1%} of the {len(df)} passengers survived "
      f"({df['survived'].sum()} survived, {len(df)-df['survived'].sum()} did not).")

# ------------------------------------------------------------------
# 2. UNIVARIATE: Survival count + rate
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(data=df, x='Survived_Label', hue='Survived_Label',
              palette=['#c0392b', '#27ae60'], ax=axes[0], legend=False)
axes[0].set_title('Passenger Survival Counts')
axes[0].set_xlabel('Outcome')
axes[0].set_ylabel('Number of Passengers')
for container in axes[0].containers:
    axes[0].bar_label(container)

axes[1].pie(df['survived'].value_counts(), labels=['Did not survive', 'Survived'],
            autopct='%1.1f%%', colors=['#c0392b', '#27ae60'], startangle=90,
            wedgeprops={'edgecolor': 'white'})
axes[1].set_title('Survival Rate (%)')
plt.tight_layout()
plt.savefig('chart_01_survival_overview.png')
plt.close()

# ------------------------------------------------------------------
# 3. UNIVARIATE: Age and Fare distributions
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df['age'], bins=30, kde=True, color='#2980b9', ax=axes[0])
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')
axes[0].axvline(df['age'].median(), color='red', linestyle='--', label=f"Median = {df['age'].median():.0f}")
axes[0].legend()

sns.histplot(df['fare'], bins=30, kde=True, color='#8e44ad', ax=axes[1])
axes[1].set_title('Fare Distribution (Raw, Right-Skewed)')
axes[1].set_xlabel('Fare (£)')
axes[1].set_ylabel('Count')
axes[1].axvline(df['fare'].median(), color='red', linestyle='--', label=f"Median = £{df['fare'].median():.0f}")
axes[1].legend()
plt.tight_layout()
plt.savefig('chart_02_age_fare_distribution.png')
plt.close()

# ------------------------------------------------------------------
# 4. UNIVARIATE: Passenger class and embarkation composition
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
order_class = ['1st Class', '2nd Class', '3rd Class']
sns.countplot(data=df, x='Pclass_Label', order=order_class, hue='Pclass_Label',
              palette='Blues_d', ax=axes[0], legend=False)
axes[0].set_title('Passengers by Ticket Class')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Number of Passengers')
for container in axes[0].containers:
    axes[0].bar_label(container)

sns.countplot(data=df, x='Embarked_Label', hue='Embarked_Label', palette='Greens_d',
              ax=axes[1], legend=False, order=df['Embarked_Label'].value_counts().index)
axes[1].set_title('Passengers by Port of Embarkation')
axes[1].set_xlabel('Port')
axes[1].set_ylabel('Number of Passengers')
for container in axes[1].containers:
    axes[1].bar_label(container)
plt.tight_layout()
plt.savefig('chart_03_class_embarked_composition.png')
plt.close()

# ------------------------------------------------------------------
# 5. BIVARIATE: Survival rate by sex and by class
# ------------------------------------------------------------------
sex_rate = df.groupby('Sex_Label')['survived'].mean()
class_rate = df.groupby('Pclass_Label')['survived'].mean().reindex(order_class)
print("\nSurvival rate by sex:\n", sex_rate.round(3))
print("\nSurvival rate by class:\n", class_rate.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bars0 = axes[0].bar(sex_rate.index, sex_rate.values, color=['#e67e22', '#16a085'])
axes[0].set_title('Survival Rate by Sex')
axes[0].set_xlabel('Sex')
axes[0].set_ylabel('Survival Rate')
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[0].bar_label(bars0, fmt='%.1f%%', labels=[f"{v:.1%}" for v in sex_rate.values])

bars1 = axes[1].bar(class_rate.index, class_rate.values, color=['#2980b9', '#f39c12', '#c0392b'])
axes[1].set_title('Survival Rate by Ticket Class')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Survival Rate')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[1].bar_label(bars1, labels=[f"{v:.1%}" for v in class_rate.values])
plt.tight_layout()
plt.savefig('chart_04_survival_by_sex_class.png')
plt.close()

# ------------------------------------------------------------------
# 6. MULTIVARIATE: Survival by class AND sex together
# ------------------------------------------------------------------
pivot = df.pivot_table(index='Pclass_Label', columns='Sex_Label', values='survived',
                        aggfunc='mean').reindex(order_class)
print("\nSurvival rate by class and sex (pivot table):\n", pivot.round(3))

plt.figure(figsize=(7, 4.5))
sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Survival Rate'}, linewidths=0.5)
plt.title('Survival Rate by Class and Sex')
plt.xlabel('Sex')
plt.ylabel('Ticket Class')
plt.tight_layout()
plt.savefig('chart_05_survival_class_sex_heatmap.png')
plt.close()

# Grouped bar version (easier to read exact magnitudes side by side)
plt.figure(figsize=(8, 4.5))
sns.barplot(data=df, x='Pclass_Label', y='survived', hue='Sex_Label',
            order=order_class, palette=['#e67e22', '#16a085'],
            errorbar=None)
plt.title('Survival Rate by Class, Split by Sex')
plt.xlabel('Ticket Class')
plt.ylabel('Survival Rate')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend(title='Sex')
plt.tight_layout()
plt.savefig('chart_06_survival_class_sex_grouped_bar.png')
plt.close()

# ------------------------------------------------------------------
# 7. BIVARIATE: Age distribution split by survival (KDE)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4.5))
sns.kdeplot(data=df[df.survived == 0], x='age', fill=True, alpha=0.4,
            color='#c0392b', label='Did not survive')
sns.kdeplot(data=df[df.survived == 1], x='age', fill=True, alpha=0.4,
            color='#27ae60', label='Survived')
plt.title('Age Distribution by Survival Outcome')
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.legend(title='Outcome')
plt.tight_layout()
plt.savefig('chart_07_age_by_survival_kde.png')
plt.close()

# ------------------------------------------------------------------
# 8. BIVARIATE: Fare by class (boxplot) - shows spread & outliers
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x='Pclass_Label', y='fare', hue='Pclass_Label', order=order_class,
            palette='Blues', legend=False)
plt.title('Fare Distribution by Ticket Class')
plt.xlabel('Ticket Class')
plt.ylabel('Fare (£)')
plt.tight_layout()
plt.savefig('chart_08_fare_by_class_boxplot.png')
plt.close()

# ------------------------------------------------------------------
# 9. BIVARIATE: Family size vs survival rate
# ------------------------------------------------------------------
family_rate = df.groupby('family_size')['survived'].agg(['mean', 'count'])
family_rate = family_rate[family_rate['count'] >= 5]  # drop very sparse group sizes
print("\nSurvival rate by family size (groups with >=5 passengers):\n", family_rate.round(3))

plt.figure(figsize=(8, 4.5))
bars = plt.bar(family_rate.index.astype(str), family_rate['mean'], color='#16a085')
plt.title('Survival Rate by Family Size Aboard')
plt.xlabel('Family Size (self + siblings/spouses + parents/children)')
plt.ylabel('Survival Rate')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.bar_label(bars, labels=[f"{v:.0%}" for v in family_rate['mean']])
plt.tight_layout()
plt.savefig('chart_09_survival_by_family_size.png')
plt.close()

# ------------------------------------------------------------------
# 10. BIVARIATE: Survival by embarkation port
# ------------------------------------------------------------------
embark_rate = df.groupby('Embarked_Label')['survived'].mean().sort_values(ascending=False)
print("\nSurvival rate by embarkation port:\n", embark_rate.round(3))

plt.figure(figsize=(7, 4.5))
bars = plt.bar(embark_rate.index, embark_rate.values, color=['#2980b9', '#e67e22', '#c0392b'])
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port')
plt.ylabel('Survival Rate')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.bar_label(bars, labels=[f"{v:.1%}" for v in embark_rate.values])
plt.tight_layout()
plt.savefig('chart_10_survival_by_embarked.png')
plt.close()

# ------------------------------------------------------------------
# 11. MULTIVARIATE: Age vs Fare scatter, colored by survival, sized by class
# ------------------------------------------------------------------
plt.figure(figsize=(8.5, 5))
sns.scatterplot(data=df, x='age', y='fare', hue='Survived_Label', style='Pclass_Label',
                 palette={'Survived': '#27ae60', 'Did not survive': '#c0392b'}, alpha=0.7, s=60)
plt.title('Age vs. Fare, Colored by Survival and Styled by Class')
plt.xlabel('Age (years)')
plt.ylabel('Fare (£)')
plt.ylim(-10, 300)  # zoom past the extreme tail for readability
plt.legend(title='Outcome / Class', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart_11_age_fare_survival_scatter.png')
plt.close()

# ------------------------------------------------------------------
# 12. CORRELATION HEATMAP (numeric features)
# ------------------------------------------------------------------
numeric_cols = ['survived', 'pclass', 'age', 'fare', 'sibsp', 'parch', 'family_size', 'has_cabin_record']
corr = df[numeric_cols].corr()
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig('chart_12_correlation_heatmap.png')
plt.close()

# ------------------------------------------------------------------
# 13. Age groups (binned) vs survival - transformation example
# ------------------------------------------------------------------
bins = [0, 12, 18, 35, 60, 100]
labels = ['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-35)', 'Adult (36-60)', 'Senior (60+)']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
age_group_rate = df.groupby('age_group', observed=True)['survived'].mean().reindex(labels)
print("\nSurvival rate by age group (binned):\n", age_group_rate.round(3))

plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(age_group_rate.index, age_group_rate.values, color='#8e44ad')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(rotation=15)
plt.bar_label(bars, labels=[f"{v:.0%}" for v in age_group_rate.values])
plt.tight_layout()
plt.savefig('chart_13_survival_by_age_group.png')
plt.close()

print("\nAll charts saved. EDA script complete.")
