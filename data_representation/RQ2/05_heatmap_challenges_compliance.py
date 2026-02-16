import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

challenges = pd.read_csv('../99_challenges_adressed.csv')
compliance = pd.read_csv('../99_compliance_resoning.csv')

merged = pd.merge(challenges, compliance, on='id', how='inner')

def split_clean(s):
	if pd.isna(s):
		return []
	parts = [p.strip() for p in str(s).split(';')]
	return [p for p in parts if p and p != 'N/A']

merged['primary_challenges_list'] = merged['primary_challenges'].apply(split_clean)
merged['compliance_reasoning_list'] = merged['compliance_reasoning'].apply(split_clean)

long_df = merged.explode('primary_challenges_list').explode('compliance_reasoning_list')
long_df = long_df.dropna(subset=['primary_challenges_list', 'compliance_reasoning_list'])

# Calculate unique paper counts per compliance reasoning approach (correct denominator)
compliance_only = merged[['id', 'compliance_reasoning_list']].explode('compliance_reasoning_list')
compliance_only = compliance_only.dropna(subset=['compliance_reasoning_list'])
unique_counts_per_compliance = compliance_only.groupby('compliance_reasoning_list')['id'].nunique()

df_absolute = long_df.groupby(['primary_challenges_list', 'compliance_reasoning_list']).size().unstack(fill_value=0)

# Column-wise normalization: divide by unique paper counts per approach
df_normalized = df_absolute.div(unique_counts_per_compliance, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Primary Challenges vs. Compliance Reasoning (Absolute)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Compliance Reasoning Approach', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('05_heatmap_challenges_compliance_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Approach'})
plt.title('Intersection Analysis: Primary Challenges vs. Compliance Reasoning (Normalized)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Compliance Reasoning Approach', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('05_heatmap_challenges_compliance_normalized.pdf', dpi=300)
plt.close()
