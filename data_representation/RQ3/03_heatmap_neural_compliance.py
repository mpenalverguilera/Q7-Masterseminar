import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
planning = pd.read_csv('../99_planning_approaches.csv')
compliance = pd.read_csv('../99_compliance_resoning.csv')

# Rename columns for consistency
planning = planning.rename(columns={'Id': 'id', 'Nerual': 'neural_architecture'})

# Merge on id
merged = pd.merge(planning, compliance, on='id', how='inner')

def split_clean(s):
	if pd.isna(s):
		return []
	parts = [p.strip() for p in str(s).split(';')]
	return [p for p in parts if p and p != 'N/A']

# Explode both neural architectures and compliance reasoning
merged['neural_architecture_list'] = merged['neural_architecture'].apply(split_clean)
merged['compliance_reasoning_list'] = merged['compliance_reasoning'].apply(split_clean)

long_df = merged.explode('neural_architecture_list').explode('compliance_reasoning_list')
long_df = long_df.dropna(subset=['neural_architecture_list', 'compliance_reasoning_list'])

# Unique paper counts per compliance reasoning (denominator for normalization)
compliance_only = merged[['id', 'compliance_reasoning_list']].explode('compliance_reasoning_list')
compliance_only = compliance_only.dropna(subset=['compliance_reasoning_list'])
unique_counts_per_compliance = compliance_only.groupby('compliance_reasoning_list')['id'].nunique()

# Create crosstab
df_absolute = long_df.groupby(['neural_architecture_list', 'compliance_reasoning_list']).size().unstack(fill_value=0)

# Column-wise normalization: divide by unique paper counts per compliance reasoning
df_normalized = df_absolute.div(unique_counts_per_compliance, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(12, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Neural Architectures vs. Compliance Reasoning (Absolute)', fontsize=16, pad=20)
plt.ylabel('Neural Architecture', fontsize=12)
plt.xlabel('Compliance Reasoning', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_heatmap_neural_compliance_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(12, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Compliance Reasoning'})
plt.title('Intersection Analysis: Neural Architectures vs. Compliance Reasoning (Normalized)', fontsize=16, pad=20)
plt.ylabel('Neural Architecture', fontsize=12)
plt.xlabel('Compliance Reasoning', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_heatmap_neural_compliance_normalized.pdf', dpi=300)
plt.close()
