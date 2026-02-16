import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

challenges = pd.read_csv('../99_challenges_adressed.csv')
memory = pd.read_csv('../99_memory_type.csv', header=None, names=['id', 'memory_type'])

merged = pd.merge(challenges, memory, on='id', how='inner')

def split_clean(s):
	if pd.isna(s):
		return []
	parts = [p.strip() for p in str(s).split(';')]
	return [p for p in parts if p and p != 'N/A']

merged['primary_challenges_list'] = merged['primary_challenges'].apply(split_clean)
merged['memory_type_list'] = merged['memory_type'].apply(split_clean)

long_df = merged.explode('primary_challenges_list').explode('memory_type_list')
long_df = long_df.dropna(subset=['primary_challenges_list', 'memory_type_list'])

# Calculate unique paper counts per memory type (correct denominator)
memory_only = merged[['id', 'memory_type_list']].explode('memory_type_list')
memory_only = memory_only.dropna(subset=['memory_type_list'])
unique_counts_per_memory = memory_only.groupby('memory_type_list')['id'].nunique()

df_absolute = pd.crosstab(
	long_df['primary_challenges_list'],
	long_df['memory_type_list']
)

# Column-wise normalization: divide by unique paper counts per memory type
df_normalized = df_absolute.div(unique_counts_per_memory, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Primary Challenges vs. Memory Types (Absolute)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Memory Type', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('02_heatmap_challenges_memory_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Memory Type'})
plt.title('Intersection Analysis: Primary Challenges vs. Memory Types (Normalized)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Memory Type', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('02_heatmap_challenges_memory_normalized.pdf', dpi=300)
plt.close()
