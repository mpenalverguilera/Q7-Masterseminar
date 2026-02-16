import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

challenges = pd.read_csv('../99_challenges_adressed.csv')
planning = pd.read_csv('../99_planning_approaches.csv')

# Rename columns for consistency
planning = planning.rename(columns={'Id': 'id', 'Nerual': 'neural_architecture'})

merged = pd.merge(challenges, planning, on='id', how='inner')

def split_clean(s):
	if pd.isna(s):
		return []
	parts = [p.strip() for p in str(s).split(';')]
	return [p for p in parts if p and p != 'N/A']

merged['primary_challenges_list'] = merged['primary_challenges'].apply(split_clean)
merged['neural_architecture_list'] = merged['neural_architecture'].apply(split_clean)

long_df = merged.explode('primary_challenges_list').explode('neural_architecture_list')
long_df = long_df.dropna(subset=['primary_challenges_list', 'neural_architecture_list'])

# Calculate unique paper counts per neural architecture (correct denominator)
neural_only = merged[['id', 'neural_architecture_list']].explode('neural_architecture_list')
neural_only = neural_only.dropna(subset=['neural_architecture_list'])
unique_counts_per_neural = neural_only.groupby('neural_architecture_list')['id'].nunique()

df_absolute = pd.crosstab(
	long_df['primary_challenges_list'],
	long_df['neural_architecture_list']
)

# Column-wise normalization: divide by unique paper counts per architecture
df_normalized = df_absolute.div(unique_counts_per_neural, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Primary Challenges vs. Neural Architectures (Absolute)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Neural Architecture', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_heatmap_challenges_neural_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Architecture'})
plt.title('Intersection Analysis: Primary Challenges vs. Neural Architectures (Normalized)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Neural Architecture', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_heatmap_challenges_neural_normalized.pdf', dpi=300)
plt.close()
