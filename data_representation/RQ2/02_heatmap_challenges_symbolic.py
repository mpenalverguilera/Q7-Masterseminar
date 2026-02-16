import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

challenges = pd.read_csv('../99_challenges_adressed.csv')
planning = pd.read_csv('../99_planning_approaches.csv')

# Rename columns for consistency
planning = planning.rename(columns={'Id': 'id'})

merged = pd.merge(challenges, planning[['id', 'Symbolic']], on='id', how='inner')

def split_clean(s):
	if pd.isna(s):
		return []
	parts = [p.strip() for p in str(s).split(';')]
	return [p for p in parts if p and p != 'N/A']

merged['primary_challenges_list'] = merged['primary_challenges'].apply(split_clean)
merged['symbolic_mechanism_list'] = merged['Symbolic'].apply(split_clean)

long_df = merged.explode('primary_challenges_list').explode('symbolic_mechanism_list')
long_df = long_df.dropna(subset=['primary_challenges_list', 'symbolic_mechanism_list'])

# Calculate unique paper counts per symbolic mechanism (correct denominator)
symbolic_only = merged[['id', 'symbolic_mechanism_list']].explode('symbolic_mechanism_list')
symbolic_only = symbolic_only.dropna(subset=['symbolic_mechanism_list'])
unique_counts_per_symbolic = symbolic_only.groupby('symbolic_mechanism_list')['id'].nunique()

df_absolute = long_df.groupby(['primary_challenges_list', 'symbolic_mechanism_list']).size().unstack(fill_value=0)

# Column-wise normalization: divide by unique paper counts per mechanism
df_normalized = df_absolute.div(unique_counts_per_symbolic, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Primary Challenges vs. Symbolic Mechanisms (Absolute)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Symbolic Mechanism', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('04_heatmap_challenges_symbolic_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Mechanism'})
plt.title('Intersection Analysis: Primary Challenges vs. Symbolic Mechanisms (Normalized)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Symbolic Mechanism', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('04_heatmap_challenges_symbolic_normalized.pdf', dpi=300)
plt.close()
