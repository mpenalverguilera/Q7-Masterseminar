import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
challenges = pd.read_csv('../99_challenges_adressed.csv')
corrections = pd.read_csv('../99_self_correction.csv')

# Merge on id
merged = pd.merge(challenges, corrections, on='id', how='inner')

def split_clean(s: pd.Series | str):
	if pd.isna(s):
		return []
	parts = [p.strip() for p in str(s).split(';')]
	return [p for p in parts if p and p != 'N/A']

# Prepare lists
merged['primary_challenges_list'] = merged['primary_challenges'].apply(split_clean)
merged['self_correction_list'] = merged['self_correction'].apply(split_clean)

# Explode to long form (cartesian per id)
long_df = merged.explode('primary_challenges_list').explode('self_correction_list')
long_df = long_df.dropna(subset=['primary_challenges_list', 'self_correction_list'])

# Calculate unique paper counts per mechanism (correct denominator)
mechanisms_only = merged[['id', 'self_correction_list']].explode('self_correction_list')
mechanisms_only = mechanisms_only.dropna(subset=['self_correction_list'])
unique_counts_per_mech = mechanisms_only.groupby('self_correction_list')['id'].nunique()

# Crosstab counts
df_absolute = long_df.groupby(['primary_challenges_list', 'self_correction_list']).size().unstack(fill_value=0)

# Column-wise normalization: divide by unique paper counts per mechanism
df_normalized = df_absolute.div(unique_counts_per_mech, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Primary Challenges vs. Self-Correction Mechanisms (Absolute)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Self-Correction Mechanism', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('01_heatmap_challenges_correction_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Mechanism'})
plt.title('Intersection Analysis: Primary Challenges vs. Self-Correction Mechanisms (Normalized)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Self-Correction Mechanism', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('01_heatmap_challenges_correction_normalized.pdf', dpi=300)
plt.close()
