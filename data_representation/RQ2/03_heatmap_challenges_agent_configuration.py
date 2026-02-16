import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
challenges = pd.read_csv('../99_challenges_adressed.csv')
agent_config = pd.read_csv('../99_agent_configuration.csv')

# Merge on id
merged = pd.merge(challenges, agent_config, on='id', how='inner')

def split_clean(s):
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).split(';')]
    return [p for p in parts if p and p != 'N/A']

# Prepare lists
merged['primary_challenges_list'] = merged['primary_challenges'].apply(split_clean)
merged['agent_configuration_list'] = merged['agent_configuration'].apply(split_clean)

# Explode to long form
long_df = merged.explode('primary_challenges_list').explode('agent_configuration_list')
long_df = long_df.dropna(subset=['primary_challenges_list', 'agent_configuration_list'])

# Calculate unique paper counts per agent configuration (correct denominator)
config_only = merged[['id', 'agent_configuration_list']].explode('agent_configuration_list')
config_only = config_only.dropna(subset=['agent_configuration_list'])
unique_counts_per_config = config_only.groupby('agent_configuration_list')['id'].nunique()

# Crosstab counts
df_absolute = long_df.groupby(['primary_challenges_list', 'agent_configuration_list']).size().unstack(fill_value=0)

# Column-wise normalization: divide by unique paper counts per configuration
df_normalized = df_absolute.div(unique_counts_per_config, axis=1) * 100

# Plot absolute counts
plt.figure(figsize=(10, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_absolute, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})
plt.title('Intersection Analysis: Primary Challenges vs. Agent Configuration (Absolute)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Agent Configuration', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('07_heatmap_challenge_agent_configuration_absolute.pdf', dpi=300)
plt.close()

# Plot normalized percentages
plt.figure(figsize=(10, 8))
sns.set_theme(style="white")
ax = sns.heatmap(df_normalized, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Percentage (%) within Configuration'})
plt.title('Intersection Analysis: Primary Challenges vs. Agent Configuration (Normalized)', fontsize=16, pad=20)
plt.ylabel('Primary Challenge Addressed', fontsize=12)
plt.xlabel('Agent Configuration', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('07_heatmap_challenge_agent_configuration_normalized.pdf', dpi=300)
plt.close()