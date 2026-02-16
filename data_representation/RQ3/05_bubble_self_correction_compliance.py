import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load data (paths relative to RQ3 directory)
self_corr = pd.read_csv('../99_self_correction.csv')
compliance = pd.read_csv('../99_compliance_resoning.csv')

# Merge on id
merged = pd.merge(self_corr, compliance, on='id', how='inner')

def split_clean(s):
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).split(';')]
    return [p for p in parts if p and p != 'N/A']

def map_self_correction(value: str) -> str:
    """Map raw self-correction to chart axis buckets."""
    val = value.strip().lower()
    if val.startswith('intrinsic') or 'reflection' in val:
        return 'Intrinsic / Reflection'
    if val.startswith('extrinsic') or 'feedback' in val:
        return 'Extrinsic / Feedback'
    return 'Other / Unspecified'

# Explode multi-valued fields
merged['self_corr_list'] = merged['self_correction'].apply(split_clean)
merged['compliance_list'] = merged['compliance_reasoning'].apply(split_clean)

long_df = merged.explode('self_corr_list').explode('compliance_list')
long_df = long_df.dropna(subset=['self_corr_list', 'compliance_list'])

# Apply mapping for X-axis buckets
long_df['self_corr_axis'] = long_df['self_corr_list'].apply(map_self_correction)

# Focus Y-axis categories ordering
compliance_order = ['Logic & Rule-Based', 'Agent-Based Auditing', 'Formal Verification']
self_corr_order = ['Intrinsic / Reflection', 'Extrinsic / Feedback', 'Other / Unspecified']

# Aggregate counts for bubble sizes
counts = (
    long_df
    .groupby(['self_corr_axis', 'compliance_list'])
    .size()
    .reset_index(name='count')
)

# Ensure category ordering for plotting
counts['self_corr_axis'] = pd.Categorical(counts['self_corr_axis'], categories=self_corr_order, ordered=True)
counts['compliance_list'] = pd.Categorical(counts['compliance_list'], categories=compliance_order, ordered=True)
counts = counts.sort_values(['self_corr_axis', 'compliance_list'])

# Plot bubble chart
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
ax = sns.scatterplot(
    data=counts,
    x='self_corr_axis',
    y='compliance_list',
    size='count',
    sizes=(80, 1200),
    hue='count',
    palette='YlGnBu',
    edgecolor='black',
    legend='brief'
)

# Improve readability
plt.title('Self-Correction Mechanism vs. Compliance Reasoning (Bubble Size = Studies)', fontsize=14, pad=14)
plt.xlabel('Self-Correction Mechanism', fontsize=12)
plt.ylabel('Compliance Reasoning', fontsize=12)
plt.xticks(rotation=20, ha='right')
plt.yticks(fontsize=10)

# Adjust legend for bubble sizes
handles, labels = ax.get_legend_handles_labels()
# Keep only size legend (first entry is hue label)
if handles and labels:
    ax.legend(title='Number of Studies', loc='upper right', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
output_path = Path(__file__).with_suffix('').name + '.pdf'
plt.savefig(output_path, dpi=300)
plt.close()
