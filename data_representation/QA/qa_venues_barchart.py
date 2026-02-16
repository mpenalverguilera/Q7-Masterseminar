import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
quality_df = pd.read_csv('../99_quality_assesment.csv')
venues_df = pd.read_csv('../99_venues.csv')

# Merge the dataframes
merged_df = pd.merge(quality_df, venues_df, on='id')

# Define the main venues to track
main_venues = {
    'AAMAS': 'AAMAS',
    'ICRA': 'ICRA',
    'AAAI': 'AAAI',
    'IEEE Access': 'IEEE Access',
    'IROS': 'IROS',
    'arXiv': 'arXiv',
    'Frontiers in Robotics and AI': 'Frontiers in Robotics and AI'
}

# Create a venue category column
def categorize_venue(venue):
    for key, val in main_venues.items():
        if key in venue:
            return val
    return 'Other Venues'

merged_df['venue_category'] = merged_df['venue_name_clean'].apply(categorize_venue)

# Calculate average scores by venue category
avg_scores = merged_df.groupby('venue_category')['total'].mean().reset_index()
avg_scores.columns = ['venue', 'avg_score']

# Count papers in each category
paper_counts = merged_df.groupby('venue_category').size().reset_index(name='count')
avg_scores = pd.merge(avg_scores, paper_counts, left_on='venue', right_on='venue_category')

# Sort by average score descending
avg_scores = avg_scores.sort_values('avg_score', ascending=False)

print(f"Average scores by venue category:\n{avg_scores}\n")

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors
colors = ['#1f77b4' if v != 'Other Venues' else '#d62728' for v in avg_scores['venue']]

bars = ax.bar(avg_scores['venue'], avg_scores['avg_score'], color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for i, (bar, count) in enumerate(zip(bars, avg_scores['count'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}\n(n={count})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize the plot
ax.set_xlabel('Venue', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Quality Score', fontsize=12, fontweight='bold')
ax.set_title('Average Quality Assessment Score by Venue', fontsize=14, fontweight='bold')
ax.set_ylim(0, 6.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add a note about Other Venues
other_count = avg_scores[avg_scores['venue'] == 'Other Venues']['count'].values[0] if 'Other Venues' in avg_scores['venue'].values else 0
fig.text(0.12, 0.02, f'Note: "Other Venues" includes {other_count} papers from various venues', 
         fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('qa_venues_barchart.pdf', dpi=300, bbox_inches='tight')
print("Bar chart saved as 'qa_venues_barchart.pdf'")
plt.close()
