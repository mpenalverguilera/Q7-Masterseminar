import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data aggregation from your dataset (handling exploded multi-values)
data = {
    'Classical Planning': [9, 5, 5, 3],
    'Formal Logic & Solvers': [7, 2, 2, 4],
    'Knowledge Representation': [6, 4, 1, 3],
    'Cognitive Architectures': [4, 0, 5, 0],
    'Reactive & Execution Control': [3, 5, 3, 3]
}

# Neural Classes as Index
index = ['Foundational Models', 'Generative & Predictive Models', 
         'Reinforcement Learning', 'Discriminative & Perception Models']

df = pd.DataFrame(data, index=index)

# Plotting
plt.figure(figsize=(12, 7))
sns.set_theme(style="white")
ax = sns.heatmap(df, annot=True, cmap="Blues", fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Studies'})

# Academic Formatting
plt.title('Intersection Analysis: Neural Classes vs. Symbolic Mechanisms', fontsize=16, pad=20)
plt.ylabel('Neural Architectural Class', fontsize=12)
plt.xlabel('Symbolic Architectural Mechanism', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('00_heatmap_neural_symbolic.pdf', dpi=300)
plt.close()