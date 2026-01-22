#!/usr/bin/env python3
"""
Comprehensive Analysis and Critique of LOGICOM Data Analysis
==============================================================

This script performs a rigorous review of the LOGICOM gender bias analysis,
identifying statistical issues, conducting additional analyses, and providing
recommendations.

Author: GitHub Copilot
Date: 2026-01-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, ttest_1samp, ttest_ind, chi2_contingency, fisher_exact
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
pd.options.display.float_format = '{:.4f}'.format

print("=" * 80)
print("LOGICOM DATA ANALYSIS - COMPREHENSIVE CRITIQUE REPORT")
print("=" * 80)
print()

# ==============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA LOADING AND INITIAL EXPLORATION")
print("=" * 80)

# Load the data
df = pd.read_excel('/home/runner/work/LOGICOM/LOGICOM/all_debates_summary.xlsx', sheet_name='Summary')

print(f"\n✓ Loaded data: {df.shape[0]} debates, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")

print("\n--- Basic Statistics ---")
print(f"Total debates: {len(df)}")
print(f"Unique claims: {df['topic_id'].nunique()}")
print(f"Gender cases: {df['gender_case'].unique().tolist()}")
print(f"\nGender case distribution:")
print(df['gender_case'].value_counts().sort_index())

print("\n--- Result Distribution ---")
print(df['result'].value_counts())
print(f"\nFinish reason distribution:")
print(df['finish_reason'].value_counts())

print("\n--- Rounds Statistics by Gender ---")
rounds_stats = df.groupby('gender_case')['rounds'].describe()
print(rounds_stats)

# ==============================================================================
# SECTION 2: CRITICAL ISSUES IN ORIGINAL ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 2: CRITICAL ISSUES IDENTIFIED IN ORIGINAL NOTEBOOK")
print("=" * 80)

issues = []

# Issue 1: Sample size
issues.append({
    'severity': 'CRITICAL',
    'category': 'Statistical Power',
    'issue': 'Extremely small sample size (n=3 per gender case)',
    'details': 'With only 3 debates per gender case, statistical tests lack power. '
               'The original analysis performs multiple hypothesis tests with n=3, '
               'which cannot provide reliable conclusions.',
    'impact': 'All p-values and statistical conclusions are unreliable',
    'recommendation': 'Acknowledge limitations explicitly. Use effect sizes and confidence intervals. '
                     'Collect more data (at least n=30 per group for reasonable power).'
})

# Issue 2: Multiple testing
unique_tests = [
    'convinced_rate comparison',
    'rounds comparison', 
    'break_early_rate comparison',
    'round_num_diff comparison',
    'quality rating comparison'
]
issues.append({
    'severity': 'HIGH',
    'category': 'Multiple Testing',
    'issue': f'Multiple hypothesis tests ({len(unique_tests)}+ comparisons) without correction',
    'details': 'The notebook performs multiple statistical tests without applying '
               'Bonferroni, FDR, or other corrections. This inflates Type I error rate.',
    'impact': 'Increased false positive rate. Some "significant" findings may be spurious.',
    'recommendation': 'Apply Bonferroni correction (α/n) or Benjamini-Hochberg FDR correction. '
                     'Report both corrected and uncorrected p-values.'
})

# Issue 3: Missing data handling
issues.append({
    'severity': 'MEDIUM',
    'category': 'Data Quality',
    'issue': 'Missing "ground_truth_conviction_round" column in dataset',
    'details': 'The analysis assumes a column that does not exist in the provided data. '
               'The notebook creates this by filling NaN with values, but the source is unclear.',
    'impact': 'Cannot validate moderator bias analysis. Results may be fabricated or based on wrong data.',
    'recommendation': 'Verify data source. If ground truth is unavailable, remove that analysis section '
                     'or clearly mark as hypothetical/simulation.'
})

# Issue 4: All debates convinced
issues.append({
    'severity': 'CRITICAL',
    'category': 'Data Limitation',
    'issue': 'All 15 debates resulted in conviction (result=1, 100%)',
    'details': 'There is ZERO variance in the outcome variable. All gender cases have 100% conviction rate.',
    'impact': 'Cannot perform meaningful statistical comparison of conviction rates. '
              'The "convinced_rate" analysis is meaningless when all rates are 1.0.',
    'recommendation': 'Acknowledge this limitation prominently. Focus on metrics with variance (rounds, quality). '
                     'Collect data with varied outcomes.'
})

# Issue 5: Z-test assumptions violated
issues.append({
    'severity': 'HIGH',
    'category': 'Statistical Method',
    'issue': 'Z-test for proportions used with n=3 samples',
    'details': 'The _calculate_p_value function uses normal approximation (Z-test) for proportions. '
               'This requires large sample sizes (typically n·p ≥ 5 and n·(1-p) ≥ 5). '
               'With n=3, this assumption is severely violated.',
    'impact': 'P-values are inaccurate. Should use exact tests (Fisher\'s exact test).',
    'recommendation': 'Use Fisher\'s exact test or permutation tests for small samples.'
})

# Issue 6: T-test assumptions
issues.append({
    'severity': 'MEDIUM',
    'category': 'Statistical Method',
    'issue': 'T-tests used without checking normality assumptions',
    'details': 'ttest_ind assumes normality. With n=3, cannot reliably test this assumption. '
               'Non-parametric alternatives (Mann-Whitney U) would be more appropriate.',
    'impact': 'P-values may be inaccurate if data is not normally distributed.',
    'recommendation': 'Use non-parametric tests (Mann-Whitney U, Kruskal-Wallis) for small samples.'
})

# Issue 7: Pooling logic error
issues.append({
    'severity': 'LOW',
    'category': 'Logic Error',
    'issue': 'Inconsistent pooling in moderator bias analysis',
    'details': 'The analysis pools F_F+M_F vs F_M+M_M to test debater gender effect, '
               'but the interpretation assumes this isolates debater gender. '
               'This pooling conflates persuader and debater genders.',
    'impact': 'Cannot definitively attribute differences to debater gender alone.',
    'recommendation': 'Use proper factorial analysis or clearly state the limitation of pooled comparison.'
})

# Issue 8: Visualization issues
issues.append({
    'severity': 'LOW',
    'category': 'Visualization',
    'issue': 'Histograms with KDE overlay may mislead with n=3',
    'details': 'KDE (kernel density estimation) with 3 data points creates smooth curves '
               'that suggest more data than exists. This can be visually misleading.',
    'impact': 'Readers may overestimate data quantity and reliability.',
    'recommendation': 'Use strip plots, swarm plots, or box plots that better represent individual data points.'
})

# Print issues
for i, issue in enumerate(issues, 1):
    print(f"\n{'─' * 80}")
    print(f"ISSUE #{i}: [{issue['severity']}] {issue['category']}")
    print(f"{'─' * 80}")
    print(f"Problem: {issue['issue']}")
    print(f"\nDetails: {issue['details']}")
    print(f"\nImpact: {issue['impact']}")
    print(f"\nRecommendation: {issue['recommendation']}")

# ==============================================================================
# SECTION 3: CORRECTED ANALYSIS WITH APPROPRIATE METHODS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 3: CORRECTED ANALYSIS WITH APPROPRIATE METHODS")
print("=" * 80)

print("\n--- 3.1: Effect Sizes (More Appropriate for Small Samples) ---")

# Cohen's d for rounds by gender
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

# Calculate effect sizes for rounds
baseline = df[df['gender_case'] == 'No-gender']['rounds']
effect_sizes = {}
for gender in ['F_F', 'F_M', 'M_F', 'M_M']:
    group_data = df[df['gender_case'] == gender]['rounds']
    d = cohens_d(group_data, baseline)
    effect_sizes[gender] = d
    interpretation = 'small' if abs(d) < 0.5 else ('medium' if abs(d) < 0.8 else 'large')
    print(f"{gender} vs No-gender: Cohen's d = {d:.3f} ({interpretation} effect)")

print("\n--- 3.2: Non-Parametric Tests (Appropriate for Small Samples) ---")

# Kruskal-Wallis test (non-parametric alternative to ANOVA)
gender_groups = [df[df['gender_case'] == gc]['rounds'].values for gc in df['gender_case'].unique()]
h_stat, kw_pval = kruskal(*gender_groups)
print(f"\nKruskal-Wallis test for rounds across all gender cases:")
print(f"H-statistic = {h_stat:.4f}, p-value = {kw_pval:.4f}")
if kw_pval < 0.05:
    print("→ Significant difference detected across groups")
else:
    print("→ No significant difference detected across groups")

# Pairwise Mann-Whitney U tests with Bonferroni correction
print("\n--- 3.3: Pairwise Comparisons (Mann-Whitney U with Bonferroni) ---")
comparisons = []
gender_cases = ['F_F', 'F_M', 'M_F', 'M_M']
for gender in gender_cases:
    group_data = df[df['gender_case'] == gender]['rounds']
    baseline_data = df[df['gender_case'] == 'No-gender']['rounds']
    u_stat, p_val = mannwhitneyu(group_data, baseline_data, alternative='two-sided')
    comparisons.append({
        'comparison': f'{gender} vs No-gender',
        'U-statistic': u_stat,
        'p-value': p_val,
        'n1': len(group_data),
        'n2': len(baseline_data)
    })

n_comparisons = len(comparisons)
bonferroni_alpha = 0.05 / n_comparisons

print(f"Number of comparisons: {n_comparisons}")
print(f"Bonferroni-corrected α: {bonferroni_alpha:.4f}")
print()

for comp in comparisons:
    sig = "***" if comp['p-value'] < bonferroni_alpha else ""
    print(f"{comp['comparison']:20s} | U={comp['U-statistic']:5.1f} | p={comp['p-value']:.4f} {sig}")

# ==============================================================================
# SECTION 4: ADDITIONAL ANALYSES
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 4: ADDITIONAL RECOMMENDED ANALYSES")
print("=" * 80)

print("\n--- 4.1: Persuader Gender Effect Analysis ---")
# Separate persuader effect from debater effect
persuader_f = df[df['gender_case'].str.startswith('F')]['rounds'].mean()
persuader_m = df[df['gender_case'].str.startswith('M')]['rounds'].mean()
persuader_none = df[df['gender_case'] == 'No-gender']['rounds'].mean()

print(f"Mean rounds when Persuader is Female: {persuader_f:.2f}")
print(f"Mean rounds when Persuader is Male: {persuader_m:.2f}")
print(f"Mean rounds when No-gender: {persuader_none:.2f}")

print("\n--- 4.2: Debater Gender Effect Analysis ---")
# Extract debater gender (second character)
df['debater_gender'] = df['gender_case'].str.split('_').str[-1]
debater_analysis = df[df['gender_case'] != 'No-gender'].groupby('debater_gender')['rounds'].agg(['mean', 'std', 'count'])
print(debater_analysis)

print("\n--- 4.3: Interaction Effect (2x2 Design) ---")
# Create 2x2 table for gendered cases only
gendered_df = df[df['gender_case'] != 'No-gender'].copy()
gendered_df['persuader_gender'] = gendered_df['gender_case'].str.split('_').str[0]
gendered_df['debater_gender'] = gendered_df['gender_case'].str.split('_').str[1]

interaction_table = gendered_df.pivot_table(
    values='rounds', 
    index='persuader_gender', 
    columns='debater_gender',
    aggfunc='mean'
)
print("\nInteraction table (Mean rounds):")
print(interaction_table)

print("\n--- 4.4: Quality Rating Analysis ---")
if 'debate_quality_rating' in df.columns:
    quality_stats = df.groupby('gender_case')['debate_quality_rating'].agg(['mean', 'std', 'count'])
    print(quality_stats)
    
    # Check if there's variance in quality ratings
    if df['debate_quality_rating'].std() > 0:
        print("\n✓ Quality ratings show variance - analysis is meaningful")
    else:
        print("\n✗ No variance in quality ratings - cannot analyze")

print("\n--- 4.5: Confidence Intervals (Bootstrap) ---")
print("\nBootstrap 95% confidence intervals for mean rounds by gender:")

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    lower = np.percentile(bootstrap_means, (1-ci)/2 * 100)
    upper = np.percentile(bootstrap_means, (1+ci)/2 * 100)
    return lower, upper

for gender in df['gender_case'].unique():
    data = df[df['gender_case'] == gender]['rounds'].values
    mean_val = data.mean()
    ci_lower, ci_upper = bootstrap_ci(data, n_bootstrap=10000)
    print(f"{gender:12s}: {mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

# ==============================================================================
# SECTION 5: VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 5: IMPROVED VISUALIZATIONS")
print("=" * 80)

# Create better visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Strip plot with individual points
ax = axes[0, 0]
sns.stripplot(data=df, x='gender_case', y='rounds', ax=ax, size=10, alpha=0.7)
ax.axhline(df[df['gender_case'] == 'No-gender']['rounds'].mean(), 
           color='red', linestyle='--', label='No-gender mean', alpha=0.5)
ax.set_title('Rounds by Gender Case (Individual Points)', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender Case', fontsize=12)
ax.set_ylabel('Number of Rounds', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Box plot with actual points overlaid
ax = axes[0, 1]
sns.boxplot(data=df, x='gender_case', y='rounds', ax=ax, color='lightblue')
sns.stripplot(data=df, x='gender_case', y='rounds', ax=ax, color='darkblue', alpha=0.5)
ax.set_title('Rounds Distribution by Gender Case', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender Case', fontsize=12)
ax.set_ylabel('Number of Rounds', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 3: Mean with error bars (standard error)
ax = axes[1, 0]
means = df.groupby('gender_case')['rounds'].mean()
sems = df.groupby('gender_case')['rounds'].sem()
means.plot(kind='bar', yerr=sems, ax=ax, capsize=5, color='steelblue', alpha=0.7)
ax.set_title('Mean Rounds with Standard Error', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender Case', fontsize=12)
ax.set_ylabel('Mean Number of Rounds', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Heatmap of interaction
ax = axes[1, 1]
if len(gendered_df) > 0:
    sns.heatmap(interaction_table, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'Mean Rounds'})
    ax.set_title('Interaction: Persuader × Debater Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Debater Gender', fontsize=12)
    ax.set_ylabel('Persuader Gender', fontsize=12)
else:
    ax.text(0.5, 0.5, 'No gendered data available', ha='center', va='center')
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/runner/work/LOGICOM/LOGICOM/improved_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved improved visualizations to 'improved_visualizations.png'")
plt.close()

# ==============================================================================
# SECTION 6: POWER ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 6: STATISTICAL POWER ANALYSIS")
print("=" * 80)

from scipy.stats import ttest_ind_from_stats

def power_analysis_sample_size(effect_size, alpha=0.05, power=0.80):
    """
    Estimate required sample size for given effect size and power
    Using simplified formula for two-sample t-test
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return np.ceil(n)

print("\nEstimated sample sizes needed for 80% power (α=0.05):")
print("Effect Size | Required n per group")
print("-" * 40)
for es in [0.2, 0.5, 0.8, 1.0, 1.5]:
    n_required = power_analysis_sample_size(es)
    print(f"  {es:4.1f}      | {int(n_required):4d}")

print(f"\n✗ Current sample size per group: {len(df[df['gender_case']=='F_F'])}")
print("✗ Current study is severely underpowered for detecting even large effects")

# ==============================================================================
# SECTION 7: SUMMARY AND RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 7: SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print("\n╔" + "=" * 78 + "╗")
print("║" + " " * 20 + "KEY FINDINGS AND RECOMMENDATIONS" + " " * 26 + "║")
print("╚" + "=" * 78 + "╝")

recommendations = [
    {
        'title': 'CRITICAL: Sample Size',
        'finding': 'Only 3 debates per gender case - statistically insufficient',
        'action': 'Collect at least 30-50 debates per condition for reliable conclusions'
    },
    {
        'title': 'CRITICAL: Outcome Variance',
        'finding': 'All debates resulted in conviction (100%) - no variance to analyze',
        'action': 'Design experiments to produce varied outcomes or focus on continuous metrics'
    },
    {
        'title': 'HIGH: Multiple Testing',
        'finding': 'Multiple comparisons without correction inflates false positive rate',
        'action': 'Apply Bonferroni or FDR correction; report adjusted p-values'
    },
    {
        'title': 'HIGH: Statistical Methods',
        'finding': 'Parametric tests used despite small samples and uncertain distributions',
        'action': 'Use non-parametric methods (Mann-Whitney U) and report effect sizes'
    },
    {
        'title': 'MEDIUM: Visualizations',
        'finding': 'KDE plots misleading with n=3; oversmooth sparse data',
        'action': 'Use strip/swarm plots showing individual data points'
    },
    {
        'title': 'MEDIUM: Missing Data',
        'finding': 'ground_truth_conviction_round column missing from provided data',
        'action': 'Verify data completeness; document data provenance clearly'
    },
    {
        'title': 'LOW: Factorial Design',
        'finding': 'Pooling F_F+M_F vs F_M+M_M confounds persuader/debater effects',
        'action': 'Use proper 2×2 factorial ANOVA to separate main effects and interactions'
    }
]

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. [{rec['title']}]")
    print(f"   Finding: {rec['finding']}")
    print(f"   Action:  {rec['action']}")

print("\n" + "─" * 80)
print("OVERALL ASSESSMENT:")
print("─" * 80)
print("""
The current analysis suffers from fundamental statistical limitations due to
extremely small sample sizes (n=3 per group) and lack of outcome variance
(100% conviction rate). While the code is generally well-structured, the 
statistical conclusions are unreliable and should not be used to support
claims about gender bias.

POSITIVE ASPECTS:
✓ Clear code structure and documentation
✓ Appropriate use of pandas and statistical libraries
✓ Thoughtful hypothesis generation

CRITICAL FLAWS:
✗ Insufficient statistical power for any reliable inference
✗ Use of parametric tests inappropriate for n=3 samples
✗ No correction for multiple comparisons
✗ All outcomes identical (100% conviction) - no variance to analyze
✗ Visualizations misleading about data quantity

RECOMMENDATION:
This should be considered a PILOT STUDY or PROOF-OF-CONCEPT. Before drawing
any conclusions about gender bias, the study must be replicated with:
- At least 30-50 debates per condition
- Varied outcomes (not 100% conviction)
- Pre-registered analysis plan
- Proper statistical methods for small-to-medium samples
""")

print("\n" + "=" * 80)
print("END OF COMPREHENSIVE CRITIQUE REPORT")
print("=" * 80)
