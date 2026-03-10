# LOGICOM Analysis Outputs

This directory contains all final artifacts from the corrected statistical analysis.

## 📁 Directory Structure

```
out/
├── tables/          # Main results for your paper
│   ├── expert_summary_corrected.xlsx
│   └── gender_summary_corrected.xlsx
├── data/            # Supplementary data (for reproducibility)
│   ├── expert_claim_level_aggregated.csv
│   ├── expert_any_vs_baseline_paired.csv
│   ├── gender_debates_filtered.csv
│   ├── expert_analysis_complete.xlsx
│   └── gender_analysis_complete.xlsx
├── figures/         # Publication-quality figures (to be generated)
└── README.md        # This file
```

## 📊 Main Results (Use These in Your Paper)

### Expert Experiment: `tables/expert_summary_corrected.xlsx`

**Key Statistics:**
- **Focus on: "Any-expert" row** (primary comparison)
- Aggregation: 3 repetitions → 1 per claim using **MEAN for all variables**
  - Binary (conviction): 2/3 = 0.67, not 1.0 (unbiased)
  - Continuous (rounds): Mean across 3 values
  - **No majority vote bias**: Mean preserves true rates
- Paired comparison: 102 expert claims vs. 102 baseline claims
- Tests: **Wilcoxon signed-rank for ALL outcomes** (all continuous after mean aggregation)
  - Conviction rate is now continuous (0.0-1.0) after mean aggregation
  - Cannot use McNemar's (requires binary data)
- **No multiple comparison correction** (single primary comparison)
- Individual expert types: Descriptive statistics only

**Columns to report:**
- `N_debates` - Sample size
- `Conviction_Rate (%)` - Percentage convinced
- `Mean_Rounds` - Average debate length
- `p_bonf_conviction` - Bonferroni-corrected p-value for conviction
- `p_bonf_rounds` - Bonferroni-corrected p-value for rounds  
- `OR_conviction` - Odds ratio with 95% CI
- `Cohen's_d_rounds` - Effect size with 95% CI

**Note on Any-expert:**
- Use columns with "_PAIRED" suffix for the most appropriate tests
- These account for the matched 102 claims design

### Gender Experiment: `tables/gender_summary_corrected.xlsx`

**Key Statistics:**
- Paired comparison: 200 claims across 6 gender conditions
- Tests: McNemar's test (conviction), Wilcoxon signed-rank (rounds)
- Bonferroni correction: 5 comparisons
- Baseline: "Persona-baseline"

**Comparisons:**
1. F_F (Female persuader, Female debater) vs. baseline
2. F_M (Female persuader, Male debater) vs. baseline
3. M_F (Male persuader, Female debater) vs. baseline
4. M_M (Male persuader, Male debater) vs. baseline
5. No-gender (No gender cues) vs. baseline

**Columns to report:**
- Same structure as expert table
- All p-values are from paired tests
- Check `p_bonf_*` columns for significance after correction

## 📈 Complete Analysis Results

### `data/expert_analysis_complete.xlsx`
Complete expert analysis DataFrame with all statistics:
- Individual expert types (Education, Diplomat, Health)
- Any-expert aggregated results
- All p-values (raw and Bonferroni-corrected)
- All effect sizes with confidence intervals

### `data/gender_analysis_complete.xlsx`
Complete gender analysis DataFrame with all statistics:
- All 6 gender conditions
- Paired test p-values
- Bonferroni-corrected p-values
- Effect sizes with confidence intervals

## 🔬 Supplementary Data

### `data/expert_claim_level_aggregated.csv`
- Claim-level expert data (N=102)
- Aggregated from 3 repetitions per claim using:
  - Majority vote for binary outcomes
  - Mean for continuous outcomes
- Use for claim-level analyses or verification

### `data/expert_any_vs_baseline_paired.csv`
- Paired comparison data (N=102 matched claims)
- Expert result vs. baseline result for each claim
- Use to verify paired statistical tests

### `data/gender_debates_filtered.csv`
- Gender experiment data (N=1000)
- 200 claims × 5 gender conditions
- Excludes invalid/failed debates
- Ready for additional analyses

## 📝 How to Use These Results

### For Methods Section
Report that you used:
- **Paired statistical tests** (McNemar's, Wilcoxon) for matched designs
- **Bonferroni correction** for multiple comparisons
- **Effect sizes** (Cohen's d, OR) with 95% confidence intervals
- **Claim-level aggregation** for expert data to address non-independence

### For Results Section

**Expert Experiment:**
```
We compared Any-expert (aggregated across expert types) to baseline using paired tests on 102 matched claims. Experts showed [report conviction rate and p-value] with an odds ratio of [OR and 95% CI]. Mean debate length was [rounds ± SD], significantly [shorter/longer] than baseline (Wilcoxon p = [value], Cohen's d = [d and 95% CI]).
```

**Gender Experiment:**
```
We compared five gender persona combinations to a neutral baseline using paired tests on 200 matched claims. After Bonferroni correction (α = 0.01), [number] comparisons remained significant. [Report specific significant findings with effect sizes].
```

## 🔄 Regenerating Results

If you need to rerun the analysis:

1. Open `LOGICOM_Data_Analysis_Part_2_2.ipynb`
2. Run cells 0-6 (setup & data loading)
3. Run cells 50-55 (corrected statistical functions)
4. Run cells 56-64 (expert analysis)
5. Run cells 65-70 (gender analysis)
6. Run cells 68-71 (export tables)

Output files will be overwritten in this directory.

## ⚠️ Important Notes

1. **Use CORRECTED results only**: The notebook contains old analyses with statistical issues. Only use results from the "CORRECTED" sections (cells 50-72).

2. **Any-expert paired tests**: For the expert experiment, the "Any-expert" comparison uses paired tests (look for "_PAIRED" columns). Individual expert types use independent tests.

3. **Bonferroni correction**: All p-values are provided both raw and Bonferroni-corrected. Report the corrected values in your paper.

4. **Effect sizes matter**: Don't rely only on p-values. Report effect sizes (Cohen's d, OR) to show practical significance.

## 📚 Additional Documentation

- **`../NOTEBOOK_GUIDE.md`** - Detailed guide to the analysis notebook
- **`ANALYSIS_SUMMARY.txt`** - Quick reference summary (generated when you run final export cell)

## 📞 Questions?

If you encounter issues:
1. Check `../NOTEBOOK_GUIDE.md` for detailed explanations
2. Review the "Summary of Corrections Applied" section in the notebook (Cell 72)
3. Verify you're using results from cells 50-72 only (not earlier sections)

---

**Last Updated**: Generated automatically when running corrected analysis cells

**Analysis Version**: Final corrected version with paired tests and Bonferroni correction
