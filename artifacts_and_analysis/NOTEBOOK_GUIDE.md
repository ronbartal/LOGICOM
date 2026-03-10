# LOGICOM Data Analysis Notebook Guide

## 📋 Overview

This guide explains the structure of `LOGICOM_Data_Analysis_Part_2_2.ipynb` and tells you **exactly** which sections to use for your paper and which can be deleted.

**Current Status**: The notebook contains TWO implementations:
1. **Cells 50-119**: Original corrected analyses (from parallel agent)
2. **Cells 145-155**: NEW simplified analyses (from latest corrections)

---

## 🎯 **QUICK ANSWER: What Should I Use?**

### For Your Paper - Use These Cells:

**Option A: Use the NEW Simplified Analysis (Cells 147-154)** ✅ RECOMMENDED
- Cleaner, simpler implementation
- Fixes the majority vote bias issue
- Focused on primary comparisons only (no unnecessary multiple testing)
- Located at: **Cells 147-154**

**Option B: Use Original Corrected Analysis (Cells 50-119)**
- More comprehensive but may have majority vote bias
- Includes more detailed breakdowns
- Located at: **Cells 50-119**

**Our Recommendation**: Run **Cells 147-154** to get final results with all corrections applied.

---

## 🗂️ Complete Notebook Structure

### ✅ **KEEP** - Essential Sections

#### **How to Find Cell Numbers:**
In VS Code/Cursor: Look at the top-left of each cell block - you'll see "Cell 0", "Cell 1", etc.
In Jupyter: Cell numbers show on the left as `[1]:`, `[2]:`, etc.

#### **Cells 0-6: Setup & Data Loading** ⚠️ **ALWAYS RUN THESE FIRST**
- **What**: Load libraries, data files, set constants
- **Status**: ✅ **REQUIRED** for ALL analyses
- **Key outputs**: 
  - `expert_debates_df` (306 debates)
  - `gender_debates_df` (1200 debates)
  - `masculinity_scores_df` (200 claims with scores)
  - Constants: `EXPERT_BASELINE`, `GENDER_BASELINE`

#### **Preprocessing Steps (Cells ~50-100)**

**⚠️ IMPORTANT: Different analyses need different preprocessing!**

**For Cells 147-154 (Basic Expert & Gender Analyses)**:
- ✅ **NO preprocessing needed!** 
- Just run Cells 0-6, then jump directly to Cells 147-154
- These cells work directly with `expert_debates_df` and `gender_debates_df`

**For Gender × Masculinity Analysis** (not yet in simplified cells):
- ⚠️ **Preprocessing REQUIRED**
- Need to run: Cells that create `masculinity_scores_debates_df`
- This merges gender debates with masculinity scores
- Look for: "Preprocess Domain Masculinity Data" section

---

#### **Cells 147-154: 🎯 FINAL CORRECTED ANALYSES (NEW)** ⭐ **START HERE**

**Cell 144**: `raise Exception("stop")` - Marker separating old from new analyses

**Cells 145-146**: Empty cells

**Cell 147**: **Expert Claim-Level Aggregation** (PART 1: EXPERT ANALYSIS)
- Aggregates k=3 repetitions using **MEAN** (not majority vote - avoids bias!)
- Creates `expert_claim_level` (N=102) and `baseline_data` (N=200)
- **Key fix**: Uses mean for binary outcomes to avoid upward bias

**Cell 148**: **Expert Statistical Tests**
- Mann-Whitney U tests for all comparisons (conviction, rounds, self-conviction)
- Compares Any-expert vs. No-gender baseline
- **No Bonferroni correction** (only 1 primary comparison)
- **Outputs**: Conviction rate %, rounds (M±SD), p-values

**Cell 149**: **Expert Effect Sizes**
- Cohen's d for rounds
- Proportion differences for conviction rates
- **Outputs**: Effect sizes with interpretation

**Cell 150**: **Gender Debater Aggregation** (PART 2: GENDER ANALYSIS)
- Female debaters: F_F + M_F
- Male debaters: M_M + F_M  
- Pivots data by topic_id for paired tests

**Cell 152**: **Gender Pivot Tables**
- Creates claim-level aggregates
- Combines across persuader genders (mean)

**Cell 152**: **Gender McNemar's Test**
- Tests conviction rate difference (paired binary)
- **Recognizes paired design** (same 200 claims)
- **Outputs**: McNemar statistic, p-value, contingency table

**Cell 153**: **Gender Wilcoxon Test** (Note: This appears as Cell 154 in the notebook - there may be an empty cell)
- Tests rounds difference (paired continuous)
- **Outputs**: Mean difference, Wilcoxon W, p-value

**Cells 154-156**: Summary and notes cells

**✅ TO USE THESE RESULTS**:
1. Run Cells 0-6 first (setup)
2. Run Cells 147-154 (corrected analyses)
3. Copy the printed results to your paper
4. Note the p-values and effect sizes

---

### 📦 **OPTIONAL** - Original Corrected Analyses

#### **Cells 50-119: Original Implementation**

These cells contain a more elaborate implementation from a parallel effort:

**Cells 50-55**: Corrected statistical helper functions
- Bonferroni correction, effect size functions
- Paired test functions
- **Note**: Some functions superseded by Cells 145-155

**Cells 56-70**: CORRECTED EXPERT ANALYSIS (original)
- Claim-level aggregation
- Paired tests for "Any-expert"
- May include majority vote (check if updated)

**Cells 71-90**: CORRECTED GENDER ANALYSIS (original)
- McNemar and Wilcoxon implementations
- Bonferroni corrections

**Cells 91-100**: Summary tables and exports
- Creates Excel summary tables
- Exports to `out/tables/`

**Cells 101-119**: Summary of corrections
- Documentation of fixes

**Status**: ✅ Keep if you want the full implementation, but Cells 147-154 are simpler

---

### ⚠️ **CAN DELETE** - Deprecated/Exploratory Sections

#### **Cells 7-49: Old Helper Functions**
- **What**: Original analysis functions with statistical issues
- **Status**: ⚠️ **DEPRECATED** - Do not use
- **Safe to delete?**: Yes, but keep for reference if needed

#### **Cells 120-143: OLD/EXPLORATORY ANALYSES**
- **What**: Preliminary analyses before corrections
- Various experiments with:
  - Gender-masculinity correlations (raw, not categorical)
  - OLS regressions on aggregated data
  - Distribution plots
  - Old statistical tests (wrong for paired design)
- **Status**: ⚠️ **DO NOT USE FOR PAPER**
- **Safe to delete?**: Yes - these have been superseded

**Cell 144**: `raise Exception("stop")` - **Marker separating old from new**

---

## 📊 Output Files (if using Cells 50-119)

If you ran the original corrected analyses (Cells 50-119), check:

```
artifacts_and_analysis/out/
├── tables/
│   ├── expert_summary_corrected.xlsx    ← Expert results table
│   ├── gender_summary_corrected.xlsx    ← Gender results table
├── data/
│   ├── expert_claim_level_aggregated.csv
│   ├── gender_debates_filtered.csv
└── figures/ (to be created)
```

---

## 🎯 Recommended Workflow

### For Your Paper (Quick Path): ⚡ **NO Preprocessing Needed!**

1. **Run Setup**: Cells 0-6 (load data)
2. **Run Expert Analysis**: Cells 147-149
3. **Run Gender Analysis**: Cells 150-154  
4. **Extract Results**: Copy printed statistics from output
5. **Create Figures**: (Next step - pending)

**Estimated time**: 2-5 minutes

**✅ Good News**: You can skip ALL preprocessing sections between Cell 6 and Cell 147!
- No need for data cleaning
- No need for masculinity merging (unless doing gender × masculinity)
- Cells 147-154 work directly with the raw loaded data

**Note**: Cells 145-146 may be empty - skip them and start at Cell 147

### For Complete Analysis (Full Path with Original Implementation):

1. **Run Setup**: Cells 0-6
2. **Run Preprocessing** (if using original Cells 50-119):
   - Domain masculinity merge (needed for original gender × masculinity)
   - Data cleaning steps
3. **Run Corrected Functions**: Cells 50-55
4. **Run Expert Analysis**: Cells 56-70
5. **Run Gender Analysis**: Cells 71-90
6. **Generate Tables**: Cells 91-100
7. **Review Summary**: Cells 101-119

**Estimated time**: 10-15 minutes

**Note**: Original implementation (Cells 50-119) may have more preprocessing requirements than simplified version (Cells 147-154)

---

## 📝 Key Statistical Decisions (Final Implementation)

### Expert Experiment
- **Design**: NOT fully paired - expert has k=3 per claim, baseline has k=1
- **Aggregation**: Use **MEAN** for all variables (including binary!)
  - **Why mean for binary?** Majority vote creates upward bias when comparing k=3 vs. k=1
  - Example: 2/3 convinced = 0.67 (NOT 1.0)
- **Tests**: Mann-Whitney U for all comparisons (treats aggregated means as continuous)
- **Primary comparison**: Any-expert vs. No-gender baseline
- **Multiple comparisons**: None (only 1 primary test per experiment)
- **Effect sizes**: Cohen's d (rounds), proportion differences (conviction)

### Gender Experiment
- **Design**: Fully paired (same 200 claims across all 6 conditions)
- **Primary comparison**: Female debaters (F_F+M_F) vs. Male debaters (M_M+F_M)
- **Tests**: 
  - McNemar's test for conviction (paired binary)
  - Wilcoxon signed-rank for rounds (paired continuous)
- **Multiple comparisons**: None (only 1 primary comparison)
- **Aggregation**: Combine F_F+M_F using MAX for conviction, MEAN for rounds

---

## 🔍 Critical Corrections Applied

### ✅ 1. Expert Majority Vote Bias Fixed
**Problem**: Using majority vote (2/3 → 1.0) inflates expert rates
**Solution**: Use mean (2/3 → 0.67) for fair comparison
**Location**: Cell 147

### ✅ 2. Gender Paired Design Recognized  
**Problem**: Original used Fisher's exact (assumes independence)
**Solution**: McNemar's test for paired binary data
**Location**: Cell 153

### ✅ 3. No Unnecessary Multiple Comparison Correction
**Problem**: Bonferroni too conservative for focused hypotheses
**Solution**: Test only primary comparisons (1-2 tests per experiment)
**Rationale**: Prespecified hypotheses don't need family-wise correction

### ✅ 4. Continuous Treatment of Aggregated Proportions
**Problem**: Can't use Fisher's exact on continuous proportions (0-1)
**Solution**: Mann-Whitney U for expert aggregated data
**Location**: Cell 148

---

## 📊 Results to Report in Paper

### Expert Findings (from Cell 148 output):

**Primary Test: Any-expert vs. No-gender**
- Conviction rate: Expert _% vs. Baseline _% (difference: +_%percentage points)
- Mann-Whitney U = _, p = _ 
- Rounds: Expert M=_ vs. Baseline M=_ (difference: _ rounds)
- Mann-Whitney U = _, p = _
- Cohen's d = _ (effect size for rounds)

**Key Interpretation**: 
- If conviction p < 0.05: "Experts achieved significantly higher conviction rates"
- If rounds p < 0.05: "Experts required significantly fewer rounds"
- If both ns: "No significant difference in final outcomes, but examine dynamics"

### Gender Findings (from Cells 153-154 output):

**Primary Test: Female vs. Male Debaters**
- Conviction: Female _% vs. Male _%
- McNemar's test: χ² = _, p = _
- Rounds: Female M=_ vs. Male M=_ (diff: _ rounds)
- Wilcoxon W = _, p = _

**Key Interpretation**:
- If p < 0.05: "Significant gender effect on [metric]"
- If p ≥ 0.05: "No significant main effect of debater gender"

---

## 🚫 What to Delete (Clean Up Notebook)

To streamline your notebook, you can **safely delete**:

### Delete Now (Won't Break Anything):
- **Cells 7-49**: Old helper functions (deprecated)
- **Cells 120-143**: Exploratory analyses with statistical issues
- **Cell 144**: The `raise Exception("stop")` marker

### Keep for Reference (Optional):
- **Cells 50-119**: Original corrected implementation
  - Only keep if you want backup/comparison
  - Delete if you're confident in Cells 145-155

### Never Delete:
- **Cells 0-6**: Setup (required)
- **Cells 147-154**: Final corrected analyses (your main results)

**After deletion, your notebook structure**:
- Cells 0-6: Setup ✅
- Cells 7-50: (Optional: Original corrected analyses)
- Cells 147-154: Final simplified analyses ✅

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Expert results don't match abstract numbers"
- **Check**: Which aggregation method was used (mean vs. majority vote)?
- **Solution**: Use Cell 147 with MEAN aggregation
- **Expected**: Slightly lower conviction rates than majority vote

### Issue 2: "Gender p-values are all > 0.05"
- **Possible**: No strong gender main effect (this is a valid finding!)
- **Next**: Examine gender × masculinity interaction (exploratory)
- **Report**: "No significant main effect of gender (p = X)"

### Issue 3: "Notebook cells out of order"
- **Solution**: Use "Cell → Run All Above" from Cell 147
- **Or**: Restart kernel and run Cells 0-6, then jump to 147-154

### Issue 4: "Do I need to run preprocessing cells?"
- **For Cells 147-154**: NO! Skip directly from Cell 6 to Cell 147
- **For gender × masculinity**: YES, need masculinity preprocessing
- **For original Cells 50-119**: Check if they reference `masculinity_scores_debates_df`

---

## 📚 Next Steps

### Still TODO:
1. ✅ Core statistical tests (DONE - Cells 145-155)
2. ⏳ Create publication figures (bar plots, effect size plots)
3. ⏳ Export summary tables to Excel
4. ⏳ Gender × Masculinity categorical analysis
5. ⏳ Create exploratory scatter plots

### For Figures:
- Figure 1: Expert conviction & rounds (bar plot)
- Figure 2: Dual-termination analysis (moderator vs. debater)
- Figure 3: Gender main effects (6-category bar plot)
- Figure 4: Gender × Masculinity (categorical, Low/Neutral/High)

---

## 📞 Questions?

**Where are my final results?**
→ Run Cells 147-154, copy the printed output

**Which cells should I include in supplementary materials?**
→ Cells 0-6 (setup) + Cells 147-154 (analyses)

**Can I delete the old analyses?**
→ Yes, delete Cells 7-49 and 120-143 safely

**How do I cite the statistical methods?**
→ "We used Mann-Whitney U tests for expert comparisons (after claim-level aggregation using means) and McNemar's test with Wilcoxon signed-rank for gender comparisons (paired design). No multiple comparison correction was applied as we tested prespecified primary hypotheses only."

---

**Last Updated**: After fixing majority vote bias and simplifying statistical approach
**Notebook Version**: Contains both original (Cells 50-119) and new (Cells 147-154) implementations
**Recommended**: Use Cells 147-154 for final paper

## 🔍 Quick Cell Finder

Can't find the cells? Here's how:
1. Open the notebook in VS Code/Cursor
2. Look at the top-left of each code/markdown block
3. You'll see "Cell X" in small gray text
4. Scroll to find:
   - **Cell 144**: `raise Exception("stop")` - marker before new analyses
   - **Cells 145-146**: Empty (skip these)
   - **Cell 147**: Expert aggregation code (starts with "# Step 1: Aggregate expert")
   - **Cell 148**: Expert statistical tests (starts with "# Step 2: Statistical tests")
   - **Cell 150**: Gender aggregation (starts with "# Step 1: Create debater gender")
   - **Cell 152-154**: Gender tests (McNemar, Wilcoxon)

## ⚡ Quick Start Cheat Sheet

**Absolute Minimum to Get Results:**
```
1. Open notebook
2. Run: Kernel → Restart Kernel
3. Run: Cells 0-6 (one by one or "Run All Above" from Cell 6)
4. Scroll to Cell 147
5. Run: Cells 147-154 (one by one or select all and run)
6. Done! Copy results from outputs
```

**Skip everything between Cell 6 and Cell 147** - it's old exploratory code and preprocessing you don't need for basic analyses.
