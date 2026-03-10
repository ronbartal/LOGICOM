# Article Draft Review - Main Fixes Needed

## Overview
This document identifies the main issues and fixes needed for the article draft. Issues are organized by section and priority.

## 📊 Status Summary

### ✅ Completed (Fixed)
- **Typos Fixed**: 
  - Line 203: "what constrains" → "which constrains" ✓
  - Line 203: "for boat" → "for both" ✓
- **Methodology Section**: All checklist items complete ✓
- **Results Section**: All checklist items complete ✓
- **Discussion Section**: All checklist items complete ✓, All HOLD items fixed ✓
- **Gender Notation**: All using M→F consistently ✓
- **Legacy-Baseline**: Clearly explained in Expert methodology ✓

### ⏸️ On Hold (User Requested Only Typos)
- **Dual-Termination Protocol** (Line 66): Grammar typos fixed ✓, Content left as-is per user preference ✓

### 📝 Methodology Status
**All items complete:**
- ✓ Expert experiment: k=3, 102 claims, aggregation method clear
- ✓ Gender experiment: k=1, 200 claims, 5 combinations, aggregation method clear  
- ✓ Dual-termination protocol is clearly explained
- ✓ Model architecture details are complete

**Nothing left to complete in Methodology section.**

---

## 🔴 HIGH PRIORITY - Critical Issues

### 1. Gender Combinations Count - Clarification Needed (Line 219)
**Issue**: States "200 claims × (4 gender combinations + 1 baseline) = 1,000 total debates"
**User Note**: This is intentional - gender analysis used only 5 combinations (4 gender + Persona-baseline). Legacy-baseline was used in expert experiment but not in gender analysis.
**Location**: Limitations section, Sample Size paragraph
**Fix**: Keep as is, but ensure Legacy-baseline is clearly explained in Expert experiment methodology section
**status** [v] - Intentional, but needs better explanation in Expert methodology

### 2. Legacy-Baseline Explanation in Expert Methodology
**Issue**: Legacy-baseline needs better explanation in Expert experiment methodology section
**User Note**: Legacy-baseline was used in expert experiment but not in gender analysis (intentional - gender analysis used only Persona-baseline)
**Location**: Methodology section, Expert experiment subsection
**Fix**: Add clear explanation of Legacy-baseline in Expert experiment methodology, explaining it's the original LOGICOM baseline with no names, no gender labels
**status** [v] - Legacy-Baseline clearly explained in line 92: "The Legacy-Baseline control condition employed the original LOGICOM persuader prompt ("You are a professional persuader...") with no expert credentials, no specialized language, no authority markers, and no names or gender labels"

### 3. Inconsistent Gender Notation (Line 165)
**Issue**: Still uses `M\_F` instead of `M→F` notation
**Location**: Results section, Gender × Domain Masculinity Interaction paragraph
**Fix**: Replace all instances of `M\_F`, `F\_F`, `M\_M`, `F\_M` with arrow notation
**status** [v] - All gender notation uses arrows (M→F) consistently throughout the document

### 4. Grammar and Typos
**Issues**:
- Line 84: Dual-termination protocol description needs better formatting  [h] - **HOLD**: User requested only typos, not formatting fixes
- Line 168: "debaters own metrics" → "debater's own metrics" or "debater metrics"  [v] - Not found in current document
- Line 203: "what constrains" → "which constrains"  [v] - **FIXED**
- Line 203: "for boat" → "for both"  [v] - **FIXED**
- Line 196: "persona related" → "persona-related"  [v] - Already correct as "persona-related"
- Line 212: "Lightweight/outdated model, SOTA" → user wants general notion, suggests "advanced model" as alternative  [v] - Already using "advanced model"

---

## 🟡 MEDIUM PRIORITY - Structure & Consistency

### 5. Limitations Section Formatting (Lines 207, 209, 211)
**Issue**: Three items use `\textbf{}` instead of `\paragraph{}` formatting
**Location**: Limitations section
**Fix**: Convert to proper paragraph format:
- Line 207: `\paragraph{Model-Specific Effects.}`
- Line 209: `\paragraph{Domain-Level Masculinity Scoring.}`
- Line 211: `\paragraph{Prompt Engineering \& Argument Quality.}`
**status** [v] - **COMPLETED**: All formatting fixes applied

### 6. Incomplete Future Research Paragraph (Line 214)
**Issue**: Sentence is incomplete/awkwardly structured
**Location**: Limitations section, Additional Future Research paragraph
**Fix**: Complete the sentence structure - fixed by adding "Additional future research" and proper comma before "which would provide"
**status** [v] - **COMPLETED**: Sentence structure fixed

### 7. Dual-Termination Protocol Description (Line 66)
**Issue**: Grammar typos and optional content enhancement
**Location**: Methodology section, Dual-Termination Protocol subsection
**Fix**: 
- Grammar typos fixed: "even though is" → "even though it is", "hold you" → "hold your"
- Content enhancement: User chose to leave current version (without conviction rating detail) - **DONE**
**status** [v] - **COMPLETED**: Grammar typos fixed, content left as-is per user preference

---

## 🟢 LOW PRIORITY - Style & Polish

### 8. Language Refinements
**Issues**:
- Line 168: "debaters own metrics" → "debater metrics" or "debater's own metrics" [f]
- Line 196: "We reveal" → "We demonstrate" (less bombastic) [v] - **FIXED**
- Line 199: "meta-cognition" → "metacognition" (one word) [v] - **FIXED**
- Line 219: Check expert experiment description - clarify "306 debates" (is this per condition or total?) [h]

### 9. Consistency Checks
- [f] Verify all gender notation uses arrows (M→F) consistently 
- [f] Check all statistical test names are consistent (Wilcoxon signed-rank test)
- [f] Ensure all figure/table references use `\cref{}`
- [h] Verify all citations are properly formatted

### 10. Missing Elements
- [h] Check if Conclusion section is needed (currently ends with Future Research)
- [f] Verify all figures are properly referenced
- [f] Check table formatting consistency

---

## 📋 Section-by-Section Checklist

### Abstract
- [v] All statistics match Results section (77.5% vs. 74.5%, 5.31 vs. 6.25 rounds, 91.8% vs. 82.4%)
- [v] Keywords are appropriate
- [v] Abstract is now one paragraph (condensed from 3 paragraphs)
- [v] needs to be one paragraph max

### Introduction
- [v] Research question is clear (Line 25: "To what extent is an LLM's willingness to concede an argument determined by the persona of itself and its opponent?" - bolded and explicit)
- [v] Contribution is well-articulated (dual-termination protocol, dyadic interactions, two experimental axes, Bias of the Judge investigation)

### Background
- [v] All citations are present:
  - Gender/Authority biases: \citep{kotek2023gender, gallegos2023bias, santurkar2023whose}
  - LOGICOM framework: \citep{payandeh2023logicom, siegel2024agreeable}
  - LogicBench: \citep{parmar2024logicbench}
  - Sycophancy/FlipFlop: \citep{laban2023flipflop, sharma2023sycophancy}
  - ELM: \citep{petty1986elaboration}
  - Social simulacra: \citep{park2022social}
  - Debate frameworks: \citep{chan2024chateval}
- [v] Theoretical framework is clear:
  - ELM (Elaboration Likelihood Model) clearly explained
  - Central route (System 2) vs. Peripheral route (System 1)
  - Social simulacra concept
  - Gap in existing research (message vs. messenger) clearly articulated

### Methodology
- [v] Expert experiment: k=3, 102 claims, aggregation method clear (line 92: "For each of the 102 expert-matched claims, we ran k=3 repetitions... yielding 306 total debates. These were aggregated to claim-level (N=102) by taking the mean across repetitions")
- [v] Gender experiment: k=1, 200 claims, 5 combinations, aggregation method clear (line 97: "four gender combinations and a baseline condition" = 5 total; line 203: "200 claims × (4 gender combinations + 1 baseline) = 1,000 total debates"; aggregation method explained in line 203)
- [v] Dual-termination protocol is clearly explained (lines 65-74: full explanation with enumeration)
- [v] Model architecture details are complete (line 77: GPT-3.5-turbo for agents, GPT-4o for moderators, temperature settings)

### Results
- [v] All statistics are accurate (verified against Analysis Summary)
- [v] Figures and tables are properly referenced (using \cref{} for fig:expert_rounds_dist, fig:mf_masculinity, tab:mf_correlations)
- [v] Gender notation is consistent (M→F) throughout
- [v] Summary of Key Findings is comprehensive (lines 188-190)

### Discussion
- [v] Theoretical implications are clear (ELM, dual-process theory, social simulacra)
- [v] Connection to previous work is well-integrated (Siegel et al., Gal & Jonny's work)
- [v] Limitations are honest but not overly defensive
- [v] Future directions are concrete (cross-architecture, extended rounds, replication, mechanistic understanding, intervention studies, other social dimensions)

### Limitations
- [v] Sample size calculations are correct (line 203: expert k=3, 306 debates aggregated to N=102; gender k=1, 1,000 debates)
- [v] All limitations are properly formatted as paragraphs (lines 207, 209, 211 now use \paragraph{} format) - **FIXED**
- [v] Future research is complete and well-structured (line 214 sentence structure fixed) - **FIXED**

---

## 🔧 Quick Fixes Summary

1. **Line 165**: Replace `M\_F` with `M→F` [v] - Already using arrows
2. **Line 168**: Fix "debaters own metrics" → "debater metrics" [v] - Not found in document
3. **Line 196**: Fix "persona related" → "persona-related" [v] - Already correct
4. **Line 212**: Replace "Lightweight/outdated model, SOTA" with specific models [v] - Already using "advanced model"
5. **Line 203**: Fix "what" → "which" [v] - **FIXED**
6. **Line 203**: Fix "for boat" → "for both" [v] - **FIXED**
7. **Line 203**: Gender count is correct (5 combinations = 1,000 debates) - intentional [v]
8. **Line 207, 209, 211**: Convert `\textbf{}` to `\paragraph{}` format [v] - **FIXED**
9. **Line 214**: Complete Future Research paragraph [v] - **FIXED**
10. **Expert Methodology**: Better explain Legacy-baseline in Expert experiment section [v] - Clearly explained in line 92

---

## 📝 Notes

- The article structure is generally good
- Main issues are: (1) gender count error, (2) formatting inconsistencies, (3) some grammar/typos
- Most content is solid, needs polish rather than major restructuring
- Consider adding a brief Conclusion section if journal requires it
