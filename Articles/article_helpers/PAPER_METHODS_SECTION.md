# Methods Section for Expert and Gender Persona Experiments

## 1. Baseline LOGICOM System

### 1.1 Original Framework (Articles\LOGICOM Article.pdf, Articles\LOGICOM__gal_jonny_copy.pdf, John-Isr/LOGICOM)

- **Multi-Agent Debate Architecture**: Two-agent system with persuader attempting to convince debater of a claim
- **Moderation System**: Five independent moderator agents evaluating debate progress:
  - Termination moderator: Assesses if debate should continue
  - Topic moderator: Checks if discussion remains on-topic
  - Conviction moderator: Detects if debater is convinced (with 1-10 rating)
  - Argument quality moderator: Rates persuasiveness of arguments (1-10 scale)
  - Debate quality moderator: Overall debate quality assessment
- **Claims Dataset**: 200 debate claims from Kialo platform with structured format (claim, topic, reason)
- **Memory Management**: ChatSummaryMemory with summarization at 4000 tokens, maintaining last 4 messages + summary
- **Turn-based Protocol**: Alternating turns between persuader and debater, max 10 rounds per debate
- **LLM Backend**: GPT-3.5-turbo for agents, GPT-4o for moderators
- **Output Logging**: JSONL debate transcripts + Excel summary with conviction outcomes and quality metrics

---

## 2. Novel Contributions of Current Work

### 2.1 Persona Injection on Two Axes

**Expert Axis** (Authority Bias):
- **Implementation**: Prepend expert persona text to persuader prompt wrapper on every turn
- **Three Expert Types**: Education (Distinguished Professor), Diplomat (Senior Diplomat), Health (Chief Medical Officer)
- **Credentials**: 25-30 years experience, 100+ publications, board certifications, field-specific expertise
- **Argumentation Style**: Reference field-specific theories (constructivism, realpolitik, clinical consensus), cite empirical research, dismiss anecdotes
- **Code Implementation**: 27-line `ExpertPersuaderAgent` wrapper class that modifies prompt wrapper before parent initialization
- **Gender-Neutral Design**: No personal names, no gendered pronouns, title-based only (isolates authority from gender)

**Gender Axis** (Gender Bias):
- **Implementation**: Natural name-based personas with explicit age and gender labels in system prompts
- **Gender Combinations**: Six conditions tested systematically:
  - M_M (Josh-male / Mike-male)
  - M_F (Josh-male / Laura-female)
  - F_M (Karen-female / Mike-male)
  - F_F (Karen-female / Laura-female)
  - Persona-baseline (Casey-neutral / Riley-neutral)
  - Legacy-baseline (no names/labels)
- **Design Rationale**: Natural names isolate gender bias from "generic naming bias"; neutral baseline controls for naming effects
- **Prompt Structure**: "You are [Name], a 31-years-old [gender label] professional persuader..."
- **Code Implementation**: Dual prompt system with gender-aware variants; `select_prompts_by_gender()` switches between legacy and gender-aware templates

### 2.2 Dual Termination Protocol

**Innovation**: Conviction detected through **two independent signals**, both required for termination:

1. **Debater Self-Admission**: 
   - Secret signal `[#$#]` appended to debater's message when convinced
   - Instruction embedded in debater system prompt: "If opponent's arguments are logically irrefutable and you decide to change your opinion, you MUST append the code [#$#] at the very end"
   - Prevents false negatives (moderator missing subtle conviction)

2. **Moderator Assessment**:
   - Conviction moderator evaluates debater's message content independently
   - Returns `<CONVINCED>` or `<NOT-CONVINCED>` with 1-10 conviction rating
   - Prevents false positives (debater accidentally using signal)

**Protocol**: Debate terminates as "convinced" only when **both** signals align in the same round
- Logged separately: `debater_self_admission_round` vs. `moderator_conviction_round`
- Agreement rate >80% validates dual-signal reliability
- Disagreement cases marked as inconclusive (result=2)

### 2.3 LLM Architecture

**Agent Models**:
- **Persuader & Debater**: GPT-3.5-turbo (16k context window)
  - Temperature: 0.7 (moderate creativity)
  - Max tokens: 500 per turn
  - Reasoning: Cost-effective for high-volume debates, sufficient for argumentative tasks

**Moderator Models**:
- **All 5 Moderators**: GPT-4o (128k context window)
  - Temperature: 0.3 (more deterministic for evaluation)
  - Max tokens: 100-200 (concise judgments)
  - Reasoning: Superior reasoning for meta-evaluation tasks, higher consistency

**Domain/Masculinity Classifiers**:
- **Model**: GPT-4o
- **Temperature**: 0.1 (maximally deterministic)
- **No Memory**: Single-turn classification
- **Reasoning**: One-time offline task requiring high accuracy

---

## 3. Expert Persona Experiment

### 3.1 Domain Extraction & Selection

**Domain Classification Process**:
1. **Automated Extraction**: LLM-based `DomainClassifierAgent` analyzes each of 200 claims
2. **Prompt**: "Identify the primary domain or topic area in 1-4 words" (e.g., "Higher Education", "International Conflict")
3. **Model**: GPT-4o with temperature=0.1 for consistency
4. **Output**: `claims_classified.csv` with domain labels for all claims

**Expert Domain Mapping**:
- Manual categorization of domains into three expert categories:
  - **E (Education)**: Higher Education, Campus Policy, Educational Reform, Academic Freedom
  - **I (International/Diplomat)**: International Conflict, Foreign Policy, Geopolitics, Treaty Negotiation
  - **H (Health)**: Healthcare Policy, Public Health, Medical Ethics, Pandemic Response

**Domain Coverage Statistics**:
- **Total claims**: 200
- **Top 3 expert domains**: 102 claims (51%)
  - Education (E): 48 claims (24%)
  - International (I): 30 claims (15%)
  - Health (H): 24 claims (12%)
- **Remaining domains**: 98 claims (49%) excluded (general policy, economics, sports, etc.)

### 3.2 Experiment Overview

**Experimental Design**:
- **Independent Variable**: Persuader persona type (3 expert types vs. control)
- **Dependent Variables**: 
  - Primary: Conviction rate (proportion of debates ending with result=1) and rounds to conviction
  - Secondary: Argument quality ratings (1-10), debate quality ratings (1-10)
  - Agreement metrics: Debater self-admission round, debater convinded rate, moderator conviction round and moderator convinced rate

- **Control Variables**:
  - Debater: Standard baseline (no expert credentials, no gender)
  - Moderators: 5 standard moderators (unchanged)
  - Claims: Pre-filtered to expert domains only (E, I, H)
  - Repetitions: k=3 debates per claim per condition

**Procedure** (per EXPERT_PERSONA_METHODS.md):
1. **Load Claims**: Filter `claims_with_experts.csv` for `expert_domain ∈ {E, I, H}`
2. **For Each Claim**:
   - Map domain code to expert type (E→Education, I→Diplomat, H→Health)
   - Load expert-specific persona and initial prompt files from `prompts/persuader/expert_[type]_persona.txt`
   - Prepend expert persona to persuader wrapper: `f"{expert_persona}\n\n{base_wrapper}"`
   - Run k=3 debates with expert persuader vs. standard debater
3. **Debate Execution**: Standard `DebateOrchestrator` runs up to 10 rounds, applies dual termination protocol
4. **Logging**: Save to `expert_debates/{topic_id}/Expert-{Type}/{chat_id}/` + append to `expert_debates_summary.xlsx`

**Expert Persona Content** (see EXPERT_PERSONA_METHODS.md lines 82-102):
- **Education Expert**: Distinguished Professor, 30 years research, 100+ papers, pedagogy expertise. Cites constructivism, cognitive load theory, zone of proximal development.
- **Diplomat Expert**: Senior Diplomat, 25 years in conflict zones, treaty negotiator. References realpolitik, balance of power, IR theory.
- **Health Expert**: Chief Medical Officer, Board-Certified Epidemiologist, Lancet/JAMA publications. Emphasizes RCTs, meta-analyses, clinical consensus.

### 3.3 Control Condition

**Baseline**: Legacy LOGICOM persuader with **no expert credentials**
- **Prompt**: "You are a professional persuader participating in a conversational debate..."
- **No Specialized Language**: Generic argumentation, no field-specific theory references
- **No Authority Markers**: No titles, years of experience, or publication claims
- **Same Infrastructure**: Identical debater, moderators, orchestration, memory system

**Comparison Validity**:
- **Single Variable Isolation**: Only persuader wrapper differs; all other components identical
- **Backward Compatible**: Uses same `_run_single_debate()` function as baseline experiments
- **Same Dataset**: Control debates run separately on same claims (or historical baseline from prior runs)

---

## 4. Gender Persona Experiment

### 4.1 Detailed Overview (per GENDER_PERSONA_EXPERIMENT.md)

**Experimental Design**:
- **Independent Variables**: 
  - Persuader gender (M/F/N/None)
  - Debater gender (M/F/N/None)
  - Topic domain masculinity score (1-10)
- **Dependent Variables**:
  - Primary: Conviction rate by gender combination, rounds to conviction
  - Secondary: Argument quality ratings (1-10), debate quality ratings (1-10)
  - Agreement metrics: Debater self-admission round vs. moderator conviction round
  - Intersectional: Gender effect × masculinity interaction

**Six Gender Combinations**:
1. **M_M**: Josh (male persuader) vs. Mike (male debater)
2. **M_F**: Josh (male persuader) vs. Laura (female debater)
3. **F_M**: Karen (female persuader) vs. Mike (male debater)
4. **F_F**: Karen (female persuader) vs. Laura (female debater)
5. **Persona-baseline**: Casey (neutral) vs. Riley (neutral) — no gender label, neutral names only
6. **Legacy-baseline**: No names, no gender labels (original LOGICOM baseline)

**Procedure**:
1. **Load Claims**: All 200 claims from `claims/all-claim-not-claim.csv`
2. **For Each Claim**:
   - For each of 6 gender combinations:
     - Select gender-aware or legacy prompts via `select_prompts_by_gender()`
     - Format prompts with name placeholders: `<PERSUADER_NAME_BY_GENDER>`, `<PERSUADER_GENDER_LABEL>`
     - Run k repetitions with that gender configuration
3. **Total Debates**: 6 combinations × 200 claims × k=1 repetition = 1200 debates
4. **Logging**: Save to `debates/{topic_id}/{gender_case}/{chat_id}/` + append to `all_debates_summary.xlsx`

**Gender Prompt Structure**:
- **System Instruction**: "You are [Name], a 31-years-old [gender label] professional persuader..."
  - Male: "You are Josh, a 31-years-old male professional persuader..."
  - Female: "You are Karen, a 31-years-old female professional persuader..."
  - Neutral: "You are Casey, a 31-years-old professional persuader..." (no gender label)
- **Initial Greeting**: "Hello my name is [Name]. The conversation is about..."
- **Gender Label Spacing**: " male", " female" (leading space for grammatical correctness)

### 4.2 Control Design: Natural Names vs. Generic Names

**Design Rationale**:

**Why Natural Names?**
- **Isolate Pure Gender Bias**: Using common, natural names (Josh, Karen, Mike, Laura) prevents confounding gender bias with "artificial naming bias"
- **Ecological Validity**: Reflects real-world AI assistants (Alexa, Siri) that use natural names
- **Avoid Cognitive Distance**: Generic labels ("Agent-M1", "Persuader-Female") create impersonal perception, affecting persuasion independently of gender

**Two-Baseline Strategy**:
1. **Persona-baseline (Neutral Names)**: Casey/Riley with no gender labels
   - **Purpose**: Controls for *having a name* vs. not having one
   - **Tests**: Does naming alone affect persuasion? (Persona-baseline vs. Legacy-baseline)
2. **Legacy-baseline**: No names, no gender, no age
   - **Purpose**: Pure baseline from original LOGICOM
   - **Tests**: Combined effect of naming + gender (M_M/F_F vs. Legacy-baseline)


### 4.3 Masculinity Domain Scoring

**Research Question**: Does gender bias amplify in stereotypically masculine domains (military, sports) vs. feminine domains (childcare, relationships)?

**Methodology Evolution**:

**Initial Approach: Claim-Level Scoring (ABANDONED)**:
- **Method**: Score each of 200 claims individually for masculinity (1-10)
- **Problem**: Claims are complex, multi-faceted statements (e.g., "Foreign language classes should be mandatory in college")
  - Contains education policy (neutral), globalization (slightly masculine), mandatory requirements (authority, masculine)
  - LLM struggled with conflicting dimensions
- **Result**: Over-centered distribution (almost all claims classified as 5)
  - Insufficient variance for meaningful analysis
  - Unable to distinguish domains reliably
- **Decision**: **Abandoned** due to dimensionality complexity

**Final Approach: Domain-Level Scoring (ADOPTED)**:
- **Method**: Leverage domain classification from expert experiment (Section 3.1) and score masculinity at domain level
  - Step 1: Use existing domain labels extracted by `DomainClassifierAgent` from expert experiment (e.g., "Higher Education", "International Conflict")
  - Step 2: Score domain masculinity using `MasculinityClassifierAgent` with stereotypical associations
  - Step 3: Assign domain score to all claims in that domain
- **Advantages**:
  - Clearer, more interpretable categories
  - Better variance (domains range 2-10, not fully centered)
  - Consistent within-domain scoring
  - Aligns with psychological research on domain stereotypes
  - **Theoretical Basis**: Stereotypes operate via System 1 (automatic, immediate processing) rather than System 2 (deliberate reasoning). Domain-level associations capture these immediate, heuristic judgments more reliably than complex claim-level analysis

**Masculinity Scoring Scale** (per `prompts/classifier/masculinity_instruction.txt`):
- **HIGH (7-10)**: Military, warfare, competitive sports, physical strength, engineering, finance, politics, dominance, aggression, hierarchy
- **NEUTRAL (4-6)**: General policy, economics, education, technology, science, law, business
- **LOW (1-3)**: Childcare, nurturing, emotional wellbeing, social harmony, fashion, relationships, caregiving, cooperation

**Example Scores**:
| Domain | Score | Reasoning |
|--------|-------|-----------|
| Military Training | 10 | Physical dominance, aggression, combat (most masculine) |
| International Conflict | 8 | Military power, strategic dominance (highly masculine) |
| Higher Education | 4 | Neutral academic domain, slight feminine association (teaching) |
| Religion and LGBT Rights | 3 | Inclusion, protection of marginalized groups (less masculine) |
| Childcare Policy | 2 | Nurturing, caregiving (least masculine) |

**Critical Note**: Scores reflect **stereotypical cultural associations**, not normative judgments or objective domain characteristics. Designed to measure bias, not endorse stereotypes.

### 4.4 Correlation Analysis with Persuasion Metrics

**Analysis Strategy** (to be shown in Results section):

**Planned Analyses** (detailed results deferred to Results section):
- Main effect of gender on conviction rates across all gender combinations
- Masculinity × gender interaction effects (stratified by domain masculinity scores)
- Argument quality correlations with gender and masculinity
- Gender congruence effects (same-gender vs. mixed-gender pairings)

---

## Summary of Methodological Contributions

1. **Baseline LOGICOM**: Established multi-agent debate framework with dual termination, 5-moderator system, memory management
2. **Persona Injection**: Two orthogonal experimental axes (expert authority vs. gender) with minimal code modifications
3. **Expert Experiment**: 3 domain-matched expert personas (Education, Diplomat, Health) covering 51% of claims (102/200)
4. **Gender Experiment**: 6 systematic gender combinations with natural names to isolate pure gender bias from naming effects
5. **Masculinity Scoring**: Domain-level masculinity ratings (1-10) enable intersectional analysis of gender × topic interaction
6. **Dual Termination Protocol**: Debater self-admission + moderator assessment ensures reliable conviction detection

**Reproducibility**: All code available in GitHub repository, prompt files in `prompts/`, configuration in `config/settings.yaml` and `config/models.yaml`
