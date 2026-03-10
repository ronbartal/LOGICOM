# Discussion Chapter Skeleton

## Overview
**Structure: Two main subsections following Option A**
- **Section 1**: Theoretical Implications & Broader Significance (3 paragraphs)
- **Section 2**: Limitations & Future Directions (2 paragraphs)

**Note:** Synthesis of Key Findings stays in Results section (end of Results, as bridge to Discussion).

**Status Field Guide:**
- `[w]` = To write in the paper (main content)
- `[?]` = Question/uncertain if needed
- `[h]` = Hold - keep but don't write yet
- `[m]` = Move to another section
- `[app]` = Move to appendix
- `[ ]` = In progress / Complete

---

## Section 1: Theoretical Implications & Broader Significance
**Status:** `[w]` **Target: 3 paragraphs**

**Notes:**
- Connect findings to theoretical frameworks (ELM, Social Simulacra, Dual-Process)
- Include your ideas about continuing Gal & Jonny's work
- Include ground truth dilemma discussion
- This section establishes the contribution and theoretical significance

### Paragraph 1: Social Interpretation - ELM Framework & Dual-Process Theory
**Status:** `[w]`

**Key Points:**
- [ ] Expert personas → peripheral route / System 1 (authority cues, automatic)
- [ ] Moderator counter-balancing → central route / System 2 (analytical evaluation, deliberate)
- [ ] Gender stereotypes operate via System 1 (automatic), moderator uses System 2
- [ ] LLMs as social actors, not just reasoning systems
- [ ] Models simulate human social cognition
- [ ] Bias reproduction vs. bias simulation

**Your Ideas:**
<!-- Add your thoughts here -->


### Paragraph 2: Connection to Gal & Jonny's Work
**Status:** `[w]`

**Key Points:**
- [ ] They show progressive models less vulnerable to logical flaws (System 2)
- [ ] You suggest they also less vulnerable to social bias (System 1)
- [ ] Open question: Is this solely because progressive model is the judge? 
- [ ] Cross-architecture design needed: [(pers,deb),mod] × [gpt3.5,gpt4o]
- [ ] **Political bias critique response**: Jonny and Gal showed models are politically left-wing biased. Critique may raise: maybe stereotyped masculine domains are just more right-wing (e.g., war).
  - **Your answer**: From LOGICOM data design - any domain has two claims exactly opposite, so if a claim about war is right-wing, its negative is left-wing. For instance "Iran remains a threat" vs. "Iran doesn't remain a threat" in geopolitical aspect or "Fees are necessary" vs. "Fees should be more regulated" in economic aspect.

**Your Ideas:**
<!-- Add your thoughts here -->


### Paragraph 3: LLM as a Judge Dilemma - Internal vs. External Evaluation
**Status:** `[w]`

**Key Points:**
- [ ] **Ground truth question**: What is the ground truth in this study?
- [ ] **Internal (debater)**: 
  - May suffer from biased metacognition
  - May "cheat"; conflicting objectives (don't change opinion vs. admit if convinced)
- [ ] **External (moderator)**: 
  - May be biased too
  - Main objective is to understand agent's cognitive state - agent knows best
- [ ] Is bias moderation "affirmative action" or problematic?
- [ ] **Broader implications**: Can an LLM admit it needs more clarification on a prompt when its objective is to understand the user and give the best answer under uncertainty?
- [h] suggest the dual termination protocol as a general dual-evaluation method for LLMs on cognitive complex tasks 

**Your Ideas:**
<!-- Add your thoughts here -->


### Optional: Gender Role Congruity Theory
**Status:** `[h]` **Hold - keep but don't write yet**

**Key Points:**
- Brief mention - why M→F shows strongest effects (stereotype-congruent authority)
- May be integrated into Paragraph 1 if space allows

---

## Section 2: Limitations & Future Directions
**Status:** `[w]` **Target: 2 paragraphs**

**Notes:**
- Be honest but not overly defensive
- Balance limitations with strengths
- Keep concise - don't over-explain
- First paragraph: Limitations
- Second paragraph: Future Directions (forward-looking)

### Paragraph 1: Methodological Limitations
**Status:** `[w]`

**Key Points:**
- [ ] **Sample Size & Statistical Power**
  - "Compared to other studies that assess the performance of LLMs, our multi-agent debate framework requires significantly more computational resources, which becomes costly for models like GPT-4. This constraint limits our ability to perform a more iterative evaluation of the model's consistency on the same claim." (from LOGICOM article - merge this here)
  - Expert experiment: N=102 claims per condition
  - Gender experiment: N=200 claims, but aggregated to 9 masculinity bins
  - Correlation analysis: small effective N

- [ ] **Master propmt regarding limitations**
  - did agent persona affected their argument quality? is this make the bias justify? maybe from the debater perspective but it still shift persuader's behaviour without aditional power or knowlage
  - can lead to new reasarch direction: is it helpfull to give your agent a persona description
  - can test if this aspect really contaminated my expert experiment via future experiment: expert persuader force to introduce it-self (as now) vs. not (and maybe even told to remain modest about it) 
    - can compare all persuasive metrics to check debater bias, argument quality metrics to evaluate moderator bias  

- [ ] **Aggregation Robustness**: Masculinity-score-level aggregation
  - N=9 bins for M→F correlations (reduced effective sample size)
  - Potential masking of claim-specific effects
  - Tryed Alternative: claim-level analysis with masculinity as continuous predictor (N=200, more power) didn't show strong relation, very noisy. maybe work with much bigger sample.
  
- [ ] **Extended Round Limits**: Explain limitation of max_round=10 compared to past LOGICOM study (stricter conviction rule)
  - Current 10-round limit may truncate debates
  - More rounds = more expensive
  - Does moderator eventually converge with debater?
  
- [ ] **Model-Specific Effects**: GPT-3.5-turbo vs. GPT-4o differences
  - Debater/Persuader: GPT-3.5-turbo; Moderator: GPT-4o
  - May not generalize to other models
  
- [ ] **Masculinity Scoring**: Domain-level vs. claim-level
  - Rationale for domain-level approach (System 1 stereotypes)
  - But may miss nuanced claim-specific effects
  - Outcome still too centered (52% scored as 5)

**Hold (don't write yet):**
- [h] **Ecological Validity**: Name-based gender signaling
  - Implicit vs. explicit gender cues
  - Real-world conversations use names, but may not capture all mechanisms
  
**Your Ideas:**
<!-- Add your thoughts here -->


### Paragraph 2: Future Research Directions
**Status:** `[w]`

**Key Points:**
- [ ] **Cross-Architecture Design**: [(pers,deb),mod] × [gpt3.5,gpt4o] to isolate model effects
  - Test if moderator counter-balancing is model-specific or general
  - Address open question from Section 1, Paragraph 2
  
- [ ] **Extended Round Limits**: Test beyond 10 rounds
  - Does moderator eventually converge with debater?
  - Long-term debate dynamics
  
- [ ] **Replication & Generalization**: 
  - Other models (Claude, Gemini, open-source)
  - Larger samples
  - Different domains
  - Cross-cultural validation
  
- [ ] **Mechanistic Understanding**: 
  - Why do moderators counter-balance? (explicit vs. implicit)
  - Attention patterns in expert vs. baseline debates
  - Token-level analysis of persuasion mechanisms
  
- [ ] **Intervention Studies**: 
  - Can we train models to resist persona effects?
  - Explicit instruction to ignore social cues
  - Adversarial training against bias
  
- [ ] **Other Social Dimensions**: 
  - Race, ethnicity
  - Age
  - Socioeconomic status
  - Intersectionality (e.g., female expert on masculine topic)

**Your Ideas:**
<!-- Add your thoughts here -->


---

## Appendix: Methodological Details
**Status:** `[app]` **Optional - only if space allows**

**Notes:**
- Technical details that don't fit in main Discussion
- Can be brief or omitted if space is tight

**Content:**
- [ ] **Domain Classification**: Automated LLM-based classification reliability, potential misclassification, inter-rater reliability

**Your Ideas:**
<!-- Add your thoughts here -->


---

## Additional Notes & Ideas
<!-- Use this space for any other thoughts, connections, or ideas that don't fit neatly into the above sections -->

**Note:** The "Main Takeaway" content (Logical robustness ≠ social immunity) should be integrated into the Results section's "Summary of Key Findings" paragraph, not repeated in Discussion. Discussion focuses on interpretation and implications, not re-stating findings.


---

## References to Integrate
<!-- List any papers, concepts, or ideas you want to make sure are mentioned -->


---

## Writing Style Notes
- [ ] Academic but accessible
- [ ] Balance detail with readability
- [ ] Connect back to Introduction/Background
- [ ] Forward-looking but grounded in findings
- [ ] Honest about limitations without being overly defensive
- [ ] **Section 1 = 3 paragraphs, Section 2 = 2 paragraphs** (total ~5 paragraphs for Discussion)
