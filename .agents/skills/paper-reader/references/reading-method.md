# Reading method

Adapted from `paper-reading` by kingslayer-bot (MIT), which builds on S.
Keshav's three-pass method in "How to Read a Paper". The user's explicit goal,
structure or output request always takes priority over this file.

## Contents

- Three depths
- The four-question note structure
- Emphasis by paper type
- Targeted reading of a section, equation or figure
- Search and parallel work

## Three depths

Choose the depth from the request, or switch between depths as needed. A paper
does not have to go through all three.

### scan: quick judgment

Read the title, abstract, introduction, section headings and conclusion, and
skim the references. Answer the Five Cs in a few connected sentences:
category (what kind of paper), context (which work it builds on), correctness
(do the assumptions look plausible), contributions, and clarity.

End with a recommendation: read further, read background first, or set aside,
and why. A scan is a first impression, not a verification. It does not need
the four-question structure; a short note with these points is enough, saved
under the normal file rules when the user wants it kept.

### understand: the main thread (default)

Read the core content and the key figures carefully; postpone complex proofs
and implementation details when they are not needed for the argument. Build the
chain problem → method or argument → conclusion → evidence → limits, and note
related work that matters but was not read.

Write the note with the four-question structure below. Done means you could
explain to someone else how the authors answer their question and which
evidence supports which conclusion. Where something is not understood, name
the gap and say what background or closer reading would close it.

### reconstruct: rebuild the work

On top of `understand`, try to rebuild the key work from the authors'
premises, then compare with the paper: rebuild the derivation for a theory
paper, the design logic for a method paper, the validation path for an
experimental paper, the taxonomy or argument chain for a survey or position
paper.

Identify explicit and hidden assumptions, the key innovation, gaps in the
argument, the conditions under which it holds, and cases where it could fail.
Keep the four-question structure; put long derivations, reproduction steps or
special topics in the relevant section or an appendix. Done means you can
rebuild the overall logic and explain its strong and weak points; length is
not the measure.

## The four-question note structure

Start with the title and a short overview. Keep bibliographic details in the
metadata rather than repeating them. For `understand` and `reconstruct`, use
these four parts unless the user asks for another structure.

### 1. What question does the paper answer?

The core problem, the context, what earlier work got wrong or left open, and
the authors' central claim. State the assumptions needed to follow the
argument, and separate what the authors state from what you infer.

### 2. How do the authors reach the answer?

Follow the path of the work: key steps, mechanisms, premises and the reasons
for each choice. Describe design, data, method, procedure and analysis
together, focusing on how they connect, rather than as separate lists.

### 3. What is the answer, and how strong is the support?

Main findings, conclusions and contributions, each paired with its evidence
and location. Separate what the authors claim from what the evidence supports.
Keep comparison targets, conditions, units and reported uncertainty with key
numbers; do not treat error bars alone as proof of significance.

### 4. What are the limits, and what is the value for the reader?

Assessment of the method and argument, limitations, open questions, what can
be borrowed, and next directions. Separate limitations the authors admit from
questions you raise. Discuss relevance to the user's own project only when the
user has described it; otherwise give generally transferable ideas. Keep only
the concepts and references needed to understand or extend the paper.

## Emphasis by paper type

These are points to emphasize, not extra required sections. Leave out what
does not apply instead of writing "not applicable". Mixed papers combine rows.

| Type | Emphasis |
|---|---|
| Experimental, empirical | design, data or samples, controls, key results, bias and uncertainty |
| Theory | definitions, assumptions, propositions, proof idea, key derivations, conditions for validity; no experiments required |
| Method, system | motivation, mechanism, difference from prior approaches, evaluation, conditions for reproduction |
| Survey | scope, search and selection method (if reported), taxonomy, consensus, disputes and gaps; narrative vs systematic |
| Benchmark, evaluation | what is measured, metrics, data composition, fairness of comparison, leakage risk, scope |
| Position, concept | central claim, chain of reasoning, support, counterexamples, testability; do not invent experimental evidence |

## Targeted reading of a section, equation or figure

- The user can ask about one section, paragraph, equation or figure without a
  full reading first.
- Explain where it sits in the paper's logic, what it means, and how it
  connects to what comes before and after. Quote only short pieces of the
  original.
- Equations: use `$...$` or `$$...$$`, explain every symbol, the key steps and
  what the result means. Do not put equations in code blocks.
- Figures: check axes, units, conditions, comparisons and uncertainty. If the
  user excludes figures, say how that limits the judgment of evidence.
- Extracting every figure or building a full figure index is only done when
  asked.

## Search and parallel work

- Explaining the paper itself does not require a literature search. Priority
  questions, recent progress, external comparisons and later rebuttals do;
  mark what you could not verify.
- A literature survey can start from a few recent entry papers and expand
  through related work, repeated citations and important follow-ups. Do not
  call this exploratory process a systematic review.
- Split work across subagents only when the task is large enough (long
  derivations, cross-checking evidence, comparing several papers). When merging
  their results, keep meaningful disagreements and go back to the source.
