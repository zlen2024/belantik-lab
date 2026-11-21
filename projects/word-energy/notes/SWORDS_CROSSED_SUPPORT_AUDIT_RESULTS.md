# SWORDS development: matched preference reversals

21 November 2026. This is a **posthoc data-support audit** using already inspected SWORDS development judgments and the existing 2,000-word filtered input manifest. No model was trained or scored, and the official test was not opened. It asks whether the exact candidate pair has opposite majority judgments across model-visible inputs. Context identity means the sorted in-vocabulary ±2 token-ID bag, not the full sentence or its meaning.

Within the previous 132 informative targets there are 2,862 strict acceptable-versus-non-acceptable candidate comparisons. After grouping identical target ID, context bag and candidate pair, all 2,862 cells remain distinct and none conflict.

| Matched pattern | Count | Scope |
|---|---:|---|
| Same target and candidate pair, different context bags, opposite preferred candidate | 26 | 6 target IDs, 19 visible context bags, 12 candidate pairs, 6 source-file proxies |
| Same context bag and candidate pair, different target IDs, opposite preference | 0 | No match under this exact rule |
| Full alternating 2×2 target/context checkerboard for one candidate pair | 0 | No match under this exact rule |

Of the 26 context-reversal matches, seven share a source-file proxy and 19 include more than one proxy. Source-file proxies do not prove independent semantic content. Several matches can share target, candidate or context IDs and must not be treated as 26 independent target types. The counts do not confirm that the truncated model-visible context contains the semantic cue that justifies each human label. The original sentence may contain additional information or annotations may be uncertain.

The absence of target reversals or checkerboards is specific to this already filtered subset, exact token-bag grouping and observed judgments. It does not establish that such natural-language examples do not exist. It does mean the current 132-example panel alone does not provide a full crossed checkerboard test under these rules. A fresh, source-aware data design is needed before using that task to train or claim broad generalization.

Independent critic found **zero different-target comparisons sharing the same visible context bag and candidate pair even before checking preference signs**. Thus the target-reversal zero reflects missing matched support rather than evidence that the model fails a target reversal. The 26 context-reversal matches derive from 17 distinct original record pairs involving 19 raw rows; nine of those rows come from the dominant source. Fifteen of the 26 matches include a candidate with exactly 50% TRUE votes. Under the frozen rule such a candidate is non-acceptable because the acceptable threshold is strictly greater than 50%. The phrase “ignore ties” in the protocol refers to candidates in the same binary class, not exclusion of 50–50 vote splits. This annotation ambiguity and concentration limit interpretation; no label threshold was changed after seeing counts.

The first execution stopped before starting a run because the script omitted the `MlflowClient` import; commit `40fa842` fixes it. Protocol `c8825a2` and the corrected code were committed before successful calculation. MLflow `word-energy-natural-context-evaluation`, run `74be8268c85643028f468705fe42cdad`, stores source hashes, aggregate audit and matched IDs/context hashes. Independent critic review is recorded separately.
