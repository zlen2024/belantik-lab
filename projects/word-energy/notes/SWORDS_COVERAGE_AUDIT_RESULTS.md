# Label-blind SWORDS development input coverage

21 November 2026. The [official SWORDS development data and repository](https://github.com/p-lambda/swords) provide contextual lexical-substitution candidates and report a CC-BY-3.0-US license. The source is described in [Lee et al. 2021](https://aclanthology.org/2021.naacl-main.345/). This audit downloaded only the v1.1 development file, SHA-256 `bc3570786455a05f11fcb4fc7f54e117642d5fb5e98f19355028c5c7ee00c954`, and inspected input/candidate strings for compatibility with the existing 2,000-word WikiText training vocabulary. It did not inspect candidate judgment labels or score any model.

| Coverage step | Number of targets |
|---|---:|
| Total | 370 |
| Single ASCII token with exact target span | 370 |
| Target appears in training vocabulary | 224 |
| Target and at least one adjacent in-vocabulary context token within ±2 positions | 223 |
| Above, with at least 1 / 2 / 5 / 10 distinct in-vocabulary single-token candidates, excluding the target itself | 223 / 219 / 193 / 126 |

The 2-candidate criterion is **availability**, not evidence that one candidate is appropriate and another inappropriate. A large majority of source candidates are not necessarily available in the small vocabulary. Each eligible example inherits the previous short context-window rule, and requiring the target itself in vocabulary is a design choice for a target-conditioned embedding comparison; another separately specified inference method may have different coverage. No performance has been measured on SWORDS.

An exact normalized substring screen found zero SWORDS development contexts of at least 80 characters embedded in the loaded WikiText train or validation paragraph strings. This incomplete screen cannot rule out shorter overlaps, paraphrases, memorized material, duplicated sources or pretraining elsewhere. Source/doc independence, labels, candidate sufficiency, scoring rule and source-cluster uncertainty remain to be designed before treating a benchmark as evidence of replacement performance. SWORDS development is now used for input coverage; its test split was not downloaded or examined.

In a separate posthoc label-blind audit, the eligible 219 targets map to 19 CoInCo source-file identifiers inferred from context metadata. One file contributes 132 of 219 targets, approximately 60%. The identifiers have not been independently reconciled against the source corpus. A future evaluation must account for this concentration and must not treat all 219 examples as independent documents. Ten source contexts shorter than 80 normalized characters were outside the long-substring overlap screen. Source concentration audit MLflow run `c630f871c2e241ff9b25dda630d06b5a`.

Critic also found that the dataset marks substitutes as lemmatized. Among 219 eligible targets, 63 surface forms differ from the provided target lemma, and 3 candidate sets include the target lemma despite excluding the surface form. Excluding those lemmas would still leave at least two candidates for those three cases. This coverage audit follows its stated surface-token rule; any later scorer and evaluation must freeze lemmatization, self-substitution and token mapping explicitly.

Audit protocol and code were frozen at commit `d33897f`. MLflow `word-energy-natural-context-evaluation`, run `60dbe81ed2dd4d2ebe2f3cd6dbf6607a`, contains the protocol, source snapshot and aggregate `coverage.json`. Raw downloaded data are local and Git-ignored.
