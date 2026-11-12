# Reading papers with evidence

## Retrieve the correct source

1. Identify title, authors, paper identifier/DOI, version and supplied URL or local file. If identity is ambiguous, resolve it before attributing claims. A user-supplied file is the source unless another version is requested.
2. Prefer matching LaTeX source or accessible HTML for equations and structure, alongside the rendered paper for figures. Verify version and rendering: source availability alone does not guarantee correct extraction, especially with custom macros.
3. If only PDF is available, use an available PDF reader/parser. Inspect rendered pages for decisive equations, tables and figures. Use OCR only where needed and flag uncertainty; do not install several parsers by default. Follow an available PDF skill for layout-sensitive inspection.
4. If access fails, state which material is accessible and what that limits. An abstract-only overview must not present full-paper conclusions.

## Build an understandable account

For a normal technical note, cover the problem, prior approach, core idea, method, essential equations, results and limitations. Adapt depth to the request instead of filling every heading mechanically.

- Attach a locator to each substantive paper-specific claim: section, equation, table, figure, or page plus paragraph when numbering is absent. In comparisons, identify the paper and version as well as the locator.
- For a key equation, preserve its notation, define each symbol, state assumptions and explain its computational meaning in plain language. Label algebraic derivations as your derivations and verify the steps.
- Distinguish what authors demonstrate from what they suggest. Identify missing ablations, scope restrictions and unsupported extrapolations without inventing criticisms.
- Explain figures using their axes, units, scale, legend, error bars and experimental setting. Do not infer exact numerical values from the shape of a plotted line. If digitization is requested, label estimates and their uncertainty.
- For reported results, retain dataset, split, metric direction, baseline, compute budget and uncertainty where available. Mark absent information as unreported. Avoid declaring a winner across incompatible protocols.
- Describe plausible leakage or confounds as questions unless evidence establishes the problem. Unexpectedly strong performance warrants investigation rather than automatic acceptance or rejection.

For a saved deep reading, include an evidence table when it improves traceability:

| Claim | Paper locator | Evidence or reasoning | Status / uncertainty |
| --- | --- | --- | --- |

Before delivery, trace the main conclusions back to inspected passages, check equation transcription and metric values, and remove or qualify unsupported claims. References listed in an unread bibliography are leads, not evidence you have inspected.

For a later Manim handoff, retain the verified equations, symbol definitions, source locators, and a clear label for any illustrative toy data. Visual metaphors must not be presented as empirical proof.
