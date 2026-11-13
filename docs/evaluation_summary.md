# Evaluation findings

## My contribution

I own the evaluation layer that turns trajectory predictions into comparable,
auditable evidence:

- immediate content, location, operation and abstention metrics;
- persistence, durability, reversion and censoring over later states;
- paired comparisons with trajectory-clustered uncertainty;
- remaining-work and over-editing proxies;
- edge-case tests and the prediction contract used by teammate models.

This work is model-independent. It is ready to receive model outputs without
duplicating model loading, training or trajectory segmentation.

## At a glance

**What I did.** I built the project's evaluation and evidence layer: a model
can hand in predictions, and the same evaluator measures whether it predicted
the right state, the right edit, avoided unnecessary edits, reduced remaining
work, and predicted changes that persisted.

**How I did it.** I reconstructed sliding prediction tasks from complete
trajectories, kept target and future states out of model inputs, scored every
prediction at edit-block and full-file level, followed edits through later
states, and used paired trajectory-clustered bootstrap intervals. Executable
tests protect the definitions and the model-output contract.

**What came out.** The simple repeat-last-edit rule finds some local edit
signal, but it is a poor policy: compared with no-change it loses 3.64
percentage points of exact match, falsely edits 51.32% of true no-edit cases,
and increases pooled remaining line work by 33.55%. Only 34.24% of eligible
reference edit blocks persist through one follow-up state. A validity audit
shows that timing tolerance does not rescue repeat-last-edit, and even a
target-aware gate between the two baselines improves pooled rework by only
0.39%. The fixed 20-case audit supports text replay for every case, standalone
syntax validation for only three, and execution for none. Repeat rework remains
negative across all 16 sufficiently sized source-model strata. On the 14,456
tasks with enough follow-up, 72.39% of repeat-last-edit's immediate match credit
is unavailable after restricting the reference to edits that survive one
follow-up state.

**What this tells the team.** A useful next-edit model cannot be selected by
content overlap alone. It must jointly improve edit location and content while
controlling intervention and remaining work. These baselines and metrics now
form the acceptance test for real models.

**What this does not yet tell us.** No real model has been scored in this
workbench, so it does not claim that any proposed model succeeds. The rework
measure is an offline line-diff proxy, not semantic correctness, developer
utility, or time saved.

## Task-tracker completion audit

| Tracker item | Status | Evidence in this workbench |
|---|---|---|
| Metric comparison and failure cases | Complete | `metric_claim_specification.csv` records every metric's inputs, calculation, denominator, censoring rule, known failure case and research role |
| Supported versus unsupported conclusions | Complete | The same specification separates what each metric supports from claims requiring execution, developer judgment or a user study; the notebook and this summary apply those boundaries |
| RQ3: temporary or abandoned work | Complete for the baseline phase | `temporary_reward_audit.csv` compares immediate and stable credit on the same eligible tasks; `temporary_reward_cases.csv` supplies eight inspectable examples; two regression tests lock the denominator and censoring behavior |

“Complete” here means the baseline evaluation method and evidence are complete.
It does not mean that a real model has passed the evaluation; that requires a
contract-compliant prediction file from the model workstream.

## What the project is trying to establish

The project asks whether a model can anticipate a developer's future edit and
whether that predicted change remains useful later in the trajectory.

One number cannot answer that question. The evaluator separates five claims:

1. **Full-state accuracy:** Does the predicted file exactly equal the observed target?
2. **Edit accuracy:** Are the content, location and operation of the edit correct?
3. **Abstention:** Does the model avoid editing when the observed target does not change?
4. **Temporal stability:** Does the predicted change appear in later states and persist?
5. **Remaining work:** After applying the prediction, is the observed target closer or farther away?

These are behavioral measurements. They do not establish semantic correctness,
developer usefulness or actual time saved.

## Is the method correct and sufficient?

The current protocol is correct for its stated purpose: comparing full-state
predictions against observed next states under one transparent reference. Its
strong points are leakage-safe model inputs, a strict missing-state policy,
paired evaluation on the same tasks, trajectory-clustered uncertainty, explicit
coverage and censoring, and separate measurements for accuracy, intervention,
remaining work and persistence.

The method was not yet sufficient for future trained-model claims in four ways.
This revision closes those gaps:

| Risk | Previous limitation | Current treatment |
|---|---|---|
| Edit matching | Greedy pairing was not guaranteed to maximize total credit | Global one-to-one linear assignment; baseline results are unchanged |
| Train/test leakage | User metadata was dropped after loading | Metadata is retained and a deterministic user-grouped split is exported |
| Reference ambiguity | One observed next state was treated as the only timing view | Repeated-context and within-five-state sensitivity are reported |
| Aggregate hiding | Results were averaged across very different edit sizes | Joint F1 and rework are stratified by changed-line magnitude |

It is still not sufficient to claim semantic correctness or developer value.
The dataset supplies observed text states, not a complete executable project or
all valid alternative edits. Human or execution-based validation remains a
separate evidence layer.

## Evaluation population

| Measure | Count |
|---|---:|
| Raw completion groups | 5,831 |
| Recorded snapshots | 53,614 |
| Strict complete trajectories | 2,292 |
| Horizon-1 prediction tasks | 16,748 |
| Horizon-1 observed edit blocks | 21,913 |
| Prediction-level baseline rows | 33,496 |

The strict cohort requires at least three states and no missing state. A
drop-None cohort is retained only as a sensitivity check. Removing a missing
observation can make two non-adjacent states appear adjacent, so it is not the
primary cohort.

## Primary scorecard

| Baseline | Tasks | Coverage | Exact match | Joint edit F1 | False intervention | Pooled rework reduction | Stable F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No change | 16,748 | 100% | 7.05% | 0.000 | 0.00% | 0.00% | 0.000 |
| Repeat last edit | 16,748 | 100% | 3.42% | 0.154 | 51.32% | -33.55% | 0.073 |

No-change is the stronger full-state floor. Repeat-last-edit captures some edit
content and operations but rarely combines the right content with the right
location. It also changes 680 of the 1,325 observed no-edit targets.

This is the decision table. Higher is better for coverage, exact match, joint
edit F1, pooled rework reduction and stable F1; lower is better for false
intervention. The metrics are intentionally not collapsed into one weighted
score because they describe different failure modes and the project has not
defined a defensible cost trade-off between them. The later tables explain each
column and diagnose why a result moves.

## Evaluation validity findings

| Check | Result | Decision |
|---|---:|---|
| Raw current equals next state | 7.05% | Keep raw exact match as the strict endpoint |
| No line-content edit after normalization | 7.91% | Treat separately from raw exact match |
| Whitespace-only target difference | 0.86% | Not large enough to explain the baseline result |
| Tasks with a repeated current state | 20.94% | Do not use a random row split |
| Tasks with repeated `previous + current` input | 7.34% | Prior state removes much, but not all, ambiguity |
| Repeated full input contexts with multiple next targets | 33.67% | One-to-many reference ambiguity is real |
| Cross-user repeated full input contexts | 0.06% | Most repetition is user-specific; split by user |
| Repeat exact match at next state | 3.42% | Primary result |
| Repeat exact match within five states | 3.88% | Timing error is not the main failure |
| Perfect no-change/repeat gate: exact gain | +0.018 points | Gating alone cannot rescue repeat |
| Perfect no-change/repeat gate: pooled rework gain | +0.39% | A richer edit generator is required |

The oracle is deliberately allowed to see the target. It is an upper bound on
what can be achieved by choosing between the two heuristics, not a deployable
model. Its very small gain is the strongest evidence that the next step should
be a real model rather than another abstention classifier around repeat-last-edit.

Change size also matters. Repeat-last-edit reaches 0.297 joint F1 for 1–2 line
targets, then falls to 0.164 for 3–5 lines, 0.081 for 6–10 lines and 0.032 above
10 lines. Future model comparisons must show this breakdown so a score is not
driven only by small edits.

## Why no-change edit F1 is 0

Exact match and edit F1 answer different questions.

| Situation | Exact match | Edit F1 |
|---|---|---|
| Current file already equals the observed target | 1 | Not a positive edit case |
| Target contains an edit and prediction returns current code | 0 | 0 because every observed edit was missed |

For an edited target, let `P` be predicted edit blocks and `T` be observed edit
blocks. No-change has `P = 0` and `T > 0`. Precision has no denominator, recall
is 0, and F1 is reported as 0 because the predictor retrieves none of the real
edits.

For a no-edit target, `P = 0` and `T = 0`. There is no positive edit class, so
the evaluator does not give it a perfect edit F1. Exact match and false
intervention measure that case instead.

Concrete example:

```text
current:    return x
target:     return x + 1
no-change:  return x
```

The prediction preserves nearly the entire file, but it misses the one observed
edit. Its edit recall and F1 are therefore 0.

## Paired evidence on the same trajectories

Both baselines are scored on the same 16,748 tasks. Confidence intervals
resample all windows from one trajectory together instead of treating related
snapshots as independent.

| Metric | No change | Repeat last edit | Delta | 95% interval | Better direction |
|---|---:|---:|---:|---:|---|
| Exact match | 7.05% | 3.42% | -3.64 points | [-3.95, -3.33] | Higher |
| Whole-file line Jaccard | 43.87% | 38.42% | -5.45 points | [-5.72, -5.18] | Higher |
| Location + content F1 | 0.00% | 15.38% | +15.38 points | [14.54, 16.26] | Higher |
| False intervention | 0.00% | 51.32% | +51.32 points | [47.17, 55.81] | Lower |
| Pooled rework reduction | 0.00% | -33.55% | -33.55 points | [-34.80, -32.27] | Higher |
| Stable F1 | 0.00% | 7.33% | +7.33 points | [6.79, 7.88] | Higher |

Repeat-last-edit learns a weak local edit pattern, but it is not a stronger
overall policy. It trades away full-state accuracy, abstention reliability and
remaining work.

## Remaining work and over-editing

The offline replay uses line-level added-plus-deleted cost. A replacement counts
both removed and inserted lines.

```text
pooled rework reduction
  = (total required lines from current - total remaining lines after prediction)
    / total required lines from current
```

| Baseline | Changed targets | Required lines | Predicted lines | Remaining lines | Pooled reduction | Improved | Worsened | Unnecessary lines on no-edit targets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No change | 15,423 | 8.44 | 0.00 | 8.44 | 0.00% | 0.00% | 0.00% | 0.00 |
| Repeat last edit | 15,423 | 8.44 | 3.53 | 11.27 | -33.55% | 1.97% | 47.39% | 1.58 |

Across changed-target tasks, repeat-last-edit increases total remaining line work
from 130,184 to 173,858 lines. A negative value means the prediction moves the
file farther from the recorded target. The per-task mean reduction is -59.22%,
but the pooled -33.55% is the primary summary because it is less sensitive to
small one-line targets.

This proxy does not measure elapsed developer time. It says only whether the
recorded target becomes closer under a transparent line-diff rule.

## Persistence, durability and censoring

An observed edit is stable when its payload remains near its location for the
required consecutive states. The target is the first state in the count.

| Consecutive states | Eligible tasks | Censored tasks | Eligible blocks | Stable blocks | Stable reference rate | Repeat stable F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 14,456 | 2,292 | 19,323 | 6,617 | 34.24% | 0.073 |
| 3 | 12,658 | 4,090 | 17,086 | 4,228 | 24.75% | 0.055 |
| 5 | 10,016 | 6,732 | 13,777 | 2,520 | 18.29% | 0.038 |

The conclusion depends on the persistence rule, so every result includes its
threshold, eligible reference count and censored count. Too little follow-up is
censored rather than labeled unstable.

Strict one-step reference reversion is:

```text
98 / 13,484 = 0.73%
```

Another 1,939 edited tasks have no state after the target and are censored. This
is a reference-trajectory diagnostic, not a model score, and it does not detect
partial or semantic reversions.

## When immediate metrics reward temporary edits

This experiment compares conventional immediate edit matching with the frozen
two-state persistence rule on exactly the same tasks. A task is eligible only
when the target has one recorded follow-up state. The 2,292 tasks without that
follow-up are censored rather than called unstable.

| Baseline | Eligible tasks | Censored tasks | Immediate credit | Stable credit | Credit removed | Removed share | Immediate joint F1 | Stable F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No change | 14,456 | 2,292 | 0.0 | 0.0 | 0.0 | N/A | 0.000 | 0.000 |
| Repeat last edit | 14,456 | 2,292 | 2,322.9 | 641.4 | 1,681.5 | 72.39% | 0.154 | 0.073 |

The removed share is `(immediate matched credit - stable matched credit) /
immediate matched credit`. It means that 72.39% of repeat-last-edit's immediate
location-and-content credit is not available once reference blocks must remain
through the follow-up state. No-change has no positive edit credit, so its share
is undefined, not zero.

Full-file exact match is too sparse to estimate this effect on changed targets:
repeat-last-edit exactly matches only 2 changed targets in the eligible set, and
1 contains only transient reference edits. That 1/2 is an illustrative case,
not a population estimate. The credit-level comparison is the reportable result
because its denominator is 2,322.9 units of immediate matched credit across the
14,456 eligible tasks.

This supports a project decision: future models must report immediate joint F1
and stable F1 together. It does **not** show why an edit disappeared, whether a
developer rejected it, or whether persistent code is semantically correct.
Eight exported cases expose the exact diffs behind the aggregate result.

## Where repeat-last-edit fails

| Target edit | Tasks | Acts on task | Content F1 | Location F1 | Operation F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|
| Replace | 10,012 | 75.29% | 0.360 | 0.386 | 0.457 | 0.227 |
| Insert | 1,893 | 60.27% | 0.138 | 0.207 | 0.269 | 0.044 |
| Delete | 789 | 82.00% | 0.370 | 0.130 | 0.015 | 0.071 |
| Mixed | 2,729 | 80.65% | 0.219 | 0.173 | 0.321 | 0.077 |

Deletions show why content overlap is insufficient: content F1 is 0.370 while
operation F1 is only 0.015. The baseline often matches text without predicting
the correct editing action.

## Fixed case and reconstructability audit

The 20 cases were selected before this final labeling pass to cover duplicate
states, high-overlap non-matches, one-step reversions, apparent repeat gains and
repeat harms. A fixed taxonomy is now applied to every case.

| Label | Cases | Meaning |
|---|---:|---|
| Correct abstention | 4 | Recorded next state has no substantive edit |
| Partial edit | 2 | Some target content is recovered, but the change is incomplete |
| Wrong content | 2 | Unsupported or incorrect text is introduced |
| Wrong operation | 1 | Insert/delete/replace action is wrong |
| Over-editing | 4 | Prediction changes more than the target requires |
| Uncertain reference | 5 | Target is transient or weak evidence of durable intent |
| Formatting-only exclusion | 2 | Excluded from semantic interpretation |

These are structured first-pass labels, not inter-rater evidence. Claims about
label reliability require an independent second reviewer.

Reconstructability is deliberately conservative:

| Evidence level | Cases | Denominator | What can be claimed |
|---|---:|---:|---|
| Deterministic text replay | 20 | 20 | Recorded full states reproduce the diff |
| Language identified | 8 | 20 | Conservative Python or JavaScript hint |
| Standalone syntax pass | 3 | 20 | Three JavaScript targets pass Node syntax checking |
| Fragment or parse failure | 5 | 20 | Python AST failure cannot distinguish invalid code from a fragment |
| Execution ready | 0 | 20 | No case has a complete project, dependencies, tests and entrypoint |

Therefore syntax-pass rate is not a general model metric, and parse failure is
not counted as incorrect code. The current data supports text evaluation, not
functional correctness.

## Source-model heterogeneity

`source_model` identifies the model behind the original completion that began
the recorded trajectory. It is not the next-edit predictor being evaluated.
The audit includes a group in the descriptive comparison only when it has at
least 300 horizon-1 tasks, 40 trajectories and 30 users. Sixteen of twenty
groups meet all three thresholds.

Across those 16 groups:

- normalized no-edit rate ranges from 5.08% to 10.72%;
- median target change is three lines for 14 groups and four lines for two;
- repeat joint location-content F1 ranges from 0.118 to 0.185;
- repeat false intervention ranges from 35.77% to 67.50%;
- repeat pooled rework reduction is negative in every group, ranging from
  -38.21% to -29.43%.

This supports one limited conclusion: the repeat baseline's over-editing and
negative-rework failure is not confined to one large source group. The table is
unadjusted for user mix, language, task difficulty, time or trajectory length;
it is not a causal ranking or a significance test of the source models.

## Frozen protocol

Protocol version `1.0-phase1-freeze` was fixed before receiving real model
results. It freezes:

- strict complete trajectories and horizon 1 as the primary population;
- the user-grouped 80/10/10 split with seed 20261104;
- the six primary behavioral metrics plus task count and model identity;
- global one-to-one linear edit assignment;
- two-state persistence as primary, with three and five as sensitivity checks;
- trajectory-clustered bootstrap with 2,000 resamples and seed 2026;
- censoring, missing-output and prediction-contract rules.

Later changes require a protocol version bump and must be labeled secondary,
unless they repair a demonstrated implementation defect.

## Validation and team handoff

Twenty-three executable tests cover:

- target and future leakage;
- exact, partial, wrong-location and wrong-operation edits;
- no-change zero-division;
- false intervention;
- persistence, reversion and censoring;
- remaining-work improvement and oversized edits;
- paired trajectory-clustered comparison;
- missing, invalid, duplicate and unknown model outputs;
- globally optimal versus greedy edit assignment;
- deterministic user-disjoint splits;
- raw versus normalized no-edit target definitions;
- completeness of the fixed case audit;
- immediate-versus-stable credit and temporary-case selection;
- consistency of the frozen scorecard and prediction contract.

The balanced case audit contains 20 cases. Eighteen are retained and two
whitespace-only cases are excluded. Rebuilding automated evidence does not
overwrite review decisions.

Model owners receive only prediction-time fields:

```text
example_id, session_id, anchor, horizon, previous, current
```

They return:

```text
example_id, model_name, horizon, prediction, generation_status,
model_version, prompt_version
```

The evaluator rejects duplicate and unknown IDs. Failed or missing outputs stay
in the coverage denominator.

## Dependency boundary

The evaluation module itself is not blocked. Its baselines, definitions,
uncertainty estimates, persistence analysis, rework proxy, tests and model
contract can all be run independently from the files in this folder.

The following **claims** are blocked until another workstream produces an
input. They should not be presented as unfinished evaluation code:

| Claim we want to make | Missing upstream input | What becomes possible once it arrives |
|---|---|---|
| A real model beats the floors | A prediction file matching the shared contract from model orchestration | Run the same paired scorecard and confidence intervals |
| Performance differs by trajectory segment | Stable segment IDs and definitions from the segmentation/data workstream | Join segment labels and report stratified results with denominators |
| Fine-tuning or a persistence objective helps | Predictions from the trained checkpoint and its comparison checkpoint | Attribute a paired change to the training intervention |
| Predictions are functionally correct | Reconstructable execution context, dependencies and tests, or validated cases | Add syntax/test-pass evidence instead of relying only on text similarity |
| The proxy represents developer value or time saved | Human judgment or user-study evidence | Validate the relationship between offline metrics and usefulness |

Missing model predictions are the immediate handoff dependency. Segmentation
is useful for subgroup analysis but is not required for the current overall
baseline conclusions. Training and executable correctness are later-stage
dependencies, not reasons to delay sharing the current work.

## Research grounding

- [DECODE](https://arxiv.org/abs/2607.25130) motivates using real in-IDE developer trajectories rather than only final commits.
- [Next Edit Prediction](https://arxiv.org/abs/2508.10074) defines the task around both edit location and content.
- [Coeditor](https://arxiv.org/abs/2305.18584) motivates prior-edit context and line-diff representations for multi-round editing.
- [When Models Edit Too Much](https://arxiv.org/abs/2609.04061) treats edit fidelity and unnecessary rewriting as distinct from correctness.

The persistence and rework rules here are project-specific extensions, not
published standard metrics. Their formulas, denominators and limitations are
therefore reported explicitly.

## Decision and next-step plan

The baseline-only evaluation is no longer blocked, and it produced important
method findings. It is now near diminishing returns for the main research
question: more heuristic baselines may add activity but will not tell us whether
a next-edit model works. The next high-information event is one real prediction
file evaluated on the frozen user-grouped test split.

Phase 1 is complete: the protocol is frozen, cases are labeled,
reconstructability is bounded, and source heterogeneity is documented. The
next steps are:

1. **Ingest one real model without changing the metric.** Require the shared
   prediction contract and score only the frozen test examples.
2. **Apply the decision gate.** A model advances only if it improves joint edit
   F1 and does not reproduce the repeat baseline's false-intervention or
   negative-rework failure. Then inspect change-size and persistence results.
3. **Use segmentation only after the overall gate.** Segment-level claims are
   diagnostic; they should explain a validated overall result, not replace it.
4. **Seek a second case reviewer only if the team wants reliability claims.**
   The existing labels are sufficient for transparent examples but not for
   inter-rater agreement.

Adding more heuristic baselines is lower value than integrating one real model.

## Meeting summary

> I own the evaluation layer for immediate edit quality, abstention, durability
> and remaining work. On 2,292 complete trajectories and 16,748 horizon-1 tasks,
> repeat-last-edit gains 15.38 points of joint edit F1, but loses 3.64 points of
> exact match and raises false intervention by 51.32 points. It also increases
> pooled remaining line work by 33.55%, improving only 1.97% of changed targets
> while worsening 47.39%. Only 34.24% of eligible edit blocks persist through
> one follow-up state, falling to 18.29% under a five-state rule. On the 14,456
> tasks with enough follow-up, 72.39% of repeat-last-edit's immediate match
> credit disappears when the reference is restricted to stable edits. The evaluator
> Phase 1 now closes with a fixed 20-case taxonomy, a conservative
> reconstructability audit, source-model sensitivity, globally optimal matching,
> a user-grouped split, a frozen protocol and 23 regression tests. The next
> high-information step is one contract-compliant real model output.

## Presentation files

- `decode_evaluation_study.ipynb`: Colab report and full workflow.
- `outputs/primary_scorecard.csv`: main decision table for presentation.
- `outputs/evaluation_validity_audit.csv`: target, ambiguity and timing checks.
- `outputs/baseline_oracle_ceiling.csv`: target-aware ceiling for the two heuristics.
- `outputs/edit_size_breakdown.csv`: performance by target change magnitude.
- `outputs/user_grouped_split_manifest.csv`: leakage-safe split for future models.
- `outputs/case_review_20.csv`: fixed labels and per-case reconstructability.
- `outputs/case_audit_summary.csv`: case and evidence-level denominators.
- `outputs/source_model_heterogeneity.csv`: descriptive source strata and thresholds.
- `outputs/metric_claim_specification.csv`: metric inputs, formulas, denominators, failure cases and claim boundaries.
- `outputs/temporary_reward_audit.csv`: immediate-versus-stable credit on the same eligible tasks.
- `outputs/temporary_reward_cases.csv`: eight concrete diffs where immediate credit is removed by the persistence rule.
- `outputs/evaluation_protocol.json`: frozen evaluation choices.
- `outputs/headline_results.csv`: immediate baseline result.
- `outputs/paired_baseline_comparison.csv`: paired deltas and clustered intervals.
- `outputs/rework_proxy_results.csv`: remaining-work and over-editing result.
- `outputs/persistence_thresholds.csv`: persistence sensitivity.
- `outputs/target_type_breakdown.csv`: insert/delete/replace analysis.
- `outputs/prediction_level_h1.parquet`: auditable prediction-level scores.
- `outputs/prediction_contract_template.csv`: model handoff format.
- `tests/test_evaluation_core.py`: metric and contract regression tests.

The notebook uses a standard Colab CPU. Full mode takes about 6–12 minutes;
quick mode is only a smoke test.
