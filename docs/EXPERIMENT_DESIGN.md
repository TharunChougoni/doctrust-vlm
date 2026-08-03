# Experiment design

## Primary question

When a document is degraded, does a compact VLM:

1. preserve answers under harmless capture noise;
2. ignore matched damage to irrelevant regions;
3. abstain when the answer evidence is destroyed?

## MVP hypotheses

- H1: Mild JPEG compression and blur reduce ANLS relative to clean images.
- H2: Answer-evidence occlusion causes a larger answer-flip rate than matched distractor occlusion.
- H3: Explicit `UNANSWERABLE` prompting reduces, but does not eliminate, false answers after evidence removal.

These are hypotheses, not repository claims.

## Unit of analysis

One source question/image pair is the unit. Every transformed variant remains tied to `source_id` so paired comparisons can be made.

Do not split variants from one source across train/development/evaluation sets.

## Current variants

| Variant | Expected behavior | Purpose |
|---|---|---|
| clean | preserve | establish capability |
| JPEG quality 35 | preserve | digital degradation |
| Gaussian blur 1.5 | preserve | capture/focus degradation |
| distractor occlusion | preserve | non-answer damage control |
| evidence occlusion | abstain | answerability/hallucination test |

## MVP metrics

- mean ANLS by answer-preserving variant;
- abstention rate after evidence removal;
- false-answer rate after evidence removal;
- latency and peak allocated VRAM by example.

After 100+ audited examples, add paired bootstrap confidence intervals, answer-flip rate and evidence-locality gap.

## Error taxonomy

Assign one category during qualitative review:

1. correct;
2. OCR/character confusion;
3. wrong region/layout binding;
4. stale answer after semantic change;
5. hallucination after evidence removal;
6. unnecessary abstention;
7. malformed/verbose output;
8. annotation or evidence-box error;
9. unknown.

## Reproducibility checklist

- fixed manifest and source IDs;
- model ID and, later, revision hash;
- deterministic decoding (`do_sample=False`);
- corruption parameters in YAML;
- raw predictions cached before normalization;
- package versions and GPU recorded;
- all transformed images manually checked in the MVP;
- no cherry-picked-only failure gallery.

## Phase 2: semantic counterfactuals

Once the MVP works, generate fictional invoices/forms with known field boxes. Create paired edits such as total/date/quantity changes while preserving layout. Add:

- counterfactual update accuracy;
- stale-answer rate;
- unseen-template evaluation.

This is deferred because reliable arbitrary text replacement in existing scanned documents is a separate engineering problem.

## Claim boundary

A small synthetic study supports a statement such as:

> Implemented a controlled proof of concept for evidence-conditioned document-VLM robustness.

It does not support:

- production robustness;
- state-of-the-art performance;
- forgery-detection capability;
- a novel model architecture;
- completion of LoRA/Qwen work that was not run.
