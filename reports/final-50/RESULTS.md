# Final 50-document two-model results

Run completed on a Tesla T4 with 50 manually audited DocVQA-derived documents, five paired variants per document, and two FP16 VLMs (500 predictions total).

## Main comparison

| Model | Clean-correct | Clean mean ANLS | JPEG conditional ANLS | Blur conditional ANLS | Distractor conditional ANLS | Evidence abstention | Conditional false-answer rate | Mean latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolVLM 2.2B | 38/50 | 0.7213 | 0.9229 | 0.9271 | 0.9228 | 0% | 100% | 1.49 s | 9,051 MB |
| Qwen2.5-VL 3B | 45/50 | 0.8969 | 0.9966 | 0.9987 | 0.9966 | 14% | 86.7% | 2.00 s | 13,041 MB |

Qwen's paired clean-ANLS improvement over SmolVLM was **+0.1756**, with a deterministic 10,000-replicate paired-bootstrap 95% interval of **[+0.0485, +0.3079]** on this selected 50-document subset. Its evidence-abstention-rate improvement was **+0.14**, interval **[+0.06, +0.24]**.

## Interpretation

Qwen performed better on clean documents and retained answers more reliably under JPEG, blur, and distractor occlusion. It also abstained sometimes after answer evidence was removed, unlike SmolVLM. However, Qwen still produced a non-abstaining answer on 86.7% of evidence-occluded cases within its clean-correct subset, so neither model demonstrated reliable evidence-grounded abstention.

These results support a controlled course/deadline proof of concept, not DocVQA-wide or production robustness claims. Evidence boxes were OCR-derived and manually audited; synthetic occlusion is not equivalent to naturally damaged documents.

## Reproducibility

The files in this directory contain raw predictions, manifests, exact configs, model revisions, environment metadata, per-model metrics, comparison tables, and paired-bootstrap output. Source images are intentionally omitted due to dataset redistribution terms.
