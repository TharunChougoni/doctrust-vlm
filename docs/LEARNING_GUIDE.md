# Learning guide

This file explains what each stage is doing. Read one section immediately before running the corresponding command.

## 1. Inference, not training

The project currently performs **inference**:

```text
document image + question
→ processor
→ tensors
→ frozen VLM
→ generated answer tokens
→ decoded text
```

No gradients are computed, no loss is backpropagated and no model weights are updated.

That is different from:

- full fine-tuning: update all or most weights;
- LoRA/QLoRA: train small low-rank adapters while freezing the base model;
- continued pretraining: train on next-token prediction over new raw data.

The MVP stays at inference because controlled evaluation must exist before adaptation has meaning.

## 2. What the processor does

`AutoProcessor` combines image preprocessing and tokenization.

For the image it typically:

1. loads RGB pixels;
2. resizes/tiles according to the model;
3. normalizes pixel values;
4. returns image tensors.

For text it:

1. applies the model's chat template;
2. converts prompt text into token IDs;
3. creates attention metadata.

Conceptually:

```text
image: [H, W, 3] → pixel_values: [B, C, H', W'] (or model-specific tiles)
text → input_ids: [B, T]
```

`B=1` in this project because only one image/question is processed at once.

## 3. What the VLM contains

Granite Vision combines:

```text
SigLIP vision encoder
→ visual feature vectors
→ connector MLP
→ Granite language model
→ answer tokens
```

The vision encoder extracts visual features. The connector maps them into a representation the language model can consume. The language model generates one token at a time conditioned on the image and question.

## 4. Why 4-bit loading helps

A normal FP16 parameter uses 16 bits. A quantized parameter uses roughly 4 bits plus quantization metadata.

Approximate weight-only intuition:

```text
2B parameters × 2 bytes (FP16) ≈ 4 GB
2B parameters × 0.5 bytes (4-bit) ≈ 1 GB + metadata
```

This is not total VRAM. Activations, image features, KV cache and CUDA runtime memory are additional. That is why the code still uses batch size one and short outputs.

Quantization here reduces inference memory. It does not automatically make the model more accurate or robust.

## 5. Why paired corruptions matter

A simple blur benchmark asks:

```text
Did accuracy decrease after blur?
```

DocTrust asks a more diagnostic question:

```text
Did the model fail because answer evidence was damaged,
or does equally sized irrelevant damage also break it?
```

The evidence and distractor boxes have equal size. Their location changes.

Expected behavior:

```text
clean                  → original answer
global mild corruption → original answer
distractor occlusion   → original answer
evidence occlusion     → UNANSWERABLE
```

This separates nuisance sensitivity from evidence dependence.

## 6. ANLS

Exact match is too strict for small formatting differences. ANLS uses normalized Levenshtein similarity:

```text
similarity = 1 - edit_distance(prediction, answer) / max(lengths)
```

DocVQA-style ANLS sets similarities below 0.5 to zero and keeps the best score among accepted answers.

Example:

```text
gold:       4,250
prediction: 4250
```

Normalization removes punctuation before comparison, so these are treated as equivalent by this implementation.

## 7. Abstention

For evidence-destroyed images, ordinary accuracy is the wrong target because the visible answer no longer exists. The desired output is:

```text
UNANSWERABLE
```

The important failure is a **false answer**: the model produces a plausible value despite missing visual evidence. In document verification, a wrong confident answer can be more harmful than no answer.

## 8. Why clean performance is always reported

A model can look robust if it is already bad on clean examples and therefore has little room to decline. Always report clean ANLS beside corrupted ANLS.

Do not summarize robustness using one number until you have enough examples and confidence intervals.

## 9. What to inspect manually

For every small MVP run, inspect:

- transformed images;
- the exact prompt;
- raw, unnormalized model output;
- accepted answers;
- evidence-box placement;
- false-answer examples;
- latency and peak allocated VRAM.

Numbers without this inspection can hide a broken evaluation pipeline.

## 10. Learning checkpoint

Before expanding the dataset, be able to explain:

```text
image/question
→ processor tensors
→ frozen VLM generation
→ decoded prediction
→ ANLS or abstention evaluation
```

Only after that should you add models, more corruptions or LoRA.
