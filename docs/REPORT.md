# Weight-Space Programming on the American Eel Model

7 November 2026 (v1); v2 added 8 November 2026

## Summary

Direct weight editing worked for exact surgery and diagnosis on the eel model, but it could not repair its remaining errors: those errors sit in the features or the labels, not in a single editable row.

- **The model is a one-class detector in a 22-class shell.** Only class 21 (American_Eel) is alive; the other 21 rows are identical and pinned near zero probability. Cutting the head to one class reproduces every metric to 16 decimal places.
- **One of three detection scales is dead.** The stride-32 branch never fires; silencing it in the weights changes nothing. Stride 16 finds every eel above about 100 px; stride 8 handles short ones, poorly.
- **The recall gap is short eels, and edits cannot close it.** Recall is 38% under 60 px and 71% at 60–100 px, against 98–100% above 150 px. The best possible linear edit of the stride-8 row buys each extra eel with 10–30 false positives on held-out frames.
- **Many of those misses are probably not real failures.** On visual review, most "short eels" are eels cut off at the dark circular opening or the frame edge, or near-invisible in dark infrared frames.
- **Mask accuracy is capped by resolution.** Eels are about 2 mask pixels wide at 512 px input; a perfect mask half a pixel off scores IoU 0.65. That explains most of the gap between mask mAP50-95 (0.549) and box mAP50-95 (0.762).
- **Generalization cannot be measured with this data.** Every available frame was used in training. As a proxy, the model is robust to brightness, blur, compression, flips and colour, but loses 0.136 mAP50 under low contrast and 0.085 under sensor noise.

**v2 result: a better model with no new data.** On the held-out half of the video, box mAP50-95 rose from 0.744 to 0.779, mask mAP50-95 from 0.528 to 0.561, and recall from 0.894 to 0.944. All of it is in one checkpoint, `weights/eel_v2.pt`:

- **The false positives were duplicates.** 329 of 403 false positives are second boxes on eels that were already found. A parameter-free peak-suppression rule in the head removes all but 1 of them, and needs no NMS.
- **The discarded second head is useful.** YOLO26 ships a one-to-many head that inference throws away. Averaging its weights with the deployed head destroys the model. An exact closed-form widening of the class branch averages the two heads' *outputs* inside one network, and that helps.
- **Weight interpolation fixes a bad fine-tune.** Fine-tuning on the first half raised recall but cost box accuracy on the second half. Moving only 25% of the way from the base weights to the fine-tuned weights kept part of the recall gain and none of the loss.
- **Temporal smoothing, flip TTA, box voting and calibration shifts did not help.** The remaining misses repeat in consecutive frames, so they are probably the partly visible eels from v1.
- **Mask boundaries are about 1.8 px off and about 1 px too fat.** Masks come from a grid at 1/4 resolution, and an eel is less than 2 grid cells wide. Thresholds, guided filtering, larger inputs and a fine-tune with full-resolution mask loss did not fix this. A small learned refiner (38k parameters) removes the fat bias and moves edges closer to the labels, but it does not raise mean IoU. Half of the IoU loss is not at the boundary: it comes from masks cut off by short boxes and from touching, dark or curled eels.

The next real step is held-out footage plus a labelling rule for partly visible eels; see Recommendations.

## Background

Weight-space programming means changing a network by reading and rewriting its weights directly, from an understanding of what each part computes, instead of training it with gradient descent.

The term comes from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming) (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, 2 November 2026). Its headline result: an LLM edited 131,072 weights (0.016%) of Qwen3.5-0.8B to remove a looping bug, and MATH-500 rose from 7.9% to 48.1%. The authors call the results early "sparks" and report failures, such as sycophancy, which could not be identified from the weights alone, and a Gemma 3 270M model whose math score editing could not raise, which they read as editing being unable to "add an ability the model lacked". An older, related line edits models arithmetically with task vectors ([Ilharco et al.](https://arxiv.org/abs/2212.04089v2); [Ortiz-Jimenez et al.](https://arxiv.org/abs/2305.12827v3)).

The goal here was to apply the idea to a small production vision model: understand the eel detector from its weights, make exact edits, and test whether any remaining errors are a localized "bug" that an edit can fix. All edits below are hand-reasoned or closed-form; a gradient-fitted probe was used once, only to measure a ceiling, never as an edit to keep.

## Model and data

The checkpoint is a 3.06M-parameter YOLO26n segmentation model with 22 class slots, of which only American_Eel is trained; all 4,986 available frames were part of its training set.

| Item | Detail |
| --- | --- |
| Checkpoint | `eel_weights_20261107.pt`, Ultralytics 8.4.22, `yolo26n-seg`, fp16, end-to-end (no NMS) one-to-one head |
| Training | 50 epochs at 512 px, fine-tuned from `eel_weights_20261022.pt` |
| Class slots | 22 (`fish`, 20 species or juveniles, `American_Eel` = 21) |
| Labels in data | 5,013 instances, all class 21; 27 frames hold two eels |
| Frames | 4,986 PNG, 640×360, grayscale-looking infrared video |
| Sampling | About 366 clips of about 13 consecutive frames, spaced about 400 frames apart, across frames 401035–529830 |
| Training list | `train.txt` lists every frame 400867–530022 (129,156); every local frame is in it |
| Eel size | Median length 23% of the frame's long side; median width about 10 px |
| Compute | CPU only (14 cores, 15 GB RAM, WSL2); no usable GPU |

The head reads as a one-class model in the weights themselves. Every scale's classifier has the same pattern:

| Classifier row | Stride 8 bias | Stride 16 bias | Stride 32 bias |
| --- | --- | --- | --- |
| 21 unused classes (all near-identical) | −12.98 | −12.03 | −8.0 |
| American_Eel | −2.17 | +1.18 | −8.00 |

The unused rows were only ever pushed down as negatives, so they never learned anything species-specific. The eel row at stride 32 is indistinguishable from them, the first sign that this scale never learned eels.

Two data issues affect tooling: the image folder is spelled `imges/`, which breaks Ultralytics' image-to-label path lookup, and `data.yaml` points train, val and test at the same list. Evaluation used copies of the frames in a correctly named folder; `data/` was left unchanged.

## Experiments and results

Eight experiments ran in order, each building on the last; scores are on training frames, so they measure fit, and every edit is judged on frames from a different part of the video than it was fitted on.

### 1. Prune the head to one class

The 21 unused rows were sliced out of every class-sized output layer (`cv3`, `one2one_cv3`, `proto.semseg`), leaving a one-class model (`weights/eel_1cls.pt`). It is exactly lossless:

| Model | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| Original, 22 classes | 0.960 | 0.893 | 0.962 | 0.762 | 0.956 | 0.549 |
| Pruned, eel only | 0.960 | 0.893 | 0.962 | 0.762 | 0.956 | 0.549 |

All metrics agree to 16 decimal places on 4,986 frames, and a frame-by-frame check on 200 frames found zero difference in boxes, confidences and mask pixels down to 0.001 confidence. Inference time moved from 49.7 to 47.1 ms per image, within run-to-run noise.

### 2. Find which eels are missed, and why

A per-object diagnostic on 1,003 eels (every fifth frame) recorded each eel's size, the best box any anchor proposes for it, and which stride detects it. Recall at 0.25 confidence is 91.9%, with 86 false positives. Of the 81 misses, 78 are score failures (a well-placed box scored below 0.25, median 0.04) and only 3 are localization failures. Stride 16 makes 846 detections, stride 8 makes 76, stride 32 makes none.

![Recall falls to 38% for eels under 60 px](report_assets/recall_by_length.png)

*Source: `results/diag_baseline.csv` · 1,003 eels, every fifth training frame, 512 px input.*

All short-eel detections come from stride 8; from about 100 px up, stride 16 takes over and recall is 93% or more.

### 3. Edit the stride-8 eel row

The stride-8 eel score is a linear function of 64 features, so features were cached once and any edit of that row was scored exactly without re-running the network. Two edits were tried: a bias shift (E1) and a rank-1 edit along the closed-form LDA direction separating missed eels from background (E2). They were fitted on the first half of the video and scored on the second half, which has 67 target eels.

| Edit | Short eels recovered (held-out, of 67) | Extra false positives |
| --- | --- | --- |
| E1 bias +1.0 | +2 | +38 |
| E1 bias +2.0 | +7 | +77 |
| E2 LDA, α=4, β=−1 | +12 | +156 |
| E2 LDA, α=8, β=0 | +38 | +1,103 |

Every recovered eel costs 10–30 false positives. The new false positives split about evenly between background, duplicates of eels stride 16 already finds, and duplicate hits on the same short eel. No edit was kept.

### 4. Ceiling test: could any linear row do better?

A class-weighted logistic regression was fitted as a probe of the best achievable stride-8 row (a measurement, not an edit to keep). At matched recall of short eels on the held-out half:

| Short-eel recall | Current row, false positives | Best linear row, held-out | Best linear row, on its own fit half |
| --- | --- | --- | --- |
| 21% | 117 | 129 | 37 |
| 30% | 245 | 234 | 112 |
| 51% | 689 | 429 | 256 |

On held-out frames the best possible row is no better at low recall and only modestly better at 51%. The 64 features do not carry what is needed to separate these eels from background, so no edit of this row can fix them. The in-sample column is the overfitting warning: the same probe looks several times better on the frames it was fitted on.

### 5. Confirm stride 32 is dead

Setting the stride-32 eel bias to −100 in both heads (`weights/eel_1cls_no_s32.pt`) left predictions on 200 frames identical: same detection counts, zero box, confidence and mask differences.

### 6. Try larger input sizes

Rerunning the diagnostic at 640 and 768 px did not raise short-eel recall; detections just moved from stride 8 to stride 16.

| Eel length | 512 px | 640 px | 768 px |
| --- | --- | --- | --- |
| Under 60 px | 38% | 33% | 36% |
| 60–100 px | 71% | 73% | 68% |
| All eels | 91.9% | 91.9% | 91.3% |

So the limit is not input pixels. It is in what the model learned, or in the labels.

### 7. Look at the missed short eels

Sixteen missed eels under 100 px were cropped and reviewed by eye, with and without their label outline. Most were not small eels: they were eels cut off as they enter the dark circular opening or leave the frame edge, and several sat in frames dark enough that no eel was visible at all. Those labels may come from tracking or interpolation. This is a judgement from low-resolution crops and needs checking by someone who knows the footage.

![Sixteen missed short eels: raw crop (left) and label outline in yellow (right); L = length in px, s = best score](report_assets/missed_short_eels.png)

### 8. Mask accuracy is capped by resolution

Ultralytics scores masks at a quarter of the input size, where the median eel is about 2 pixels wide at 512 px. Each label polygon was rasterized at that resolution, and again shifted by half a pixel, to see what a perfectly shaped but slightly misplaced mask scores:

| Input size | Mask grid | Mean IoU | Share at IoU 0.75 or above | Share at 0.9 or above |
| --- | --- | --- | --- | --- |
| 512 px | 128×72 | 0.65 | 14% | 0.4% |
| 640 px | 160×90 | 0.71 | 30% | 0.4% |
| 768 px | 192×108 | 0.74 | 49% | 1.0% |
| 1024 px | 256×144 | 0.80 | 80% | 2.5% |

At 512 px, even near-perfect masks mostly fail the stricter IoU thresholds that make up mAP50-95, so the gap between mask and box mAP50-95 is mostly geometry. Running the existing model at 640 px did not help (mask mAP50-95 fell from 0.554 to 0.530); a gain would likely need training at the larger size, which was not tested.

## Generalization and overfitting

Whether the model is overfit cannot be measured with this data, because every available frame was in its training set; what can be controlled is that the edits themselves are not overfit, and how brittle the fit is.

- **No held-out data exists.** `train.txt` lists all 129,156 frames of the video with no gaps, and all 4,986 local frames are among them. The metrics above (box mAP50 0.962) are training-set fit.
- **Edits were tested on a separate part of the video.** Neighbouring frames are near-duplicates, so a random or every-fifth-frame split leaks. The first split used here did leak and was replaced: edits were fitted on the first half of the video and scored on the second half, with no shared clips.
- **The leak matters.** In experiment 4, the same probe looked several times better on its own half than on the other half (37 against 129 false positives at 21% recall). An edit judged in-sample would have looked like a success.
- **Robustness proxy.** The model was re-scored on 1,000 frames under realistic changes. Large drops under mild changes are a sign that the fit depends on this footage's exact look.

![Only low contrast and noise cut box mAP50 by more than 0.013](report_assets/robustness.png)

*Source: `results/robustness_baseline.json` · 1,000 training frames (every fifth), pruned one-class model.*

The model shrugs off brightness, blur, JPEG compression, horizontal flips, colour removal and input-size changes. It is brittle to low contrast and to sensor noise, the conditions of turbid water or low light, and the same conditions as the missed eels in experiment 7. Grayscale changes nothing, consistent with infrared footage.

## Assessment

Weight-space editing earned its place here as a microscope and a scalpel, not as a repair tool, because the model's remaining errors are not a localized bug.

| Use | Outcome | Evidence |
| --- | --- | --- |
| Reading the model from its weights | Worked | Dead classes and the dead stride-32 eel row were visible in the biases before any data was run |
| Exact surgery | Worked | One-class head: every metric identical |
| Proving a component unused | Worked | Stride-32 ablation: predictions identical |
| Locating the failure | Worked | Misses traced to stride-8 scoring, not localization |
| Bounding what an edit can do | Worked | Probe ceiling showed the stride-8 features cannot separate the missed eels |
| Repairing recall | Did not work | Best edits cost 10–30 false positives per eel recovered |
| Repairing masks | Not attempted | The limit is the mask grid's resolution, not the weights |

### Could these findings have come without the weights?

Mostly yes. The weights were a fast first look that produced hypotheses; most findings were confirmed by observing outputs, activations or labels, which is ordinary model debugging.

| Finding | Could it come from logs, metrics or observation? | What the weights added |
| --- | --- | --- |
| 21 of 22 classes dead | Yes, trivially: the labels contain only class 21, and the per-class val table shows 0 instances for the rest | Confirmed they are inert (pinned near zero), so they cannot cause false positives |
| One-class pruning is lossless | Predictable from the above; proven only by running both models | Made the edit possible; the proof was observational |
| Stride 32 is dead | Not from standard logs or metrics, which report nothing per scale; findable only by instrumenting the head to record which anchor made each detection | The one finding the weights gave first and cheaply: its eel bias matched the dead classes' rows (a hint, confirmed by observation) |
| Short eels are the recall gap | Yes: predictions plus labels, binned by size; COCO-style AP small/medium/large would show it | Nothing |
| Misses are score failures, not localization | Needs the head's raw per-anchor scores (outputs, not weights) | Nothing |
| Stride-8 features cannot separate missed eels | Needs internal activations plus a probe | Nothing directly |
| Mask accuracy capped by resolution | Yes, from labels alone plus the 1/4-resolution mask grid | Nothing |
| Brittle to low contrast and noise | Observation only | Nothing |
| No held-out data | From `train.txt` | Nothing |

Only "stride 32 is dead" is invisible in normal logs and metrics, and even that is reachable by observing the head's outputs. The training run's `results.csv` and `args.yaml`, if available, would likely have shown some of this too (for example `nc=22` trained on single-class data).

The contrast with the MIT result is instructive. Qwen's looping was a specific behaviour driven by a few directions in a few layers, so a small edit could remove it. The eel model's misses come from missing information (dark, cut-off or ambiguous eels) and from resolution, so there is no small set of weights whose change would supply it. The MIT authors report the same boundary on Gemma 3 270M.

## v2: false positives, the hidden second head, weight interpolation and mask boundaries

v2 started from the false positives. Their main cause was duplicate boxes on eels that were already found, which the deployed head does not suppress. Removing them, using the discarded second head, and interpolating fine-tuned weights raised held-out box mAP50-95 from 0.744 to 0.779 and mask mAP50-95 from 0.528 to 0.561.
Experiment 15 then examined mask boundary pixels at full resolution.

**Protocol.** "Held-out" below means the second half of the video (frames 470754–529830, 2,493 frames). Nothing in v2 was fitted on it: the fine-tune trained on the first half only, and every decoding rule has no fitted parameters. "Full" means all 4,986 frames. "Subset" means every fifth frame (1,000 frames), used for fast screening. Fitness is the sum over box and mask of 0.1 × mAP50 + 0.9 × mAP50-95.

### 9. False-positive census

Every detection at score ≥ 0.25 on all frames was classified against the labels.

| Class | Count | Meaning |
| --- | --- | --- |
| True positive | 4,613 | Matches a labelled eel at IoU ≥ 0.5 |
| Duplicate | 329 | Second box on an eel that already has a true positive |
| Mislocalized | 73 | Overlaps an eel at IoU 0.1–0.5 |
| Background | 1 | Overlaps nothing |

- **Duplicates are the main false positive.** 166 sit at the same scale, and 165 of those are on the anchor cell next to the true positive. The other 163 are cross-scale doubles (stride 8 and stride 16 both fire). Their scores are about 0.67 × the true positive's score, so a confidence threshold cannot remove them.
- **Mislocalized boxes are mostly on missed eels.** 62 of the 73 sit on an eel with no true positive: a partial box on a curled eel, or one box spanning two eels.
- **Background false positives do not exist** (1 in 4,986 frames).

![Duplicate detections](report_assets/v2/fp_duplicate.png)

![Mislocalized detections](report_assets/v2/fp_mislocal.png)

A box refined from its own mask was tested as a fix for mislocalized boxes. It failed: true positives at IoU ≥ 0.5 fell from 100% to 90%, because the mask is too coarse (see experiment 8).

### 10. The discarded one-to-many head

YOLO26 trains two complete heads: a one-to-one head (deployed, no NMS) and a one-to-many head that is normally thrown away. Both are in the checkpoint.

| Full set | Box mAP50 | Box mAP50-95 | Mask mAP50-95 |
| --- | --- | --- | --- |
| Deployed one-to-one head | 0.962 | 0.762 | 0.549 |
| One-to-many head + NMS 0.7 | 0.971 | 0.771 | 0.553 |
| One-to-one head + NMS 0.5 on top | 0.978 | 0.781 | 0.568 |

The deployed head leaves duplicates. Light NMS on its output already removes most of them.

**Weight averaging of the two heads fails.** Interpolating the one-to-one weights toward the one-to-many weights destroyed the model: box mAP50 fell to 0.145 at t = 0.5 and to 0.794 at t = 0.25. A class-branch-only blend gave mAP50 0.501, with precision 0.028. The two heads share no basis, so their weights cannot be averaged.

**Output averaging works, and it can be written exactly into the weights.** `merge_heads_wide.py` builds one class branch of double width (128 channels). Its output equals 0.5 × one-to-one logits + 0.5 × one-to-many logits, with a maximum error of 4e-5:
- BatchNorm is folded into each conv.
- The first depthwise conv is interleaved.
- The middle 1×1 convs are block-diagonal.
- The last 1×1 conv is [0.5·W₁ | 0.5·W₂].

The result is a normal checkpoint (3.18M parameters) that loads in stock Ultralytics. This is weight-space programming in the MIT sense: a closed-form rewrite that adds a capability (an ensemble) with no training.

### 11. Peak suppression: NMS-free duplicate removal

The duplicates are on neighbouring anchor cells and neighbouring scales. A CenterNet-style rule removes them inside the graph, before decoding: an anchor keeps its class logit only when that logit is the maximum of its own 3×3 neighbourhood and is not beaten by the overlapping cells of the next scale up or down. Other anchors get −20 on their logit. There are no fitted parameters.

| Full set | Box mAP50 | Box mAP50-95 | Mask mAP50-95 | Fitness |
| --- | --- | --- | --- | --- |
| Deployed | 0.962 | 0.762 | 0.549 | 1.3715 |
| Peak suppression | 0.976 | 0.789 | 0.576 | 1.4229 |
| Peak suppression + NMS 0.5 | 0.976 | 0.790 | 0.577 | 1.4249 |
| Class mean of both heads + peak + NMS 0.5 | 0.976 | 0.793 | 0.578 | 1.4288 |

After peak suppression, duplicates at score ≥ 0.25 fall from 329 to 1. Mislocalized boxes fall from 73 to 58. True positives stay at 4,611 against 4,613.

Variants tested on the subset (peak suppression alone scores 1.4314):

| Variant | Subset fitness | Result |
| --- | --- | --- |
| 5×5 window | 1.4339 | Same within noise |
| Soft penalty −8 / −4 / −2 instead of −20 | 1.4314 / 1.4161 / 1.4025 | A hard rule is needed |
| Stride-8 or stride-16 logit shift ±0.5 | 1.424–1.432 | No gain; calibration is not the problem |
| Input 576 px | 1.4086 | Worse |
| Mask threshold other than 0 | Worse | The default is optimal |

### 12. Other decoding ideas (negative or small)

- **Flip test-time augmentation.** Ultralytics `augment=True` is silently a no-op for this end-to-end segmentation model. A head-level flip TTA was built: mirror the batch, un-mirror the maps, swap left/right box distances, and average. It helps only the one-to-many head with NMS (subset 1.4098 → 1.4215). It hurts with the one-to-one head or with peak suppression, and it doubles the compute.
- **Box voting.** Each NMS-kept box is replaced by the score-weighted mean of all overlapping candidates. Mask coefficients can also be averaged. On the one-to-many head it adds +0.007 (subset 1.4098 → 1.4172), and the best variant (one-to-one class, one-to-many box) reaches 1.4327. That is still below peak suppression with class mean (1.4359). The cluster IoU (0.5–0.7), score exponent and minimum-score options made no difference.
- **Temporal re-scoring.** Frames come in clips of about 13 consecutive frames. A detection's score was raised (or blended) with the score of overlapping detections in neighbouring frames, and missing boxes were interpolated. 49 variants were tried; the best changed AP50-95 by +0.0002. The reason: of the 401 eels still missed after peak suppression, only 51 have a confident detection in a neighbouring frame. The model misses the same eels in consecutive frames, so neighbours cannot fill the gap.

### 13. Fine-tuning and weight interpolation (WiSE-FT)

The model was fine-tuned for 6 epochs on the first half of the video only:
- SGD, lr 0.002, cosine schedule;
- stronger brightness and scale augmentation.

It was then scored on the held-out half.

| Held-out half, peak suppression | Box R | Box mAP50 | Box mAP50-95 | Mask mAP50-95 | Fitness |
| --- | --- | --- | --- | --- | --- |
| Base model | 0.936 | 0.972 | 0.772 | 0.556 | 1.3881 |
| Fine-tuned, 6 epochs | 0.954 | 0.974 | 0.748 | 0.540 | 1.3535 |
| Fine-tuned backbone only (base head) | 0.954 | 0.974 | 0.751 | 0.544 | 1.3585 |
| Fine-tuned box branches only | 0.935 | 0.971 | 0.765 | 0.552 | 1.3786 |
| Fine-tuned class branches only | 0.938 | 0.973 | 0.768 | 0.554 | 1.3836 |
| Interpolation α = 0.15 | 0.943 | 0.975 | 0.773 | 0.558 | 1.3924 |
| **Interpolation α = 0.25** | 0.946 | 0.977 | 0.774 | 0.558 | **1.3928** |
| Interpolation α = 0.35 | 0.949 | 0.977 | 0.773 | 0.557 | 1.3915 |
| Interpolation α = 0.5 | 0.953 | 0.978 | 0.770 | 0.556 | 1.3879 |
| Interpolation α = 0.75 | 0.952 | 0.977 | 0.764 | 0.552 | 1.3793 |

(The α = 0 checkpoint reproduces the base model exactly, which checks the tool.)

- **Fine-tuning trades box accuracy for recall.** Recall rose by 1.8 points, but box mAP50-95 fell by 2.3 points.
- **The damage is in the backbone.** The fine-tuned backbone alone gives almost all of the loss. The fine-tuned head branches alone give neither the gain nor the loss.
- **Interpolation keeps part of the recall gain and none of the loss.** θ = θ_base + 0.25 · (θ_ft − θ_base) is the best point. It adds 1 point of recall and 0.5 points of box mAP50, and box mAP50-95 rises a little. Gains are small but consistent across α = 0.15–0.35.

A second fine-tune trained only the head, with the backbone frozen. It lost box accuracy too (held-out fitness 1.3522; interpolated at α = 0.5: 1.3781). So the loss does not come only from the backbone. This recipe on this footage pulls boxes away from the label style. Interpolation is the step that keeps the gain.

### 14. The v2 model on the held-out half

| Held-out half (frames 470754–529830) | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Fitness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deployed model, as shipped | 0.950 | 0.894 | 0.955 | 0.744 | 0.946 | 0.528 | 1.3346 |
| + NMS 0.5 on top | 0.974 | 0.936 | 0.973 | 0.765 | 0.963 | 0.549 | 1.3768 |
| + peak suppression | 0.977 | 0.936 | 0.972 | 0.772 | 0.964 | 0.556 | 1.3881 |
| + class mean of both heads + NMS 0.5 | 0.979 | 0.937 | 0.973 | 0.777 | 0.966 | 0.559 | 1.3960 |
| + WiSE-FT α = 0.25 | 0.977 | 0.944 | 0.976 | 0.779 | 0.968 | 0.561 | 1.4004 |

**All of this is baked into one checkpoint, `weights/eel_v2.pt`.** It is built in two steps:
1. `wise_ft.py` makes the α = 0.25 interpolation.
2. `merge_heads_wide.py --box-o2m --mask-o2m` replaces the deployed head with the exact widened class mean and the one-to-many box and mask branches.

It loads in stock Ultralytics. `experiments/eel_v2_infer.py` adds peak suppression (a 20-line patch of `forward_head`) and optional NMS 0.5. It reproduces the best held-out score exactly (fitness 1.4004).

| `weights/eel_v2.pt` + peak + NMS 0.5 | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Fitness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Held-out half | 0.977 | 0.944 | 0.976 | 0.779 | 0.968 | 0.561 | 1.4004 |
| Full set | 0.978 | 0.952 | 0.979 | 0.796 | 0.975 | 0.581 | 1.4347 |
| Full set, deployed model for comparison | 0.960 | 0.893 | 0.962 | 0.762 | 0.956 | 0.549 | 1.3715 |

Without the NMS 0.5 step, `eel_v2.pt` + peak scores 1.3839 on the held-out half. The class-mean head is less sharp than the one-to-one head, so it needs the NMS step.

### 15. Mask boundaries at full resolution

The standard mask mAP hides boundary errors. Ultralytics validation scores masks at 1/4 of the input resolution and downsamples the labels to match. `eval_config.py --native 1` scores at full input resolution instead (logits upsampled before the threshold, labels at `mask_ratio=1`).

| Held-out half | Mask mAP50 (1/4 res) | Mask mAP50-95 (1/4 res) | Mask mAP50 (full res) | Mask mAP50-95 (full res) |
| --- | --- | --- | --- | --- |
| Deployed model | 0.946 | 0.528 | 0.944 | 0.505 |
| `eel_v2.pt` + peak + NMS 0.5 | 0.968 | 0.561 | 0.965 | 0.538 |

The v2 gain holds at full resolution: +0.033 mask mAP50-95.

**Diagnosis.** `mask_boundary_diag.py` compares each matched mask with its rasterized label polygon at 640×360 frame resolution (2,319 held-out eels):

| Measure | Value |
| --- | --- |
| Mean mask IoU | 0.733 |
| Label eel width | about 9 px |
| Mean boundary error | 1.76 px |
| Signed boundary error | +0.93 px (masks too fat) |
| Mask area / label area | 1.14 |
| Boundary F-score at 1 px / 2 px | 0.535 / 0.743 |

- **The cause is the prototype grid.** Masks come from a grid at 1/4 of the input size. One grid cell covers about 5 frame pixels, and an eel is less than 2 cells wide. Upsampling cannot place the edge within 1 pixel. On a 9 px eel, one pixel of error costs about 20% IoU.
- **The labels are not the cause.** They have a median of 45 vertices per eel, one every 7 px, with sub-pixel coordinates.
- **Half of the IoU loss is not at the boundary.** 23% of missed eel pixels are outside the predicted box: the mask is cut off where the box is too short. About half of all error pixels are more than 1 px from the label boundary. They come from touching or crossing eels, very dark eels, and curled eels.

![Median-IoU masks: label green, prediction red](report_assets/v2/mask_median.png)

![Worst-IoU masks: label green, prediction red](report_assets/v2/mask_worst.png)

**Fixes without training (all failed):**

| Fix | Mean IoU | Result |
| --- | --- | --- |
| Mask threshold −2 to +2 logits | 0.632–0.734 | Best at 0 to +0.5; the curve is flat |
| Guided filter with the infrared frame as guide | 0.689–0.733 | No IoU gain; boundary F@1 +0.03 at best |
| Inference at 640 / 768 px | 0.734 / 0.733 | The model learned blurred logits at 512 px |
| Crop the mask to the box grown by 1–8 px | 0.708–0.730 | More spill than recovered eel |

**Learned boundary refiner.** `train_mask_refiner.py` trains a small fully convolutional network (37,921 parameters, dilated 3×3 convs, receptive field about 35 px). Its inputs are the full-resolution infrared crop, the upsampled YOLO mask logits and the box mask. It predicts a correction to the logits. Its last layer starts at zero, so the untrained refiner reproduces the YOLO mask exactly; this was checked. It was trained for 12 epochs on 2,321 eels from the first half and scored on 2,319 eels from the second half.

| Held-out eels | YOLO mask | Refined |
| --- | --- | --- |
| Boundary F at 1 px | 0.535 | **0.572** |
| Boundary F at 2 px | 0.743 | **0.770** |
| Mean boundary error | 1.76 px | **1.57 px** |
| Signed boundary error | +0.93 px | **+0.19 px** |
| Mask area / label area | 1.14 | **0.99** |
| Mean IoU | 0.733 | 0.733 |

- **The refiner fixes the boundary pixels.** It removes the fat-mask bias and moves edges about 0.2 px closer to the labels.
- **It does not raise IoU.** IoU is dominated by the large errors above, which no boundary correction reaches. Allowing the refined mask to extend past the box edge (by 2 px, 4 px or the whole window) changed nothing.
- **It works end to end.** `load_eel_v2(weights, refiner="weights/mask_refiner.pt")` refines each mask inside `predict_nms`. On 29 held-out eels run through the wrapper, boundary F at 1 px rose from 0.54 to 0.60, and area went from 1.14× to 1.00×. The refiner costs about 60 ms per eel on one CPU thread (mean crop 11,700 px; measured while another job shared the CPU).

**Fine-tune with full-resolution mask loss.** The mask loss in Ultralytics training is also computed at 1/4 resolution by default. `finetune.py --mask_ratio 1` computes it at full input resolution. Settings: first half only, 4 epochs, lr0 0.0005, mild augmentation (`--gentle 1`), batch 8. The result was then interpolated and merged exactly like `eel_v2.pt`. Held-out half, scored at full resolution:

| Model (all with peak suppression + NMS 0.5) | Box R | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Fitness |
| --- | --- | --- | --- | --- | --- |
| `eel_v2.pt` | 0.944 | 0.779 | 0.965 | 0.538 | 1.3786 |
| Mask fine-tune, raw | 0.960 | 0.767 | 0.966 | 0.531 | 1.3627 |
| Mask fine-tune, WiSE α = 0.5, no merge | 0.953 | 0.775 | 0.966 | 0.535 | 1.3738 |
| Mask fine-tune, WiSE α = 0.25 + merge | 0.945 | 0.780 | 0.965 | 0.539 | 1.3812 |
| Mask fine-tune, WiSE α = 0.5 + merge | 0.951 | **0.781** | **0.968** | **0.540** | **1.3833** |
| Mask fine-tune, WiSE α = 0.75 + merge | 0.954 | 0.779 | 0.967 | 0.540 | 1.3816 |

- **The raw fine-tune repeats experiment 13.** Recall rises, box accuracy falls.
- **After interpolation and merge, the gain is small but consistent.** All three α values beat `eel_v2.pt`, by +0.003 to +0.005 fitness. At 1/4 resolution, α = 0.5 + merge gives 1.4031 against 1.4004.
- **The full-resolution mask loss did not fix the boundaries.** Mask mAP50-95 rose by only 0.002. The prototype grid limits what training at this input size can reach.
- **`eel_v2.pt` is kept.** The gain is the same size as differences treated as noise elsewhere in this report. To build the variant: `wise_ft.py weights/eel_1cls.pt results/v2/train/ft_mask1/weights/last.pt weights/wise_mask1_a0.5.pt 0.5`, then `merge_heads_wide.py weights/wise_mask1_a0.5.pt weights/eel_v2_mask1.pt 0.5 --box-o2m --mask-o2m`.

### Training log note

The original training log shows the same picture from the outside. mAP50 was 0.965 after epoch 1 and 0.959 after epoch 50. Box mAP50-95 rose only from 0.728 to 0.759, and mask mAP50-95 from 0.493 to 0.504. The "validation" set in that log was the training list. After its first epoch, the 50-epoch fine-tune mostly polished localization on frames it had already seen. That fits experiment 13: more training on this footage moves box accuracy, not detection.

## Recommendations

Held-out footage and a labelling decision come first, because without them neither generalization nor the true miss rate can be known.

1. **Get held-out footage** from a different day, site or camera, and keep it out of training. It is the only real test of overfitting; the existing scripts can score it immediately.
2. **Set a labelling rule for partly visible eels.** Review the missed short eels (eels entering the opening, at the frame edge, or invisible in dark frames) and decide whether they should be labelled. This decides whether the remaining 8% miss rate is real.
3. **If retraining:**
    - add contrast and noise augmentation, the two conditions the model is brittle to;
    - train at 640–768 px if mask accuracy matters (the mask ceiling rises from 0.65 to 0.71–0.74 IoU), or give the mask prototypes a higher resolution (stride 2 instead of 4); inference at a larger size without retraining does not help (experiment 15);
    - set `nc=1` so the head carries no dead classes.
4. **Deploy `weights/eel_v2.pt` with peak suppression and NMS 0.5** (`experiments/eel_v2_infer.py`). On the held-out half it adds 5 points of recall and 3.5 points of box mAP50-95. It costs about 4% more parameters (3.18M). If an unchanged checkpoint is required, use peak suppression alone on `weights/eel_1cls.pt`. It needs no new weights and gives most of the gain (held-out fitness 1.3881 against 1.3346).
5. **If fine-tuning on new footage, interpolate afterwards.** Score α = 0.1–0.5 between the old and new weights on held-out frames (`experiments/wise_ft.py`). Do not ship the raw fine-tune.
6. **If mask boundary pixels matter more than IoU** (for example, for eel width or area measurements), add the refiner: `load_eel_v2("weights/eel_v2.pt", refiner="weights/mask_refiner.pt")`. It removes the 14% area excess. It costs about 60 ms per eel on one CPU thread. It does not raise mask mAP.
7. **Fix the data plumbing:** rename `imges/` to `images/` and give `data.yaml` a real validation split by time or by recording, not by random frame.

## Reproducibility

Everything lives in the project folder `cnn-weight-edit/`, runs with `uv run python <script>` in a uv environment (Python 3.12, CPU PyTorch 2.15, Ultralytics 8.4.22), and long jobs run in tmux via `experiments/run_queue.sh`. Report images are in `report_assets/`.

| Script | Purpose | Output |
| --- | --- | --- |
| `experiments/evaluate.py` | Ultralytics val on a dataset yaml | `results/baseline_22cls.txt`, `results/pruned_1cls.txt` |
| `experiments/prune_to_eel.py` | Slice the head to class 21 | `weights/eel_1cls.pt` |
| `experiments/compare_preds.py` | Frame-by-frame equality check of two checkpoints | Console, `results/ablate_s32.log` |
| `experiments/diagnose.py` | Per-eel found/missed, size, stride, score (any input size) | `results/diag_*.csv`, `results/diag_imgsz*.log` |
| `experiments/stride8_features.py` | Cache the 64 stride-8 features per anchor | Scratch caches (about 350 MB each) |
| `experiments/stride8_edit_search.py` | Score bias and LDA edits on the cache, time split | `results/stride8_edit_search.txt` |
| `experiments/stride8_probe_ceiling.py` | Best-linear-row ceiling test | `results/stride8_probe_ceiling.txt` |
| `experiments/ablate_stride32.py` | Silence the stride-32 eel row | `weights/eel_1cls_no_s32.pt` |
| `experiments/mask_res_ceiling.py` | Half-pixel mask IoU ceiling | `results/mask_ceiling.log` |
| `experiments/robustness.py` | Perturbation suite | `results/robustness_baseline.json` |

v2 scripts (results in `results/v2/`, one JSON line per evaluation in `runs.jsonl`, tables via `experiments/runs_table.py <filter>`):

| Script | Purpose |
| --- | --- |
| `experiments/fp_analysis.py`, `fp_sheet.py` | False-positive census and contact sheets |
| `experiments/eval_config.py` | Evaluation harness: head choice and mixing, peak suppression, flip TTA, box voting, post-NMS, logit shifts, mask threshold |
| `experiments/blend_heads.py` | Weight blend of the two heads (negative result) |
| `experiments/merge_heads_wide.py` | Exact widened output-average merge of the two class heads |
| `experiments/dump_dets.py`, `temporal_rescore.py` | Detection dump and temporal re-scoring (negative result) |
| `experiments/finetune.py` | Fine-tune on the first half, validate on the second |
| `experiments/wise_ft.py` | Weight interpolation between base and fine-tuned checkpoints, per parameter group |
| `experiments/eel_v2_infer.py` | Deployable inference: peak suppression and optional NMS |
| `experiments/queue_runner.sh` | File-based job queue (`results/v2/queue.txt`) run in tmux |
| `experiments/eval_config.py --native 1` | Mask mAP at full input resolution |
| `experiments/mask_metrics.py` | Mask IoU, boundary F-score, signed boundary distance, area ratio; guided filter |
| `experiments/mask_boundary_diag.py` | Full-resolution boundary diagnosis, threshold sweep, guided-filter variants |
| `experiments/mask_refine_boxes.py` | Mask crop with grown boxes (negative result) |
| `experiments/dump_mask_crops.py` | Per-eel crops (frame, mask logits, label, box) for the refiner |
| `experiments/train_mask_refiner.py`, `train_mask_refiner_net.py` | Train the boundary refiner (first half), score it (second half) → `weights/mask_refiner.pt` |
| `experiments/eval_mask_refiner.py` | Score the refiner with different output regions |

Rebuild `weights/eel_v2.pt`:
1. Fine-tune with `finetune.py --data split_train.yaml --epochs 6 --name ft_e6`.
2. Run `wise_ft.py weights/eel_1cls.pt results/v2/train/ft_e6/weights/last.pt weights/wise_a0.25.pt 0.25`.
3. Run `merge_heads_wide.py weights/wise_a0.25.pt weights/eel_v2.pt 0.5 --box-o2m --mask-o2m`.

Intermediate checkpoints are git-ignored.

Caveats and issues met along the way:

- **Training-set evaluation.** All v1 scores, and all v2 "full" and "subset" scores, are on training frames; read them as fit. The v2 held-out half was in the original model's training set too. Only the v2 fine-tune and interpolation kept it out. So the held-out numbers test the v2 changes, not the original model's generalization.
- **Dataset copies.** Because of the `imges/` spelling, frames were copied (1.1 GB) into a correctly named scratch folder, with a second copy of the labels remapped from class 21 to 0 for the one-class model. Those copies are temporary and outside the project.
- **Sample sizes.** The diagnostic used 1,003 eels; the edit search had 67 held-out target eels; the visual review covered 16 crops. Small-bin percentages (42 eels under 60 px) are noisy.
- **Job failures.** One caching job was killed for running out of memory and was rerun with chunked processing. Three chained wait-loops hung by matching their own command line and were stopped; no results depended on them.

## Sources

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming), Takehi, Teoh, Zhang and Khattab, MIT CSAIL, 2 November 2026
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089v2), Ilharco et al.
- [Task Arithmetic in the Tangent Space](https://arxiv.org/abs/2305.12827v3), Ortiz-Jimenez et al.
- [Ultralytics documentation](https://docs.ultralytics.com)
- [Robust fine-tuning of zero-shot models (WiSE-FT)](https://arxiv.org/abs/2109.01903), Wortsman et al.
- [Objects as Points (CenterNet)](https://arxiv.org/abs/1904.07850), Zhou et al.: the peak-suppression idea
