# Practical Work — Report Layout

Skeleton for the MSc Practical Work report, laid out against the IML grading guidelines
(targeting the **Excellent (1)** column of every row). Each section states its purpose,
a word budget, a content checklist, and the rubric criterion it satisfies.

**Hard constraints from the rubric**

| Constraint | Value |
|---|---|
| Length | 2 000–5 000 words, max 10 pages, excl. references / story summary / appendix |
| Introduction | min. 300 words |
| Limitations & conclusion | min. 300 words |
| Figures/tables | min. 4, all with titles, complete axes, legends, **error bars everywhere applicable** |
| Passing requirement | must pass *each* of Scientific Work (0.50), Scientific Documentation (0.20), Report (0.30) |
| Supervisor feedback | high-level feedback on the report **once** only — budget for it |

**Scope note (resolve before writing).** The repository currently trains on **Messidor-2
colour fundus** (972/246/527 images, 5 DR grades) with a frozen **DinoV3 ViT-S/16** encoder,
and the checkpoint in `retfound/` is `RETFound_mae_natureCFP.pth` (fundus weights). The
OCT path (`RETFound_mae_natureOCT.pth`, `data/OCT2017/`) is wired in
`scripts/precompute_reps.py` and `scripts/train.py` but no OCT data is present yet. This
outline is written for **OCT as the primary modality**, with the fundus runs framed as the
pilot / development phase. If OCT data does not materialise, swap the framing (fundus
primary, OCT as future work) — the section structure does not change.

---

## 0. Front matter

**Title (rubric 0.05 — precise, descriptive, draws attention to the experiments)**

Candidates, all naming *what* is probed, *how*, and *on what*:

1. *What Does a Retinal Foundation Model Encode? Probing SSL Representations of OCT Scans with Representation-Conditioned Generation*
2. *Seeing Through the Encoder: Representation-Conditioned Diffusion as a Visual Probe of SSL Features in Retinal OCT*
3. *Generative Probing of Self-Supervised OCT Representations for Downstream Disease Classification*

Pick one and keep it consistent with the abstract's first sentence.

**Abstract (~150–200 words, not counted against the rubric rows but read first)**
Problem → method in one sentence (RCDM-style representation-conditioned generation with a
ViT/flow-matching denoiser) → what was measured (encoding fidelity, invariance,
dimension-level structure) → headline numbers → what it implies for downstream OCT
classifiers.

---

## 1. Introduction — min. 300 words (rubric weight 0.10)

**Target: 450–550 words.**

Checklist for the *Excellent* column ("comprehensive background that leads seamlessly into
the experiments; objectives clearly and thoroughly defined; relevance, importance and
potential impact compellingly explained"):

- [ ] **Clinical framing (1 para).** OCT is the workhorse modality for retinal disease;
      labelled OCT at scale is expensive and grader-dependent, so foundation models
      (RETFound, DINOv2/v3-family) are pre-trained self-supervised and then frozen and
      linear-probed / fine-tuned for grading tasks.
- [ ] **The problem (1 para).** When such a frozen encoder underperforms on a downstream
      grading task, we cannot tell *why*: is the pathology signal absent from `h`, or
      present but not linearly decodable? Standard tools (linear probes, saliency,
      t-SNE/UMAP) answer this only indirectly — a probe reports *decodability*, not
      *content*.
- [ ] **The idea (1 para).** Bordes et al. (RCDM) showed that a generative model
      conditioned on `h` inverts the encoder: sampling several images from one `h` makes
      *encoded* content visible (what stays constant) and *discarded* content visible
      (what varies). Transfer this from ImageNet/ResNet-50 to retinal OCT and a
      medical foundation encoder.
- [ ] **Objectives, stated as explicit research questions.** Use the same numbering
      throughout the report (RQ1–RQ4 in §3) so Methods, Results and Conclusion align.
- [ ] **Why it matters / impact (short para).** A visual, annotation-free audit of what a
      frozen retinal encoder keeps and drops — informing encoder choice, augmentation
      design, and where a downstream classifier's ceiling actually comes from.
- [ ] **Contributions, bulleted (3–4 items).** e.g. (i) a modernised representation-
      conditioned generator (ViT denoiser + flow matching + adaLN-Zero) for retinal
      images at 224 px trained on <1k images; (ii) a quantitative encoder-inversion
      fidelity protocol; (iii) an invariance audit under clinically-motivated transforms;
      (iv) a dimension-level manipulation study linking representation axes to visible
      structure.
- [ ] Cite densely here (rubric 0.05 for referencing): RETFound, MAE, DINOv2/v3, RCDM,
      RCG, MAGE, JiT, DiT/adaLN-Zero, flow matching, CFG, FID/IS/KID.

**Do not** put method detail here. One forward-reference sentence ("Section 3 describes …")
is enough.

---

## 2. Background / Related Work (~400–600 words)

Keep it short — the seminar paper already covers the field; the report is graded on the
experiments, not the survey. Fold into the Introduction if space is tight.

- [ ] SSL for retinal imaging: MAE pre-training, RETFound (ViT-L/16, 1.6 M retinal images,
      separate CFP and OCT weights), what linear probing does and does not tell us.
- [ ] Representation-conditioned generation: RCDM (encoder inversion as a probe) → RCG
      (representations as a generative prior) → MAGE (generation and representation
      quality co-improve). One or two sentences each; cite, don't re-explain.
- [ ] Denoiser/generator background: diffusion vs. flow matching, ViT denoisers (JiT),
      adaLN-Zero conditioning, classifier-free guidance.
- [ ] **Gap statement (2–3 sentences).** RCDM-style probing has been demonstrated on
      natural images with CNN encoders. It has not been applied to a medical foundation
      encoder in the low-data regime typical of clinical datasets — which is what this
      work does.

---

## 3. Research questions and hypotheses (~150–250 words, or a boxed list in §1)

State these once, explicitly, and reuse the numbering everywhere. Each RQ must map to at
least one experiment in §4 and one figure/table in §5.

| RQ | Question | Hypothesis | Experiment | Primary metric |
|---|---|---|---|---|
| **RQ1** | Is a frozen retinal SSL representation `h` invertible — can a generator reconstruct the scan it came from? | `h` retains enough structure for faithful, scan-specific reconstruction | E1 (fidelity) | MRR / mean rank, FID, ± CI |
| **RQ2** | What does `h` encode vs. discard? | Global anatomy and pathology stage are stable; fine texture/position varies across samples | E2 (multi-sample variance) | per-`h` sample variance (LPIPS/SSIM), qualitative grids |
| **RQ3** | Which transformations is `h` invariant to? | Encoder is invariant to flips/crops it saw as augmentations, but not to scale, greyscale, or intensity — clinically relevant, since those correlate with device/protocol | E3 (invariance probe) | cosine sim of `h` vs. transform, + generated-sample agreement |
| **RQ4** | Is the representation structured — do individual dimensions carry separable, editable factors? | Masking/substituting the dimensions shared across a nearest-neighbour set removes/transplants a specific factor | E4 (dimension masking / substitution) | qualitative + downstream probe delta |
| **RQ5** *(bridge to the stated goal)* | Do the findings predict downstream classifier behaviour? | Information visibly absent from generations is also information a linear probe cannot recover | E5 (linear probe under ablated `h`) | balanced acc. / AUC, mean ± 95% CI |

RQ5 is what turns this from "pretty pictures" into the stated motivation ("improve
predictive models later on"). Keep it even if the result is negative — the rubric
explicitly rewards a properly executed hypothesis that does not work out.

---

## 4. Methods & experiments (rubric weight 0.20 — the largest single report row)

**Target: 900–1 200 words.** *Excellent* requires publication-ready detail, a logical
order, full reproducibility, experiments that thoroughly address the RQs, **and detailed
hardware + precise compute time**.

### 4.1 Data
- [ ] Dataset, source, licence, preprocessing. OCT: OCT2017 (Kermany) — 4 classes
      (CNV / DME / DRUSEN / NORMAL); state exact split sizes as used.
      Fundus pilot: Messidor-2, 972 train / 246 val / 527 test, 5 DR grades
      (`anodr` 568, `bmilddr` 151, `cmoderatedr` 193, `dseveredr` 41, `eproliferativedr` 19).
- [ ] **Report the class imbalance explicitly** and how it is handled — the fundus tail
      classes have 19–41 training images, which materially limits what the generator can
      learn about advanced pathology. This belongs in Methods *and* Limitations.
- [ ] Split protocol: patient-level or image-level? State it; if image-level, flag the
      leakage risk (Messidor-2 has two eyes per patient).
- [ ] Preprocessing chain: `Resize(224) → CenterCrop(224) → ToTensor →
      Normalize(ImageNet mean/std)` for the encoder; `[-1, 1]` for the generator. Say
      explicitly that these two normalisations are kept separate and never mixed.
- [ ] Encoder input is **fixed at 224 px** regardless of generator resolution (14×14
      positional grid), and why.

### 4.2 Encoders (the objects under study)
Table of every encoder probed — this is the independent variable of the whole study:

| Encoder | Pre-training | Arch | `h` | Role |
|---|---|---|---|---|
| RETFound MAE | MAE, 1.6 M retinal images (OCT weights) | ViT-L/16 | 1024 (CLS) | primary — domain foundation model |
| DINOv3 ViT-S/16 | DINO-style joint-embedding | ViT-S/16 | 384 (CLS) | contrastive/joint-embedding contrast |
| ResNet-50 (ImageNet, supervised) | supervised | CNN | 2048 (avgpool) | out-of-domain / supervised control |

The comparison **MAE-style vs. joint-embedding vs. supervised** is the scientific spine:
it is what lets the report say something general about SSL representation spaces rather
than about one checkpoint. Note in the text that CLS-token extraction differs by API
(`last_hidden_state[:, 0]` for HF, `forward_features(x)[:, 0]` for timm) and that all
encoders are frozen (`eval()`, `requires_grad=False`) and their representations
precomputed once (`scripts/precompute_reps.py` → `{"paths", "reps", "encoder", "h_dim"}`,
index-aligned).

### 4.3 Generator (JiT-RCDM)
Describe as *"RCDM's conditioning idea on a modern denoiser"*, and be explicit that this
is a re-implementation choice, not a novelty claim:

- [ ] Conditioning path: `h → Linear(h_dim → cond_dim) + SiLU → h_proj`;
      `c = h_proj + t_emb`; broadcast to all blocks via **adaLN-Zero**
      (6 modulation vectors per block, zero-init output ⇒ identity at step 0);
      applied 25× per forward pass (12 blocks × 2 + final layer).
- [ ] Why adaLN-Zero instead of RCDM's conditional BatchNorm (cBN modulates
      `(B,C,H,W)` feature maps; a ViT has `(B,N,D)` token sequences).
- [ ] Denoiser: JiT ViT — patch embed, RoPE, qk-norm, RMSNorm, SwiGLU.
      Preset table (`JiT_S_16`: hidden 384, depth 12, heads 6, patch 16, cond_dim 128,
      196 tokens @224, ~25 M params; `JiT_S_32` for fast local runs).
- [ ] Objective: flow matching, **x-prediction**, `t ~ sigmoid(N(-0.8, 0.8))`,
      `z_t = t·x + (1−t)·ε`, `L = MSE(model(z_t, t, h), x)`.
- [ ] Sampler: 50-step Heun ODE, CFG with a **learnable `null_h` parameter**
      (trained via `p_uncond` dropout, included in EMA), CFG blending at the `x_pred`
      level.
- [ ] Training details: AdamW (β₂ = 0.95), LR, linear warmup, batch size, total steps,
      EMA decay 0.9999, precision, augmentation (h-flip only, justified by encoder flip
      invariance — and verify this claim in E3 rather than asserting it).
- [ ] **Design choices that are ours, flagged as such**: `cond_dim = 128` bottleneck
      (regularisation for <1k images), model presets, learnable `null_h`. The rubric
      rewards *reasoning*; give one sentence of justification each, and where an
      ablation exists, point to it.

### 4.4 Experiments
One subsection per RQ. Each must state: inputs, procedure, number of repetitions/seeds,
and the metric with its uncertainty estimate.

- **E1 — Encoder-inversion fidelity (RQ1).** Generate 1 image per test representation;
  re-encode; rank all generated representations by distance to each conditioning `h`;
  report **mean rank and MRR** (RCDM's protocol) plus FID against real images. Run for
  every encoder in §4.2. *Uncertainty:* bootstrap CI over the test set for MRR;
  ≥3 sampling seeds for FID (report mean ± std) — a single FID number will cost the
  "error bars are systematically shown" criterion.
- **E2 — What is encoded vs. discarded (RQ2).** N images (stratified over classes),
  k samples per `h`, **fixed vs. resampled noise**. Quantify inter-sample variability
  (pixel std, SSIM/LPIPS between samples of the same `h`) instead of relying only on
  visual grids — this is the single biggest lever from "Satisfactory" to "Excellent".
- **E3 — Invariance probe (RQ3).** Deterministic transform set already implemented in
  `scripts/invariance_images.py`: crop128, zoom×2, rot90, mirror-v, mirror-h, translate,
  greyscale (+ a noise variant). Two levels of measurement, both already in
  `scripts/invariance_probe.py`: (a) **backbone invariance** = cosine similarity
  `⟨h(T(x)), h(x)⟩`; (b) **generator-visible invariance** = generate with the **same
  starting noise** across variants so all visual change is attributable to `h`
  (`invariance_probe_noise.py` is the uncontrolled counterpart — state why both exist).
  *Scale it up:* run over ≥30 images per class, not 1, so the cosine similarities become
  distributions with CIs, and test encoder differences with a paired Wilcoxon
  signed-rank test + Holm–Bonferroni correction across transforms.
  Add a clinically-motivated transform (intensity/contrast shift, speckle noise, or
  B-scan crop for OCT) — this is where the medical framing earns its keep.
- **E4 — Dimension-level structure (RQ4).** k-NN in representation space → dimensions
  shared across the neighbourhood → (a) zero them, (b) substitute them from another
  scan; generate and describe what survives. Report on ≥10 query scans so it is not
  anecdotal, and state the heuristic's assumptions.
- **E5 — Downstream link (RQ5).** Linear probe (and/or k-NN classifier) on frozen `h`
  for the grading task, evaluated on: full `h`, `h` with the E4-identified dimensions
  masked, and `h` from transformed inputs found non-invariant in E3. ≥5 seeds,
  report **mean ± 95% CI**, compare conditions with a paired test (McNemar on the same
  test set, or a paired t-test/Wilcoxon over seeds). This is the row that connects the
  probing work to "improve predictive models later on".
- **Ablations (supporting, not headline).** CFG scale sweep (runs exist for 1.0–3.0 —
  report FID and fidelity vs. CFG, not just sample grids), `cond_dim` 64 vs. 128,
  patch 16 vs. 32, training-length effect (55k vs. 100k steps).

### 4.5 Implementation, hardware and compute — **do not omit**
The *Excellent* column explicitly demands "detailed and comprehensive description of the
hardware (e.g. GPU, CPU) and precise compute time". Fill this table from
`wandb/` run metadata and checkpoint timestamps rather than estimating:

| Run | Encoder / dataset | Preset | Steps | Batch | Hardware (GPU, VRAM, CPU) | Wall-clock | GPU-hours |
|---|---|---|---|---|---|---|---|
| main | | `JiT_S_16` | 100 000 | | | | |
| ablation: cond_dim 64 | | | | | | | |
| ablation: patch 32 | | `JiT_S_32` | | | | | |
| sampling / eval | | | | | | | |
| **Total** | | | | | | | |

Also state: framework + versions (PyTorch, timm, transformers), seeds, precision, and the
total project compute including failed runs (the `16_50k_wrong_h` run is worth one honest
sentence — a debugging finding reported transparently reads as good research practice).

---

## 5. Results & interpretation (rubric weight 0.20)

**Target: 900–1 200 words.** *Excellent* = "data interpreted perfectly, results very well
described, the story is appealing, relation to introduction/aim/broader context of high
quality". Structure it as one subsection per RQ, in the same order as §3/§4.

- [ ] **5.1 Fidelity (RQ1).** Table 1 + Figure 1. Lead with the number, then the reading.
      Compare encoders; compare against RCDM's ImageNet MRR values as external context.
- [ ] **5.2 Encoded vs. discarded (RQ2).** Figure 2 grids + the variability metric.
      Say concretely *which* clinical structures are stable across samples and which are
      not (e.g. retinal layer topology vs. speckle texture; lesion presence vs. exact
      lesion count/position). If the generator cannot render fine lesions at all,
      say so plainly and attribute it (generator capacity vs. representation content —
      and explain how E1/E2 disentangle those two explanations).
- [ ] **5.3 Invariance (RQ3).** Figure 3: cosine similarity per transform per encoder,
      **with error bars**, plus paired-test results. This is likely the most quotable
      result: which nuisance factors survive into the representation, and therefore which
      ones a downstream classifier can silently latch onto.
- [ ] **5.4 Dimension structure (RQ4).** Figure 4 masking/substitution grids.
- [ ] **5.5 Downstream (RQ5).** Table 2. Does the visual finding predict the probe result?
      State agreement *and* disagreement.
- [ ] **5.6 Ablations.** Compact table; CFG scale, `cond_dim`, patch size, training length.

Writing rules for this section: every claim is anchored to a figure/table number; every
number carries an uncertainty; interpretation is separated from description (one
"what we see" sentence, then one "what it means" sentence); each subsection closes by
tying back to the RQ and, where relevant, to what it implies for a clinical classifier.

### Figure & table plan (rubric 0.20 — min. 4 figures/tables)

| # | Type | Content | Error bars? |
|---|---|---|---|
| Fig. 1 | Pipeline schematic | scan → frozen encoder → `h` → adaLN-Zero → ViT denoiser → samples; mark what is frozen vs. trained | n/a |
| Fig. 2 | Qualitative grid | conditioning scan ∥ k generated samples, one row per class; fixed noise | n/a (caption states k, CFG, steps, seed) |
| Fig. 3 | Bar/box plot | cosine sim of `h` under each transform, grouped by encoder | **yes** — CI over images |
| Fig. 4 | Qualitative grid | dimension masking and substitution | n/a |
| Fig. 5 | Line plot | FID / MRR vs. CFG scale (and/or vs. training step) | **yes** — over seeds |
| Table 1 | Quantitative | per-encoder FID, IS/KID, mean rank, MRR ± CI | **yes** |
| Table 2 | Quantitative | linear-probe performance under `h` ablations, mean ± 95% CI, p-values | **yes** |
| Table 3 | Setup | hardware + compute per run (may live in Methods) | n/a |

Caption style: self-contained (a reader who sees only the figure understands it), stating
n, what the error bars are (SD / SEM / 95% CI — say which), and the sampling settings.

---

## 6. Limitations & conclusion — min. 300 words (rubric weight 0.10)

**Target: 450–600 words.** *Excellent* = "all major limitations discussed in-depth;
conclusion strong, concise, and logically derived".

Limitations to cover honestly (all of these are real for this project):

- [ ] **Data scale.** <1 000 training images for a generative model; tail classes with
      19–41 examples. Generative quality is a confounder for every qualitative claim.
- [ ] **The inversion confound.** A missing feature in a sample could mean "not in `h`"
      *or* "the generator cannot render it". State how E1/E2/E5 bound this, and that the
      bound is partial.
- [ ] **One generator per encoder.** Cost scales linearly with the number of
      representations studied (RCDM's own limitation), so the encoder comparison is
      necessarily small-n.
- [ ] **CLS-token-only conditioning.** Patch tokens are discarded; conclusions are about
      the global representation, not the full feature field.
- [ ] **Modality/weights mismatch** if fundus weights are used on OCT or vice versa —
      state exactly which checkpoint was used where.
- [ ] **Metric caveats.** FID with an ImageNet Inception backbone is poorly calibrated for
      medical images (report KID and/or a domain-encoder FID alongside, and say why).
- [ ] **Probing heuristic.** k-NN dimension selection is simple and may not transfer
      across encoders/datasets.
- [ ] **No clinical validation.** No ophthalmologist reader study; "looks like a lesion"
      is not "is a lesion".

Conclusion: 2–3 paragraphs, no new results. What was asked (RQ1–RQ5), what was found,
what it means for building predictive models on frozen retinal encoders, and 2–3 concrete
next steps (patch-token conditioning; representation-space sampling à la RCG for
augmentation of rare grades; reader study; scaling to full OCT2017).

---

## 7. References (rubric 0.05)

- [ ] One consistent style (the seminar paper's ACM-style is fine — reuse the `.bib`).
- [ ] Every method/dataset/metric claim cited; no uncited numbers from other papers.
- [ ] Minimum set already in the seminar bibliography: Bordes 2022 (RCDM), Li 2024 (RCG),
      Li 2023 (MAGE), He 2021 (MAE), Balestriero 2023, Dhariwal & Nichol 2021,
      Ho 2020, Dosovitskiy 2021, Heusel 2018, Salimans 2016.
- [ ] To add: RETFound (Zhou et al., *Nature* 2023), OCT2017 (Kermany et al. 2018),
      Messidor-2 (Decencière et al. 2014 / Abràmoff et al. 2013), DINOv2/v3,
      JiT (Li et al. 2025), DiT/adaLN-Zero (Peebles & Xie 2023), flow matching
      (Lipman et al. 2023), CFG (Ho & Salimans 2022), KID (Bińkowski et al. 2018).

---

## 8. Appendix (not counted in the page limit; supervisors need not read it)

Move here anything that would push the main text past 10 pages:
extra qualitative grids, per-class breakdowns, full hyperparameter dumps, failed-run notes,
exact CLI invocations, additional ablations.

---

## 9. Scientific Documentation — the repository (rubric weight **0.20**, own passing grade)

This is graded separately and is fully in your control. Checklist for the *Excellent*
column ("runs easily, produces the **exact** results reported, exceptional documentation,
best programming practices, high modularity, meticulous data instructions, thorough
environment setup"):

- [ ] **README** covering: what the project is → environment setup → data acquisition →
      exact commands to reproduce **each numbered figure and table** in the report.
      A `Reproducing the report` section mapping `Figure 3 → python scripts/invariance_probe.py …`
      is the single highest-value addition.
- [ ] **Pinned dependencies** — `requirements.txt` / `environment.yml` with versions
      (torch, timm, transformers, torchvision, pytorch-fid, wandb). None exists yet.
- [ ] **Data instructions** — Messidor-2 and OCT2017 are access-controlled; document how
      to request them, the expected directory layout, and the preprocessing script.
      Same for RETFound / DINOv3 weights (gated downloads; document the
      `use_gated_mlp: false` config fix, which silently changes results if missed).
- [ ] **Seeds and determinism** — every script takes `--seed`; record seeds in checkpoints
      and in figure metadata.
- [ ] **Tidy the tree before submission**: consolidate the work currently split across
      `main`, `claude/retfound-encoder` and the `.claude/worktrees/` copies (the
      invariance-probe scripts — the core novelty of the study — currently live only in a
      worktree branch); remove or archive `guided_diffusion/` if the JiT path is the only
      one used; add a `.gitignore` for `wandb/`, `checkpoints/`, `data/`, `__pycache__`,
      `.DS_Store`.
- [ ] **Keep `CHANGES.md`** — the explicit upstream-attribution log (what came from RCDM,
      what from JiT, what is ours) is exactly the "best practice" evidence the rubric
      rewards. Update it for the RETFound/OCT path.
- [ ] Checkpoint/artifact availability: state where the trained weights live, or that they
      are available on request, with sizes.

---

## 10. Scientific Work — evidence to surface (rubric weight **0.50**)

Half the grade is on the research process, not the text. Make the evidence visible:

- [ ] **Reproducibility & statistics** (0.40 sub-row, *Excellent* = "comprehensive
      reproducibility measures and statistical significance testing"). Concretely for this
      project: ≥3 seeds for every generative metric; bootstrap CIs for MRR; ≥30 images per
      class in the invariance probe with paired Wilcoxon + Holm–Bonferroni; ≥5 seeds for
      the downstream probe with McNemar between conditions. **Decide the tests before
      running the experiments** and state them in Methods.
- [ ] **Project management** (0.40 sub-row). The change log, the branch/worktree history,
      and the ablation sequence (cond_dim 64 → 128, S32 → S16, the `16_50k_wrong_h`
      debugging run) already document an organised process — reference it in the report's
      experimental-design narrative rather than hiding the false starts.
- [ ] **Motivation & independence** (0.20 sub-row). Log supervisor interactions and what
      changed as a result; remember the report itself gets **one** round of high-level
      feedback, so send it only once the structure below is complete.

---

## 11. Suggested writing order and length budget

| Order | Section | Words | Why this order |
|---|---|---|---|
| 1 | §3 RQs + §4.4 experiment table | 300 | Freezes the study design; everything else follows |
| 2 | §4 Methods | 1 000 | Written from the code, while it is fresh |
| 3 | §5 Results + all figures | 1 100 | Figures first, prose second |
| 4 | §1 Introduction | 500 | Written last-but-one so it promises exactly what was delivered |
| 5 | §6 Limitations & conclusion | 500 | |
| 6 | §2 Background | 500 | Trim first if over length |
| 7 | Abstract + title | 200 | |
| | **Total (main text)** | **~4 100** | within 2 000–5 000, ~9–10 pages with figures |

---

## 12. Open items to resolve before drafting

1. **Modality decision** — OCT2017 + `RETFound_mae_natureOCT.pth`, or stay on Messidor-2
   fundus + `RETFound_mae_natureCFP.pth`? Currently only the fundus data and CFP weights
   are on disk. This decides the title, dataset section, and the clinical framing.
2. **Encoder set** — is the RETFound vs. DINOv3 vs. supervised-ResNet-50 three-way
   comparison in budget (one generator per encoder ≈ one full training run each)? If not,
   drop to two and say so in Limitations.
3. **RQ5 scope** — is the downstream probe part of this report or explicitly future work?
   Recommended: keep it, even as a small experiment, since it is the stated motivation.
4. **Compute accounting** — recover wall-clock and GPU type per run from `wandb/` before
   those runs are pruned.
5. **Statistics plan** — fix the tests and the number of seeds now; retrofitting error
   bars after the fact is what costs the "Excellent" grade on two separate rubric rows.
