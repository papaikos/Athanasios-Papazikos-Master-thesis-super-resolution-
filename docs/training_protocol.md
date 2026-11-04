# Agent handoff: one D3 SwinIR confirmation run on a standard L4

**Protocol date:** 9 October 2026.  
**Purpose:** implement, verify and run one scientifically traceable RGB x4 SwinIR training experiment on the expanded D3 dataset.  
**Status of this document:** execution plan, not evidence that a VM or training run already exists.

## 1. Human decisions and scope

The user explicitly chose:

- One model, **classical SwinIR RGB x4**, trained **from scratch** on **D3 only**.
- One **non-Spot, STANDARD NVIDIA L4 VM**.
- **100,000 optimizer updates**, using one training seed, with metrics throughout training.
- A **120-hour maximum covering the entire VM lifecycle**, including setup, staging, training, evaluation, export and cleanup.
- Save and verify outputs, then **delete the VM and its disposable disks**. Finish early if the target is reached; do not keep a completed VM running until the deadline.
- The agent initially **babysits actual training**, then ends its conversation once the unattended run is proven healthy.
- After that, **check progress every six hours and persist a report**. The user will inspect completion and return to the conversation later.
- **The user alone manages money. Never inspect billing, budgets, credits, spending, prices, invoices, budget alerts or cost estimates. Never make a monetary gate or ask for budget approval.** Time and resource-health controls are the agent's responsibility.

These decisions supersede earlier A100, three-model, multi-seed and 300,000-update proposals. This is a distinct **100k-update confirmation protocol**, not the final three-condition thesis comparison. Do not train D1/D2, start other substantive model fits, use pretrained initialization, switch to Spot, extend the deadline, or inspect financial information.

When the user instructs you to execute this handoff, carry out the implementation, verification, deployment and initial supervision without routine permission handoffs. Resolve ordinary reversible implementation choices yourself. An actual missing credential, permission or unavailable required service is a blocker; document it precisely. Reading this document by itself is not evidence that execution has occurred.

## 2. Scientific objective and success definitions

Demonstrate that the complete D3 pipeline trains correctly and produces reproducible learning curves and validation-selected checkpoints on the current release.

Distinguish these outcomes:

1. **Engineering success:** correct data, finite updates, working metrics, exact sample accounting, resumable checkpoints, durable outputs, functioning supervision and cleanup.
2. **Learning evidence:** full validation reconstruction metrics improve relative to initialization. Report the comparison with bicubic interpolation honestly, whether positive or negative.
3. **Target reached:** exactly 100,000 successful optimizer updates were completed.
4. **Time limited:** stopped before 100,000 because of the fixed cutoff; preserve the actual update count and original schedule.
5. **Scientific limitation:** one seed and a bounded run do not establish convergence, architecture superiority, cross-degradation superiority or training-seed uncertainty.

Never redefine success after seeing results. Failure to beat bicubic is a result to investigate, not a reason to select a favorable metric or hide checkpoints. Do not stop early merely because a score looks good.

## 3. Inputs and provenance

### 3.1 Local sources

Original handoff:

`<archive>/Maritime_RGB_Training_Agent_Handoff.md`

Expanded release:

`<archive>/Maritime_RGB_Expanded_20261008`

Relevant delivered files under that root:

- `control/training-index.json`: wrapped READY index.
- `control/release-receipt.json`, `control/download-receipt.json`.
- `control/code-delivery/`: scientific code identity receipts.
- `control/cloud-cleanup-receipt-20261009.json`.
- `code/src/maritime_uint8_dataset.py`: current paired loader.
- `code/src/maritime_uint8.py`: quantization and storage hashes.
- `code/src/maritime_recipes.py`: D3 degradation definition.
- `code/scripts/maritime_uint8_download.py`: verified staging.
- `code/scripts/maritime_uint8_training_smoke.py`: shard selection and masked L1 helper only.

Keep this delivered package unchanged. Build the research trainer and infrastructure in a separate versioned project. Do not reuse `src/swinir_train.py` unchanged: it uses legacy readers/losses and constrains a historical run to 2,000 updates. The tiny engineering smoke network is not the research model. Its held-out CLI options are not a valid evaluator.

### 3.2 Cloud locations

```text
Project: my-gcp-project
Bucket: gs://my-maritime-rgb-bucket/
Release: maritime-rgb/expansion-20261007/releases/maritime-uint8-expanded-20261008-v1/
New outputs: maritime-rgb/training-runs/<UTC>-swinir-d3-s42-confirmation-100k/
```

Generate a unique run ID and output prefix; do not overwrite another run. Existing source, Mother, SkySat, historical checkpoints and release/control objects are protected. Never delete the shared bucket, sweep prefixes, or restore deleted construction intermediates.

Save this exact input pin as `uint8-index-pin.json` in the new run's configuration directory:

```json
{
  "uri": "gs://my-maritime-rgb-bucket/maritime-rgb/expansion-20261007/control/final-delivery-wrapper-v1/delivery-index-0888d9772668b6cfe3b36f952dbffdaede4f887bf59dfff561b5b8903829c6bc.json",
  "generation": "1791488399208942",
  "bytes": 8075,
  "crc32c": "+K9b5w==",
  "sha256": "0888d9772668b6cfe3b36f952dbffdaede4f887bf59dfff561b5b8903829c6bc"
}
```

D3 main manifest identity:

```json
{
  "uri": "gs://my-maritime-rgb-bucket/maritime-rgb/expansion-20261007/releases/maritime-uint8-expanded-20261008-v1/d3/manifest.json",
  "generation": "1791487695855602",
  "bytes": 824241040,
  "crc32c": "g8vf9Q==",
  "sha256": "ac3d1a0df5fa93fe258fcde82eed6922905cc827ddaab7050dcda227fd859528"
}
```

Frozen scientific code recovery pin, if transferring the local verified package is unavailable:

```json
{
  "uri": "gs://my-maritime-rgb-bucket/maritime-rgb/expansion-20261007/control/uint8-final-runtime-v2/code-10e0a5a964670f7b5d028fbed9114a3d55db63b1ac00d8d582ab40c7833c3203.tar.gz",
  "generation": "1791452627086300",
  "bytes": 516872,
  "crc32c": "NuUXfQ==",
  "sha256": "10e0a5a964670f7b5d028fbed9114a3d55db63b1ac00d8d582ab40c7833c3203"
}
```

Read the wrapped index and completed receipts for dependent pins. Verify generation, size, CRC32C and SHA-256 when staging. The retained scientific CPU image is not a CUDA training environment.

### 3.3 Dataset contract

| Property | Required value |
|---|---|
| Main TRAIN | 17,952 D3 identities |
| Main validation | 115 D3 identities, all Maxar |
| Main test | 127 identities; sealed for this confirmation run |
| Separate aerial validation | 15 identities; final diagnostic only |
| Separate aerial test | 48 identities; sealed |
| Stored arrays | uint8 RGB |
| Loader arrays | float32 RGB in [0,1], already divided by 255 |
| LR full image | [3,128,128], 10 m sampling |
| HR full image | [3,512,512], 2.5 m sampling |
| Scale | x4 |
| Support masks | Boolean, fully true in this release; valid support, not ship/cloud labels |
| D3 recipe | d3-variable-psf-noise-v1, fixed stored realization |

Do not divide by 255 twice, fit a per-patch color stretch, create new LR, resample noise each epoch, concatenate Mother, or add historical datasets. Preserve the release's recipe seed 20261006 separately from training seed 42.

Use TRAIN for all optimizer/preflight/overfit tests. Main validation selects checkpoints. Aerial validation is diagnostic only. **Do not run main or aerial test metrics, visual selection or training on this confirmation run.** Preserve those tests for the later frozen thesis comparison; metadata-only leakage checks are allowed.

## 4. Build the research package before allocating the GPU

Prepare as much as possible locally first: source, dependencies, unit tests, lifecycle controller, packaging and a launch plan. Do not consume the VM lifetime writing a trainer from nothing if that can be done locally.

Recommended separate project structure; these are files to implement, not existing delivered entry points:

```text
research/
  configs/d3_swinir_100k.yaml
  pins/uint8-index-pin.json
  data_adapter.py
  model_factory.py
  metrics.py
  train.py
  evaluate.py
  checkpointing.py
  runtime_control.py
  reporting.py
  finalize.py
  tests/
infra/
  provision.py
  cleanup_controller.py
  startup.sh
  systemd/
  tests/
```

Pin code revisions and dependencies in a lock file/container digest. Use a tested Linux Python/PyTorch/CUDA combination with BF16 on L4; resolve official package compatibility before launch and save exact versions. Do not silently upgrade packages during a run. A previously successful CPU or older GPU run does not validate this environment.

Prepare an authenticated managed cleanup/progress controller outside the VM. Its lifecycle and reporting duties are described below. Test it with fake resource identities and a short-lived non-destructive fixture before trusting it with the training VM. No extra GPU is needed for this controller.

## 5. Exact model and training protocol

### 5.1 Architecture

Use official classical SwinIR, pinned initially to:

`https://github.com/JingyunLiang/SwinIR/tree/6545850fbf8df298df73d81f3e8cba638787c8bd`

```yaml
experiment: maritime-d3-swinir-confirmation-100k-v1
seed: 42
initialization: random_from_scratch
model:
  in_chans: 3
  upscale: 4
  img_size: 64
  patch_size: 1
  window_size: 8
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2.0
  qkv_bias: true
  qk_scale: null
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
  ape: false
  patch_norm: true
  use_checkpoint: false
  img_range: 1.0
  upsampler: pixelshuffle
  resi_connection: 1conv
optimizer:
  name: Adam
  lr_peak: 0.0002
  betas: [0.9, 0.99]
  eps: 0.00000001
  weight_decay: 0.0
training:
  target_updates: 100000
  warmup_updates: 2000
  min_lr: 0.000001
  effective_batch_size: 16
  preferred_microbatch_size: 4
  preferred_accumulation_steps: 4
  lr_crop_size: 64
  hr_crop_size: 256
  loss: masked_l1
  precision: bf16_autocast_fp32_loss_and_master_parameters
  ema: false
  early_stopping: false
  gradient_clipping: false
evaluation:
  precision: fp32
  batch_size: 1
  full_frame: true
  self_ensemble: false
  tiled_inference: false
  primary_metric: mean_per_image_rgb_psnr
  prediction_clip: [0.0, 1.0]
  quantize_prediction: false
  extra_border_shave_hr_pixels: 0
```

Use LayerNorm as in the pinned implementation; record every resolved constructor argument and actual parameter count. Preserve the model's internal RGB mean subtraction/restoration `(0.4488, 0.4371, 0.4040)`. Do not load any pretrained weights. Save the initial state and hash before all preflight mutations; reset to it for the substantive run.

The YAML is a protocol specification. Implement schema validation and map it explicitly to actual constructors; do not assume an upstream CLI understands this file.

### 5.2 Schedule and sampling

The scheduler advances once per successful optimizer update, not per microbatch. For update number `u` in 1..100000, use this update's learning rate:

```text
u <= 2000:
  lr(u) = 2e-4 * u / 2000
u > 2000:
  lr(u) = 1e-6 + 0.5 * (2e-4 - 1e-6)
                   * (1 + cos(pi * (u - 2000) / 98000))
```

Test the first, warm-up-boundary and final rates. If time-limited, keep this original schedule; do not compress it retrospectively or label the run target-complete.

Shuffle TRAIN uniformly in deterministic permutations. Continue across permutation boundaries without dropping samples. Use random aligned LR 64x64 crops and matching HR 256x256 crops: LR origin `(x,y)` maps to HR `(4x,4y)`. Select uniformly among the eight square rotations/reflections and apply the same geometry to LR, HR and both masks. No color augmentation or extra degradation.

Use explicit independent RNG streams or stateless augmentation keyed to the consumed sample position. Record consumed positions, identity and crop/augmentation choices sufficiently to reproduce the next batch. Worker prefetch must not advance the checkpoint's authoritative consumed cursor.

Start with microbatch 4 and accumulation 4. During preflight only, lower to 2x8 or 1x16 if required by measured VRAM. Keep effective batch 16. Freeze the chosen combination before step 1. Accumulation must produce the correct batch-mean gradient. OOM during the substantive run is a fault to record, not permission to silently change the model or input size.

Use BF16 autocast for training with FP32 loss and optimizer state. BF16 normally does not need a gradient scaler; record this explicitly. Validate FP32 full-frame inference and mixed-precision training on the actual GPU. Any precision fallback must be settled and recorded before step 1. Set and record TF32 and deterministic-backend settings; require measured same-runtime resume behavior rather than claiming universal bitwise reproducibility.

At 100,000 updates and effective batch 16, exposure is 1,600,000 sampled crops, approximately 89.1 passes through TRAIN identities. These are crop exposures, not 89 complete full-image epochs.

## 6. Data staging and preflight gates

Stage D3 TRAIN and validation only. The wrapped index also supplies the separate aerial validation metadata/data. A supported downloader command, run from the frozen code root with it on PYTHONPATH, is:

```bash
python -m scripts.maritime_uint8_download \
  --index-pin /work/run/config/uint8-index-pin.json \
  --output /work/data/maritime-uint8-expanded-20261008 \
  --datasets d3 --splits train validation --plan-only

python -m scripts.maritime_uint8_download \
  --index-pin /work/run/config/uint8-index-pin.json \
  --output /work/data/maritime-uint8-expanded-20261008 \
  --datasets d3 --splits train validation
```

Do not use `--allow-format-only` or `--include-mother-native`. These VM paths are proposed paths; create them deliberately and keep the data read-only after verified staging.

Use `MaritimeUint8Dataset`, selected through manifest membership. The manifest is approximately 824 MB and expands in memory when parsed. Parse sequentially and release temporary structures; avoid retaining multiple copies in workers. Start loader diagnosis with zero workers, then benchmark a small worker count. If a compact selector/cache is necessary, preserve membership, pins and hash checks and prove equivalence; do not bypass validation to gain throughput.

Required gates before substantive training:

1. READY index and COMPLETE receipt agree; staged files and scientific code pass their pinned checks.
2. Exactly 17,952 main TRAIN, 115 main validation and 15 auxiliary validation identities; no sampled or incomplete validation masquerades as full validation.
3. Split identities are disjoint. Audit recorded HR hashes, source scene/site/capture IDs and footprints for leakage using available metadata. Record unknowns. Confirmed leakage blocks clean held-out claims; do not silently re-split or exclude samples.
4. Metadata join maps sample IDs to source/group without guessing from appearance. Main validation is Maxar; no claim of held-out SkySat/EarthView performance.
5. Loader tensors, masks, normalization and aligned crop transforms pass tests. Synthetic coordinate-grid tests catch x4 offsets and flips.
6. Actual model maps cropped TRAIN `[N,3,64,64]` to `[N,3,256,256]` and full validation `[1,3,128,128]` to `[1,3,512,512]`; loss, gradients and weights remain finite.
7. A bounded small-subset TRAIN overfit check lowers deterministic L1 on the same examples. Label it engineering-only and reset all model/optimizer/sampler/RNG state afterward.
8. Metric identity/perturbation tests and an independent reference comparison pass. Validation leaves model parameters, buffers and training RNG/sampler state unchanged.
9. A save/reload test restores the same next consumed samples, augmentation, learning rate and optimizer update within the declared numerical tolerance. Merely loading model weights is insufficient.
10. A real checkpoint is uploaded, read back and hash-verified; both deadlines and the external controller are tested before unattended execution.

Preflight must be bounded. Do not spend the 120 hours retrying an environment that never becomes ready. Preserve diagnostic evidence and clean up on an unrecoverable failure.

## 7. Metrics throughout training

### 7.1 Metric definitions

Training loss uses **unclipped** prediction:

```text
L1 = sum(abs(prediction - HR) over valid RGB values)
     / (3 * number of valid spatial pixels)
```

Validation reconstruction scores use prediction clipped to [0,1], unquantized float HR and prediction, full RGB, and no extra border shaving. Compute per-image L1 and MSE over valid RGB values, then:

```text
PSNR_i = 10 * log10(1 / MSE_i)
reported PSNR = arithmetic mean of PSNR_i over the 115 images
```

Use float64 metric accumulation and preserve per-image rows. Pooled-MSE PSNR, if included, needs a different explicit name. Represent exact-zero MSE with a flag and an infinity-safe serialization; do not cap it to a flattering finite value. Record prediction min/max, out-of-range fraction and nonfinite counts before clipping. Empty support fails; current release support is all true.

SSIM: pinned implementation/version; 11x11 Gaussian window, sigma=1.5, K1=0.01, K2=0.03, population covariance, data_range=1; per-channel then RGB mean. Average only window centers with full valid support and no padding. This naturally removes the five-pixel window margin, with no additional border shave. Test against the equivalent independent float64 definition, targeting absolute agreement <=1e-6.

Define the bicubic baseline once: PyTorch bicubic interpolation to the HR shape, `align_corners=False`, `antialias=False`, pinned Torch version. Apply identical scoring/clipping. Cache all validation baseline per-image scores and hashes. Never change baseline settings after observing model results.

### 7.2 Cadence

| Event | Cadence | Required records |
|---|---|---|
| Initial reference | Step 0, after reset and before training | Full main validation metrics for random initialization and bicubic; initial weight hash |
| Training aggregates | Every 50 successful updates or 60 seconds, whichever comes first | Interval mean raw L1, LR, gradient norm, update/microbatch counts, throughput and sample exposure |
| Health heartbeat | Every 60 seconds | State/phase, update, heartbeat UTC, deadline, last checkpoint and last successful upload |
| GPU/host telemetry | Every 60 seconds | GPU utilization/VRAM/temperature, host RAM, disk free, loader wait, update time; no financial information |
| Early validation | Steps 500 and 1,000 | Full 115-image PSNR, SSIM, L1, MSE; per-image CSV |
| Regular validation | Every 2,000 updates or 60 minutes since the last full validation, whichever comes first | Same full validation; baseline gains; best-checkpoint decision |
| Fixed TRAIN diagnostic | At full validation events | Same deterministic 32 TRAIN identities, full-frame, same metrics; clearly label as training diagnostics |
| Image panels | Step 0, 1,000, then every 10,000 updates and final | Fixed IDs/coordinates/ranges; bicubic/SR/HR/error panels |
| Unattended progress report | Every 6 hours from T0 | Section 11 report, saved outside the VM |
| Final evaluation | Once substantive training ends | Selected best checkpoint on full main validation and separate 15-image aerial validation |

Select fixed TRAIN diagnostic IDs deterministically from TRAIN only and log them. Choose visual validation IDs before observing predictions, preserving a representative reproducible set. Do not use fixed diagnostics as extra training or a second selection criterion.

Run validation serially at an optimizer-update boundary, after saving the evaluated model state. Never overlap validation with changing training weights. Coalesce simultaneous step/time triggers, save trigger reasons, restore training mode and RNG state, and persist cadence state across resume. Validation and checkpoint time belong in the 120-hour lifecycle.

Log training metrics from a single well-defined optimizer update interval; distinguish interval mean, optional EMA display, fixed TRAIN metrics and validation metrics. Do not label moving augmented crop loss as full-dataset validation loss.

Select best using **highest full-main-validation mean RGB PSNR after at least one optimizer update**, with earliest checkpoint winning an exact tie. SSIM is secondary and does not override selection. Step-0 metrics are a reference only. Aerial validation never selects weights or hyperparameters.

Save machine-readable JSONL/CSV and TensorBoard events, plus regularly refreshed static plots or an HTML report. No publicly exposed TensorBoard port or external experiment-tracking account is required. Durable files must remain readable after the VM is gone.

## 8. Checkpoints, resume and export integrity

### 8.1 Cadence and retention

- **Initial:** scratch weights and full resolved configuration before preflight; preserve the hash used for step 1.
- **Latest resumable:** every 500 updates or 15 minutes, whichever comes first, at a completed optimizer boundary.
- **Best:** every full-validation improvement; store the checkpoint matching the evaluated step and metrics.
- **Milestones:** 5k, 10k, 25k, 50k, 75k and 100k updates, when reached.
- **Final/interrupted:** on completion, cutoff, supported termination signal or handled failure.

All saved checkpoints use atomic local writes. Keep at least two most recent complete resumable checkpoints, all milestone checkpoints, the selected best and the initial state. Local deletion/pruning may remove only run-owned redundant files after their required durable copies are verified. Do not delete previous cloud checkpoints just to save space during this confirmation experiment.

### 8.2 Full resume state

Include model, optimizer, scheduler and scaler if applicable; completed optimizer update; target update count; best validation record; sampler permutation/epoch and **consumed** cursor; all RNG states; augmentation seed scheme; effective/microbatch sizes and accumulation; consumed crop count; last logging/validation/checkpoint event state; source/config/data hashes; environment; and the original absolute timestamps.

Checkpoint at accumulation boundaries. A crash mid-update resumes the preceding complete update; it must not increment the update count for skipped or failed updates. On resume, discard stale prefetched batches and reconstruct the next sequence. Use monotonic timers within a process and persist UTC/event state for restart. Never grant more time on resume.

### 8.3 Durable verification

Write immutable checkpoint objects, for example `checkpoints/step-000005000-<sha256>.pt`. Upload accompanying size/SHA metadata and verify by a separate streamed readback. Publish `latest.json` or `best.json` **only after** successful verification. Use generation preconditions/atomic pointer replacement so a partial upload cannot become the resume target.

Sync and verify metrics/status at least every five minutes and each checkpoint before declaring it durable. Remote status must include both current local update and last remotely verified update; never imply the newest local checkpoint is backed up when it is not.

Before leaving the run unattended, reload a checkpoint from the durable object, verify its hash/schema, and prove that the resume contract works. Final export is not the first backup attempt.

## 9. Cloud configuration and the 120-hour policy

### 9.1 Machine and storage

| Item | Configuration |
|---|---|
| Machine | `g2-standard-12` |
| GPU | Exactly one NVIDIA L4, 24 GB |
| Host | 12 vCPUs, 48 GB RAM |
| Provisioning | `STANDARD`, never Spot/preemptible |
| Maintenance | `TERMINATE` |
| Automatic restart | Disabled; no hidden restart/replacement loop |
| Boot | Supported ordinary Ubuntu LTS image, resolved and pinned; approximately 100 GB pd-balanced |
| Working disk | Approximately 200 GB pd-ssd, run-owned and explicitly auto-delete |
| Durable output | Unique prefix in the existing bucket, outside the VM/disks |
| Network/security | Existing authorized network; authenticated access; no public dashboards or credential files |

G2 already includes its L4. Do not add a second accelerator. G2 does not support pd-standard; do not use a Deep Learning VM boot image where G2 documentation disallows it. Install and verify the driver/runtime on the supported OS. Resolve current image identifiers and compatibility from official documentation at execution time.

Prefer an available G2 zone in europe-west4, near retained data. If unavailable, try another supported zone/region with STANDARD L4 availability, recording it. Check only needed GPU/resource quota, capacity, IAM and network access. **Do not call billing/budget/credit/pricing APIs or read historical monetary allowance files.** Resource quota is not a monetary budget. Do not request quota increases or broaden IAM unnecessarily.

### 9.2 Absolute timeline

Before the first VM creation request, persist T0 in UTC in a durable immutable run-control object. Also record the provider's eventual creation timestamp. This conservative clock starts no later than the VM's life; setup and failed retries do not reset it.

| Boundary | Absolute time | Action |
|---|---|---|
| Ordinary completion | When update 100,000 is reached | Immediately stop training and finalize |
| Soft training cutoff | T0 + 114 hours | Stop new updates; save; final validation/export |
| Finalization target | T0 + 117 hours | Required artifacts verified; request cleanup |
| External cleanup cutoff | T0 + 118 hours | Controller enforces deletion of exact run-owned VM/disks |
| Native backup deletion | T0 + 119 hours | Compute Engine absolute termination time, action DELETE |
| User maximum | T0 + 120 hours | Must not intentionally retain or extend the VM |

Early failures also finalize and clean up; do not leave a failed VM idle. Never continue training to use up remaining time. Never reset T0, change these timestamps later, automatically start another 120-hour run, or replace a resource after cleanup cutoff.

Use **absolute `termination-time`**, not `max-run-duration=120h`: relative duration can reset after stop/start. An absolute timestamp is still editable by privileged callers, so the software must reject extensions and verify the saved value.

Google documents possible delay before automatic termination begins and does not promise exact deletion completion at a particular second. The early cleanup and native margin enforce the requested policy without making an impossible provider-latency guarantee. Report any actual cleanup failure immediately in durable status; never infer absence from an API/auth error.

### 9.3 Creation and readback contract

The following is a **reference Bash command**, not a complete launcher. Implement resolution of its variables, service account, working disk, startup package and ownership labels before using it. The saved timestamps and exact run ID are authoritative.

```bash
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=g2-standard-12 \
  --provisioning-model=STANDARD \
  --maintenance-policy=TERMINATE \
  --no-restart-on-failure \
  --termination-time="$NATIVE_DELETE_AT_UTC" \
  --instance-termination-action=DELETE \
  --no-deletion-protection \
  --boot-disk-auto-delete \
  --boot-disk-type=pd-balanced --boot-disk-size=100GB \
  --image="$PINNED_OS_IMAGE" --image-project="$IMAGE_PROJECT" \
  --subnet="$SUBNET" \
  --service-account="$RUN_SERVICE_ACCOUNT" \
  --scopes=cloud-platform
```

API scope is not IAM authorization: grant only the existing/minimum needed permissions. The guest reads required source objects and writes its unique run prefix. The separate controller handles exact run-owned compute deletion. Do not print tokens or create long-lived service-account keys.

Read back the created instance and disk identities:

```bash
gcloud compute instances describe "$VM_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --format='json(id,name,creationTimestamp,status,machineType,guestAccelerators,scheduling,resourceStatus.scheduling,deletionProtection,disks)'
```

Assert one L4, expected G2 type, provisioning STANDARD, preemptible false, automaticRestart false, onHostMaintenance TERMINATE, instanceTerminationAction DELETE, expected absolute scheduling.terminationTime, no relative maxRunDuration, deletionProtection false, and autoDelete true for **every disposable attached disk**. Once RUNNING, verify `resourceStatus.scheduling.terminationTimestamp` against the saved native cutoff by parsing UTC values.

Record numeric VM ID and full disk resource identities before training. Names alone are insufficient for later cleanup. Never delete a different resource that has reused a name.

## 10. Detached execution and independent cleanup

The run must continue with the SSH session, browser and Codex conversation closed.

Use a boot-persistent `systemd` service for a run supervisor, which launches training, tracks deadlines, handles signals and invokes bounded idempotent finalization. Keep an independent local deadline timer as defense in depth. A bare background shell, notebook or `nohup` command is insufficient as the only lifecycle control.

The service must check the stored phase before starting: do not start training if already complete, in finalization, or past cutoff. Resume only from a complete verified checkpoint with matching configuration/data identity. Allow at most two automatic transient process-restart attempts with a 60-second delay and persist the attempt counter; numerical/data-integrity failures must not enter an automatic retry loop.

Implement an **external managed controller**, for example authenticated Cloud Run plus Cloud Scheduler, before VM launch. It must not depend on the agent's laptop, the guest OS, an open conversation or continued SSH access. A stopped VM can lose its effective native runtime termination timestamp; the external controller closes this gap.

Recommended separation:

- Cloud Scheduler invokes the authenticated controller every five minutes for narrow lifecycle enforcement. This is a safety check, not a five-minute research report.
- Controller reads the durable immutable run-control object, guest status, checkpoint verification pointers and exact registered resource IDs.
- At six-hour boundaries it writes the richer progress report in section 11.
- On verified finalization, expiration of a bounded finalization grace period, an explicitly reconciled unrecoverable STOPPED state, or the cleanup cutoff, it deletes only this run's recorded ephemeral resources, following the state rules below.
- It polls bounded delete operations and verifies exact VM/disk absence. An API error is a failure, not absence.
- It writes a cleanup receipt, removes its Scheduler job and stops recurring checks. Configure the service with zero minimum instances; remove run-specific controller resources last when safely possible, preserving cleanup evidence.

Use authenticated Cloud Scheduler OIDC invocation for Cloud Run, least needed permissions and guarded resource identity checks. Do not create a public unauthenticated deletion endpoint. Lock concurrent controller invocations and make all transitions idempotent. Reject control objects whose run ID, deadline or resource identities differ from the registration record.

Grant the Scheduler caller service account Cloud Run invoker on the selected service; use the service URL without query parameters as the OIDC audience. Direct calls to Google APIs use the appropriate OAuth credential, not that OIDC token. Scheduler delivery is at least once, so deduplicate callbacks and report slots using durable state outside the container. Set service minimum instances to zero and check for revision-level overrides. Scaling to zero is not immediate, and a five-minute schedule is not itself a hard deadline. Retain the independent native VM cutoff. Remove the Scheduler job only after compute cleanup and its durable receipt are verified. If the controller requests its own service deletion, save the operation identity first and mark that deletion as requested, not independently verified; verify final controller disappearance from the returning agent or another independent control surface.

Keep `phase`, `failure_reason`, `cleanup_ready` and `finalization_deadline_utc` as separate fields. A fatal training error enters FINALIZING with `cleanup_ready=false`; it must not trigger immediate deletion while the guest exports. Persist a finalization deadline equal to the earlier of finalization entry plus two hours and T0+118h. Normal target completion uses the same bounded grace. Delete early when required exports have been verified and `cleanup_ready=true`, or at that grace deadline if export cannot finish, or at T0+118h. The periodic controller must not reset the grace period. For an already STOPPED/unrecoverable guest, the controller records the failure and last previously verified durable checkpoint, marks any final export incomplete, and cleans up without pretending that the guest ran finalization. Do not automatically boot it again for export. Native T0+119h deletion remains unchanged.

Test normal completion, duplicate callbacks, guest failure, STOPPED VM, missing heartbeat, failed upload, API failure and approaching deadline. If the independent deadline mechanism or its permissions cannot be verified, **do not leave the GPU unattended**. Save evidence and clean up; do not substitute a promise to return later.

Keep regular verified backups throughout. At the hard cleanup cutoff, do not extend VM life to retry a stalled final upload. Preserve all previously verified durable artifacts, flag `FINAL_EXPORT_INCOMPLETE` if needed, and record the last verified update and any potentially lost local interval. The deadline and continuous backup design mean the newest unsynced bytes cannot be guaranteed during a storage outage. Never delete shared source/output bucket data to solve this condition.

## 11. Six-hour checks and progress reports

This requirement was explicitly added by the user. The reports run independently after the initial conversation ends; no interactive confirmation or new conversation turn is required every six hours.

Anchor report slots to **T0 + 6h, 12h, 18h, ...** until completion/cutoff. Persist slot IDs so retries do not duplicate reports. The five-minute controller schedule may produce each report shortly after its due time; record due and actual UTC times. A retry writes a clearly labelled catch-up for a missed slot, not invented historical measurements. Write a final report immediately on completion/failure, even between six-hour slots.

Each report must contain:

```text
run_id; report_slot; due_utc; observed_utc; run phase
VM status and exact identity; STANDARD/L4 configuration still matches
completed_updates / 100000; updates since prior report; crop exposure
training raw L1 interval trend; learning rate; gradient-health summary
latest FULL validation step/time/count; RGB PSNR, SSIM, L1, MSE
delta from bicubic and from initialization; change since prior report
best FULL validation PSNR, selected step and checkpoint SHA
last local checkpoint vs last remotely verified resumable checkpoint
backup age; metric-log age; heartbeat age
recent GPU/host telemetry and age of that telemetry
observed throughput; estimated time to target; remaining absolute time
alerts, recovery/cleanup actions, and next report due time
```

ETA uses actual recent optimizer-update timing and measured validation/checkpoint overhead. Label it an estimate and never move the deadline to match it. Do not turn ETA into a monetary estimate.

Store immutable `progress/slot-0001.json` and `.md` reports plus an atomic `progress/latest.json` pointer. Refresh a simple readable `progress/index.html` or Markdown index with learning curves. Keep these private; the user can inspect them using existing authenticated access after the VM is deleted.

Use explicit health states such as HEALTHY, STALE_TELEMETRY, VALIDATION_OVERDUE, BACKUP_STALE, STALLED, FAILED and FINALIZING. Distinguish a legitimate recorded validation/upload phase from a stuck optimizer. Suggested triggers: heartbeat stale >10 minutes; checkpoint verification stale >30 minutes; full validation overdue >90 minutes; no progress/phase completion >30 minutes outside a declared bounded operation. Treat stale data as unknown, never healthy by default.

No progress report changes learning rate, architecture, data, metric policy or checkpoint selection. A plateau is logged, not automatically tuned away. Fatal nonfinite/data-integrity failures stop the trainer and invoke finalization. If remote checkpoint age exceeds 30 minutes, pause new updates at a safe boundary and attempt bounded backup recovery; failure must finalize using the latest durable state without extending any cutoff.

Do not set up a second redundant Codex heartbeat automation for this handoff. The external service provides the explicitly requested six-hour checks while allowing the conversation to end. If the user later specifically asks for in-app notifications or agent reasoning at each interval, treat that as a separate requested monitoring change, keeping the original deadline.

## 12. Initial babysitting and ending the conversation

After gates pass, reset to the saved scratch initialization and start the one substantive seed-42 fit. Observe approximately **30–60 minutes of actual training**, not merely VM boot time. Stay until the checks below are satisfied; cap startup troubleshooting at approximately two hours after attempted training start, always inside the original deadline. An unresolved failure requires safe finalization/cleanup rather than unattended departure.

Before ending the conversation, verify and record:

- Increasing successful optimizer update count, finite losses/gradients/weights and measured real-data throughput.
- At least one completed post-initialization **full 115-image validation**, with PSNR/SSIM and per-image output.
- At least one remotely readback-verified **full resumable checkpoint**, with the tested resume path.
- Healthy detached service, plausible GPU/host telemetry and sufficient measured disk headroom.
- Correct effective batch, LR schedule and logging cadence.
- Saved immutable deadlines, control-plane deletion readback and working external lifecycle controller.
- An immediate initial progress report, plus verified scheduling/state for the first T0+6h report. Do not wait six hours just to end the conversation.
- Original inputs unchanged; no test metrics, financial queries or unrelated resources touched.

Then end the conversation with a concise truthful status: **training is running**, not completed. Include run ID, project/zone/VM, actual step, initial/current validation, durable latest checkpoint, private progress location, next report time, training cutoff and native deletion time. Show UTC and Europe/Athens timestamps with the correct timezone offset for each date; store UTC internally.

Explain that training, six-hour reporting, finalization and cleanup now run without the open chat. Do not keep the conversation alive for the full experiment. Do not ask the user to babysit ordinary runtime operations.

## 13. Finalization and cleanup sequence

On target reached, soft cutoff or failure:

1. Stop launching new optimizer updates; complete or abandon the current accumulation safely, retaining the last successful optimizer count.
2. Atomically save final/last state and persist outcome: TARGET_REACHED, TIME_LIMITED, FAILED or INTERRUPTED, with reason and timestamps.
3. Freeze best-checkpoint identity. Run final **main validation** and separate **15-image aerial validation** on that checkpoint if operationally possible within the reserved window. No test evaluation. A missing final evaluation is an explicit gap, not fabricated results.
4. Save learning curves, per-image metrics, fixed visual panels, telemetry, progress history, source/environment/config pins and a concise scientific report.
5. Upload required artifacts and verify their SHA/size by readback. Write `artifacts-manifest.json`, then `FINALIZED.json` only once the declared required set is durable.
6. Controller verifies the receipt and exact resource identities, deletes the VM and disposable disks early, and confirms absence through API readback. Check for detached run-owned disks as well as attached auto-delete disks.
7. Persist `CLEANUP_VERIFIED.json` with resource IDs, operation outcomes and observed absence; remove recurring Scheduler checks. Do not mark cleanup verified merely because a DELETE request was accepted.

Keep scientific outcome, export outcome and cleanup outcome as separate fields. For example, TARGET_REACHED can coexist with FINAL_EXPORT_INCOMPLETE or CLEANUP_FAILED. Avoid a single misleading COMPLETE flag.

Final scientific reporting must state the actual update count, seed, model/config hashes, selected validation step, data source scope, metric convention, bicubic comparison and limitations. Confidence intervals may use paired scene/site-group resampling when defensible groups exist; report group count and uncertainty limitations. A single seed does not provide seed variability.

## 14. Durable output layout and return-to-chat procedure

```text
<run-prefix>/
  control/run-control.json
  control/resource-registration.json
  control/deadline-verification.json
  config/resolved.yaml
  config/uint8-index-pin.json
  provenance/data-audit.json
  provenance/source-group-map.csv
  provenance/source-archive-and-diff/
  provenance/environment-lock-and-hardware.json
  preflight/results.json
  preflight/resume-verification.json
  checkpoints/initial-<sha>.pt
  checkpoints/step-<update>-<sha>.pt
  checkpoints/latest.json
  checkpoints/best.json
  logs/train.jsonl
  logs/validation.jsonl
  logs/telemetry.jsonl
  logs/events.jsonl
  tensorboard/
  metrics/bicubic-main-validation.csv
  metrics/main-validation-step-<update>.csv
  metrics/final-aerial-validation.csv
  figures/
  progress/slot-<id>.json
  progress/slot-<id>.md
  progress/latest.json
  progress/index.html
  status/guest-status.json
  status/controller-status.json
  final/report.md
  final/artifacts-manifest.json
  final/FINALIZED.json
  final/CLEANUP_VERIFIED.json
```

Record the actual output URI in a local user-facing launch summary before ending initial supervision. Never leave the user with only this template prefix.

When the user later says the run has finished, inspect the persisted run registration and durable status first. Do not launch/restart training or create another VM automatically. Verify final receipts, artifact hashes and actual VM/disk absence; finish only any remaining authorized run-specific cleanup. Report the recorded validation results and learning curves, explain target/time-limited/failure status, and identify the best model checkpoint. Preserve sealed test sets.

## 15. Acceptance checklist for the executing agent

- [ ] Only one D3 scratch model, seed 42, 100k target; all resolved settings frozen.
- [ ] Data pins/counts/support/alignment/leakage checks recorded; frozen inputs unchanged.
- [ ] Correct STANDARD g2-standard-12, one L4; no Spot or financial inspection.
- [ ] Absolute UTC deadline persisted before creation; guest and external controls independently tested.
- [ ] Masked loss, gradient accumulation, scheduler, metric and evaluator-purity tests pass.
- [ ] Same-runtime next-batch/next-update resume test passes with stated numerical tolerance.
- [ ] Full metrics throughout training; no held-out optimizer updates and no test-set evaluation.
- [ ] Frequent atomic checkpoints and readback-verified durable copies.
- [ ] Six-hour progress reports scheduled and tested; no dependence on an open chat.
- [ ] Initial 30–60-minute babysitting completed with full validation and verified backup before ending conversation.
- [ ] Run reaches 100k or is honestly reported time-limited/failed; no time extension.
- [ ] Final artifacts and scientific report preserved; exact VM/disposable-disk cleanup verified or explicitly failed.
- [ ] User receives actual status/output locations and can return later without reconstructing run identity.

## 16. Evidence and official references

Data facts and retained cloud pins come from the delivered handoff and wrapped index. During planning, the current uint8 loader, shard selector/downloader and legacy trainer interfaces were inspected. Full payload integrity, GPU compatibility and controller execution are future acceptance gates; none is claimed merely by this document.

- SwinIR source: https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py
- Classical configuration: https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/options/swinir/train_swinir_sr_classical.json
- Published training details: https://openaccess.thecvf.com/content/ICCV2021W/AIM/supplemental/Liang_SwinIR_Image_Restoration_ICCVW_2021_supplemental.pdf
- G2 machine specifications/limitations: https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#g2-vms
- G2 creation: https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series
- Absolute runtime limits and restart semantics: https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime
- Instance creation flags: https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create
- Instance scheduling/readback schema: https://docs.cloud.google.com/compute/docs/reference/rest/v1/instances
- Disk auto-delete behavior: https://docs.cloud.google.com/compute/docs/disks/modify-persistent-disk#modify_a_disks_auto-delete_setting
- Authenticated Scheduler invocation of Cloud Run: https://docs.cloud.google.com/run/docs/triggering/using-scheduler
- Scheduler HTTP authentication: https://docs.cloud.google.com/scheduler/docs/http-target-auth
- Cloud Run minimum instances: https://docs.cloud.google.com/run/docs/configuring/min-instances
- Scheduler delivery semantics: https://docs.cloud.google.com/scheduler/docs/overview

The 100k/batch-16/warm-up-cosine settings are this confirmation protocol, not the original paper's published training recipe. Verify current infrastructure CLI/API compatibility before execution while preserving the human decisions and fixed time limit.
