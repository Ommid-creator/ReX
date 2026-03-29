# ReX Repository Knowledge Summary

ReX (Responsibility EXplanations) is an explainable AI framework that generates causal responsibility explanations for neural network predictions by systematically masking/occluding input regions and observing classification outcomes.

---

## Directory Structure

```
rex_xai/
├── explanation/
│   ├── explanation.py          # Single explanation extraction (Global/Spatial/Contrastive)
│   ├── multi_explanation.py    # Multiple explanations via spotlight search
│   ├── rex.py                  # Main orchestration entry point
│   └── evaluation.py           # Evaluation metrics
├── mutants/
│   ├── mutant.py               # Mutant class (masked input variants)
│   ├── occlusions.py           # Occlusion strategies (spectral, context)
│   ├── box.py                  # Hierarchical box/region definitions
│   └── distributions.py        # Distribution sampling for box splitting
├── responsibility/
│   ├── responsibility.py       # Core causal explanation algorithm
│   ├── resp_maps.py            # Responsibility map data structure
│   └── prediction.py           # Model prediction wrapper
├── input/
│   ├── config.py               # CausalArgs configuration
│   ├── input_data.py           # Data loading & mask_value setup
│   └── onnx.py                 # ONNX model support
├── output/
│   ├── visualisation.py        # Plotting & visualization
│   └── database.py             # Database storage
├── utils/
│   ├── _utils.py               # Enums and utility functions
│   └── logger.py               # Logging
├── lib.py                      # Public library API
└── rex_wrapper.py              # Wrapper for external use
```

---

## Masking / Occlusion Logic

### Occlusion Strategies — `rex_xai/mutants/occlusions.py`

Two strategy functions are used to replace masked-out regions with plausible data rather than a constant.

**`spectral_occlusion(mask, data, noise=0.02, device)`**
- For 1D spectral / time-series data.
- Identifies contiguous masked regions and linearly interpolates between their boundary values.
- Optionally adds Gaussian noise to the interpolated values.

**`context_occlusion(mask, data, context, noise=0.5)`**
- Replaces masked regions with corresponding values from a background/context image (e.g., a healthy scan).
- Optionally smooths the context via a Gaussian filter (sigma = `noise`).

### Mask Value Configuration — `rex_xai/input/input_data.py`

`Data.set_mask_value(m)` translates the `mask_value` config string/value into what gets stored on the `Data` object:

| Config value | Behaviour |
|---|---|
| `int` / `float` | Scalar replacement |
| `"min"` | Replace with minimum data value |
| `"mean"` | Replace with mean data value |
| `"spectral"` | Calls `spectral_occlusion()` |
| `"context"` | Calls `context_occlusion()` |
| `"random"` / `"linear"` | Handled in `causal_explanation()` |
| `"none"` | No replacement (NaN) |

### Applying a Mask — `rex_xai/mutants/mutant.py`

**`_apply_to_data(mask, data: Data)`**
- If `mask_value` is callable (spectral/context): `mask_value(mask, data.data)`
- If `mask_value` is a scalar: `torch.where(mask, data.data, mask_value)`
- Otherwise defaults to replacing with `0`.

The **`Mutant`** class wraps a boolean mask tensor and the 4 active boxes it represents:
- `mask` — boolean tensor: `True` = keep, `False` = occlude
- `apply_to_data()` — produces the masked input passed to the model
- `area()` — counts visible (non-masked) pixels/voxels

---

## Box / Region System — `rex_xai/mutants/box.py`

A hierarchical tree (via `anytree`) partitions the input space into rectangular regions.

**`BoxInternal` / `Box`**
- 2D bounds: `row_start`, `row_stop`, `col_start`, `col_stop`
- 3D bounds additionally: `depth_start`, `depth_stop`
- `name` — hierarchical string like `"R:0:1:2"`
- `distribution` — controls how splits are biased (`Uniform`, `Binomial`, `BetaBinomial`, `Adaptive`)

**`spawn_children(min_size, mode, map=None)`**
- Recursively splits a box into 4 sub-boxes by picking a random interior split point.
- 2D: splits at a random `(row, col)` interior point → 4 quadrant boxes.
- 3D: randomly selects 2 of 3 axes and splits along both.
- Respects `min_size`; in `Adaptive` mode uses the responsibility map to bias splits toward high-responsibility regions.

---

## Responsibility Map Computation

### `rex_xai/responsibility/resp_maps.py` — `ResponsibilityMaps`

Stores per-class responsibility score arrays:
- `maps` — dict: class ID → numpy array
- Aggregation style: **Additive** (sum) or **Multiplicative** (product)

**`update_maps(mutants, args, data, search_tree)`**
- For each passing mutant (prediction matched target), distributes responsibility among its 4 active boxes.
- If `args.weighted`, scales by prediction confidence.
- If `args.concentrate`, scales by tree depth (deeper = more concentrated).

### `rex_xai/responsibility/responsibility.py` — `causal_explanation()`

Core iterative algorithm:
1. Initialise box tree (root = entire input).
2. Maintain a queue of active box names.
3. Each iteration: pop a box, split into 4 children, generate all 14 non-empty subsets of children as mutant masks, run batched predictions.
4. Passing mutants → update responsibility maps → add children to queue.
5. Prune queue using the configured `Queue` strategy (`Area`, `All`, `Intersection`, `DC`).
6. Halt when max tree depth or search iteration limit is reached.

---

## Explanation Extraction — `rex_xai/explanation/explanation.py`

The `Explanation` class extracts a minimal sufficient mask from the accumulated responsibility map.

### Strategy: Global (`__global()`)

Insertion-based search:
1. Sort all pixels by descending responsibility.
2. Accumulate pixels in chunks of `chunk_size`.
3. Batch-test accumulated masks.
4. Stop when target class is predicted at or above `minimum_confidence_threshold`.

### Strategy: Spatial (`__spatial()`)

Spatial expansion:
1. Start with a circle centred at the highest-responsibility pixel (or a provided centre).
2. Expand radius by `spatial_radius_eta` each step.
3. Stop when target classification is reached.
4. Optionally refine with a global search within the spatial region.

### Strategy: Contrastive (`contrastive()`)

Runs insertion (sufficiency) and deletion (necessity) tests in parallel:
- **Insertion**: accumulate pixels until target class is predicted → sufficiency mask.
- **Deletion**: remove pixels until target class is no longer predicted → necessity mask.
- Optionally computes a completeness mask (regions that flip the class).

---

## Spotlight / Multi-Explanation Strategy — `rex_xai/explanation/multi_explanation.py`

`MultiExplanation` extends `Explanation` to find multiple diverse explanations.

### `extract()`

1. Run `__global()` → first explanation (spotlight #1).
2. For each subsequent spotlight up to `args.spotlights`:
   - Blank the previously found region in the responsibility map (set to 0).
   - Call `spotlight_search()` to find the next best region.
3. Collect all found explanation masks and confidences.

### `spotlight_search(origin=None)`

Finds the next explanation starting from a spatial point:
1. Start from `origin` (random if not provided).
2. Call `__spatial()` with a bounded number of expansions.
3. If not found and budget remains:
   - `objective="none"`: try random new locations.
   - `objective="mean"` or `"max"`: take small steps in the direction of improving mean/max responsibility score (step size = `args.spotlight_step` pixels).
4. Budget is controlled by `args.max_spotlight_budget`.

### `separate_by(dice_coefficient, reverse=True)`

Groups explanations into non-overlapping clause sets:
1. Compute pairwise Dice coefficients between all explanation masks.
2. Mark pairs with Dice > threshold as conflicting.
3. Find maximal subsets with no internal conflicts.
4. Sort by total area; return ordered list of groups.

### Output Styles

- `"separate"`: each explanation saved to its own file.
- `"composite"`: all explanations overlaid in a single image with different colours/opacities.

---

## Configuration — `rex_xai/input/config.py` (`CausalArgs`)

Key parameters grouped by concern:

**Masking / Occlusion**
| Parameter | Purpose |
|---|---|
| `mask_value` | Occlusion method: `0`, `"mean"`, `"spectral"`, `"context"`, `"random"`, `"linear"` |
| `context` | Enable context occlusion |
| `context_location` | Path to context data file |
| `occlusion_noise` | Gaussian noise / filter sigma for occlusion |

**Explanation Extraction**
| Parameter | Purpose |
|---|---|
| `strategy` | `Global`, `Spatial`, `MultiSpotlight`, `Contrastive` |
| `chunk_size` | Pixels per insertion step |
| `minimum_confidence_threshold` | Target confidence level |
| `batch_size` | Batch processing size |
| `complete` | Compute completeness explanations |

**Spatial Strategy**
| Parameter | Purpose |
|---|---|
| `spatial_initial_radius` | Starting circle radius |
| `spatial_radius_eta` | Radius expansion factor per step |
| `no_expansions` | Max spatial expansion steps |

**Spotlight / Multi-Explanation**
| Parameter | Purpose |
|---|---|
| `spotlights` | Number of spotlight explanations to find |
| `spotlight_step` | Movement step size in pixels |
| `spotlight_objective_function` | `"mean"`, `"max"`, or `"none"` |
| `max_spotlight_budget` | Max spotlight search iterations |
| `permitted_overlap` | Dice threshold for `separate_by()` |
| `multi_style` | `"separate"` or `"composite"` output |

**Box / Responsibility**
| Parameter | Purpose |
|---|---|
| `tree_depth` | Max recursion depth |
| `min_box_size` | Minimum region size |
| `distribution` | `Uniform`, `Binomial`, `BetaBinomial`, `Adaptive` |
| `weighted` | Weight responsibility by prediction confidence |
| `concentrate` | Scale responsibility by tree depth |
| `responsibility_style` | `Additive` or `Multiplicative` aggregation |

---

## End-to-End Data Flow

```
CausalArgs
  └─ Data.set_mask_value()          → stores occlusion function or scalar on Data
       └─ causal_explanation()
            ├─ box.spawn_children() → split input space into 4 sub-boxes
            ├─ Mutant(mask, boxes)  → boolean mask over input
            ├─ _apply_to_data()     → masked input tensor
            ├─ prediction_func()    → model output
            └─ resp_maps.update_maps() → accumulate responsibility scores
  └─ Explanation / MultiExplanation
       ├─ __global() / __spatial()  → minimal sufficient mask
       ├─ spotlight_search()        → additional diverse explanations
       └─ separate_by()             → non-overlapping explanation groups
```

---

## Key Enums — `rex_xai/utils/_utils.py`

```python
Strategy          = ["Global", "Spatial", "MultiSpotlight", "Contrastive"]
Queue             = ["Area", "All", "Intersection", "DC"]   # queue pruning
SpatialSearch     = ["NotFound", "Found"]
ResponsibilityStyle = ["Additive", "Multiplicative"]
Distribution      = ["Binomial", "Uniform", "BetaBinomial", "Adaptive"]
```

---

## File-to-Concern Map

| Concern | Primary file(s) |
|---|---|
| Spectral / context occlusion | `rex_xai/mutants/occlusions.py` |
| Applying masks to data | `rex_xai/mutants/mutant.py` |
| Mask value configuration | `rex_xai/input/input_data.py` |
| Hierarchical region splitting | `rex_xai/mutants/box.py` |
| Distribution / sampling | `rex_xai/mutants/distributions.py` |
| Responsibility accumulation | `rex_xai/responsibility/resp_maps.py` |
| Core causal search loop | `rex_xai/responsibility/responsibility.py` |
| Single explanation (Global/Spatial/Contrastive) | `rex_xai/explanation/explanation.py` |
| Multi-explanation / spotlight | `rex_xai/explanation/multi_explanation.py` |
| Configuration parameters | `rex_xai/input/config.py` |
| Orchestration / entry point | `rex_xai/explanation/rex.py` |
