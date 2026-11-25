# Module 2 — Preprocessing

Classical OpenCV preprocessing between acquisition and learning/inspection.
No neural network lives here.

```
validate → ROI → optional alignment → optional denoise
         → optional illumination correction
         → letterbox/resize → BGR→RGB → ImageNet normalize → tensor
```

The **same** `pipeline.preprocess()` call is used for training and inspection.
A loaded profile stores its preprocessing settings, and inspection uses the
profile's settings, not the current YAML defaults.

| File | Purpose |
|---|---|
| `pipeline.py` | Orchestrates the full order; measures latency; returns `Prepared` + geometry |
| `quality.py` | Blur, brightness, saturation, foreground occupancy, image size; limits learned from good images |
| `roi.py` | Fixed-box ROI from the profile; bounds-checked crop |
| `align.py` | Optional ECC translation/rotation to a reference good image; refuses large corrections |
| `denoise.py` | Optional median/bilateral denoise; **off by default** |
| `illumination.py` | Optional background-field / gamma / CLAHE correction; **off by default** |
| `transform.py` | Aspect-preserving letterbox, ImageNet normalization, model→source heatmap mapping |
| `augment.py` | Train-only, defect-safe shift/rotation/brightness/contrast; optional symmetric flip |

## Important decisions

- **Letterbox, never stretch.** A 16:9 photo is resized to fit 256×256 and
  padded; the scale and padding are recorded so the heatmap can be mapped
  back to the original photograph.
- **Quality is measured on the ROI**, so background does not hide a blurred
  or badly framed part.
- **ROI is a fixed box per profile.** Automatic segmentation is deferred.
- **Alignment is off by default.** A fixed jig is safer; warping can hide a
  real deformation. When enabled, only translation + rotation are allowed,
  and a correction above `max_shift_px` / `max_rotation_deg` becomes
  `BAD_IMAGE`, not `DEFECTIVE`.
- **Denoise and illumination correction are off by default.** Blur can erase
  a scratch; equalisation can hide discolouration that is itself a defect.
- **No global brightness normalisation.** ImageNet mean/std is the only
  normalization; a finish change must remain visible.
- **Augmentation never uses blur, noise, cutout or cut-paste.** Those mimic
  defects and would teach the model that defects are normal.

## Configuration

```yaml
preprocess:
  keep_aspect: true
  pad_value: 0
  roi: {enabled: false, x: 0, y: 0, width: 0, height: 0}
  align:
    enabled: false
    reference: ""
    max_shift_px: 8.0
    max_rotation_deg: 2.0
  denoise: {enabled: false, method: median, ksize: 3}
  illumination: {enabled: false, method: background, sigma: 40.0, ...}

quality:
  enabled: true
  check_foreground: true
  require_same_size: true
```

The preprocessing keys that change feature geometry (size, ROI, alignment,
denoise, illumination, letterbox) are part of `config_hash`. A profile cannot
be loaded under a different preprocessing setup.

## Heatmap mapping

`transform.map_to_source(anomaly_map, prepared.geometry)` returns a map at the
**original photograph's** resolution. It crops the letterbox bars, resizes to
the ROI, applies the inverse alignment warp when alignment was used, and
places the result at the ROI offset. The CLI writes the heatmap on the
original photo, not on the 256×256 model view.

## Tests

`tests/test_preprocess.py` covers:

- output tensor shape/dtype and ImageNet normalization
- BGR→RGB channel order
- aspect-ratio preservation and letterbox padding
- ROI crop and geometry
- heatmap mapping back to original coordinates
- alignment reduces a known shift; large corrections are refused
- aligned heatmap maps back to the original pose
- illumination correction preserves a scratch
- denoise is off by default and opt-in
- stored profile geometry wins over current config
- quality rejects blurred, dark, wrong-size and empty-frame images
- a 4 px scratch on a 512 px photo is still visible after resizing to 256
- preprocessing latency and per-stage timings are recorded

```python
from module_2_preprocessing.pipeline import preprocess
from module_2_preprocessing.transform import map_to_source
from module_2_preprocessing import quality, roi, align, denoise, illumination, augment
```
