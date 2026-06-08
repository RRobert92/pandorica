<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="resources/assets/pandorica-lockup-dark.png">
    <img alt="PANDORICA" src="resources/assets/pandorica-lockup.png" width="520">
  </picture>
</p>

# PANDORICA

Analytical tools for electron microscopy.

## Bundled tools

- **`pandorica.stitch`** — serial-section tomogram stitcher: image-driven
  coarse→fine alignment — the image fixes each section's global pose (rotation,
  shift, anisotropic stretch) and the microtubules drive only the fine residual
  warp — with a diffeomorphism-guarded TPS warp, global pose solve, and CPU/GPU
  volume export. Falls back to an MT-only pose solve when volumes are absent.
  Plain-language tour: [`HOW_IT_WORKS.md`](pandorica/stitch/HOW_IT_WORKS.md).
  Dense reference: [`pandorica/stitch/README.md`](pandorica/stitch/README.md).
- **`pandorica.napari`** — napari plugin: visually validate the stitcher on
  real datasets and record coarse-alignment ground truth by hand.

## Install

```bash
pip install pandorica                # core stitcher + CLI
pip install "pandorica[napari]"      # add the napari validator widgets
```

## Use

```bash
python -m pandorica.stitch.cli <input_dir>
```

```python
from pandorica.stitch.cli import run_stitch
run_stitch("path/to/sections")       # writes <input_dir>/stitched_output/
```

If `tardis_em` is also installed, the same entry point is exposed as the
`tardis_stitch` console script.

## Citation

If `pandorica` contributes to a publication or presentation, please cite the
software:

```bibtex
@software{kiewisz_pandorica_2026,
  author  = {Kiewisz, Robert},
  title   = {pandorica: analytical tools for electron microscopy},
  year    = {2026},
  version = {1.0.3},
  url     = {https://github.com/RRobert92/pandorica},
  license = {PolyForm-Noncommercial-1.0.0}
}
```

### Related prior work (scholarly context; not required by pandorica's license)

Pandorica's stitcher is a from-scratch Python reimplementation in the
serial-section EM lineage; it does not include code from these projects.
Citing them is a scholarly courtesy when positioning your work, not a
legal obligation imposed by pandorica:

- Lindow *et al.*, *Journal of Microscopy* (2021), SerialSectionAligner —
  [doi:10.1111/jmi.13039](https://doi.org/10.1111/jmi.13039)
- Weber *et al.*, *PLoS ONE* (2014), microtubulestitching

See [`pandorica/stitch/README.md`](pandorica/stitch/README.md) for full author
lists and the wider related-work list (IMOD, msemalign, Okapi-EM, …).

## License

**PolyForm Noncommercial License 1.0.0** — free for research, study, and use
by educational, public-research, charitable, public-health, environmental,
and government institutions, regardless of funding source. Commercial use
requires a separate license; see [COMMERCIAL.md](COMMERCIAL.md) to request
one. Full terms in [LICENSE](LICENSE).

Contributions require a Developer Certificate of Origin sign-off — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

Robert Kiewisz — <robert.kiewisz@gmail.com>
