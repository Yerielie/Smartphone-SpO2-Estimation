# Data

This project uses the public dataset from Hoffman et al. (2022), *Smartphone camera oximetry in an
induced hypoxemia study*, npj Digital Medicine 5, 146. The data is released under the MIT license
at <https://github.com/ubicomplab/oximetry-phone-cam-data>.

The data is **not** committed to this repository. To download it, run this from the repository root:

```bash
git clone --depth 1 https://github.com/ubicomplab/oximetry-phone-cam-data data/raw
```

`data/raw/` is listed in `.gitignore`.

## Files used

| Path | Content |
|---|---|
| `data/raw/data/ppg-csv/{Left,Right}/10000{1-6}.csv` | Smartphone PPG: columns `R,G,B` = mean pixel value per frame, 30 Hz |
| `data/raw/data/gt/10000{1-6}.csv` | Reference oximeters, 1 Hz. We use `SpO2 5` and `Pulse 5` (Masimo Radical-7) |
| `data/raw/data/gt/metadata.csv` | Column descriptions for the ground-truth files |

## Notes found while loading

- Every ground-truth file ends with a `Collection Halted` footer row, which is dropped.
- In `gt/100004.csv` the first header cell is empty instead of `Time`. The loader renames the first column.
- Timestamps are consecutive 1-s steps with no gaps. `SpO2 5` and `Pulse 5` contain no zeros
  or blanks in these files, but zeros are still treated as missing.
- The camera and oximeters were started and stopped together, so ground-truth row *i* is aligned
  to frame *i*·30. The frame counts are not exactly 30 × the number of ground-truth rows. They range
  from 29.97 to 30.09 frames per row, a difference of −29 to +79 frames (−0.97 s to +2.63 s by the
  end of a recording). See `results/part1_alignment.csv`.
