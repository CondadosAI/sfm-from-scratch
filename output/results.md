# templeRing: mini-SfM vs pycolmap

| templeRing           | registered   |   3-D points |   track len |   mean reproj (px) |   time (s) |
|----------------------|--------------|--------------|-------------|--------------------|------------|
| mini-SfM (this repo) | 47/47        |        7,036 |         4.3 |               0.25 |         23 |
| pycolmap 4.1.1       | 47/47        |        7,625 |         6.2 |               0.3  |          9 |

Note: pycolmap self-calibrates (SIMPLE_RADIAL) while mini-SfM uses the dataset's known K; reprojection errors are therefore not computed against identical camera models.
