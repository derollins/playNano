# Test data

## `aris_sample_0.aris`

The first 2 of the 39 frames of the `HeightTrace` channel of `00T2_P3_0000.ARIS` (DNA origami, high-speed AFM, Asylum
Research Cypher), from:

> Heath, G. (2023). *DNA Origami Raw AFM Data - NanoLocz: Image analysis platform for AFM, high-speed AFM and
> localization AFM* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.10670413>

Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This file is not covered by playNano's GPL-3.0
licence.

Changes from the original, to keep the file small (356 KB):

- Only frames 0 and 1 are kept, in `/DataSet/Resolution 0` and `/DataSetInfo/Frames`.
- Only the `HeightTrace` channel is kept: the other channels' images and their entries in
  `/DataSetInfo/Global/Channels` and the `ChannelNames` attribute are removed.
- `/DataSetInfo/Series/Time` and `/DataSetInfo/Series/FrameDirection` are shortened to 2 entries.
- Images are stored gzip-compressed; their values are unchanged.
- Everything else (global parameters, attributes and masks) is copied unchanged.

Heights are stored in metres (`DataUnits = "m"`).
