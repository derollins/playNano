# playNano branding

Editable source files for the playNano logo and icon, the colour palettes they use, and how the
published versions are made from them.

**Edit the files here, then re-export.** The published copies are generated from these and are used by
the docs, the README and the GUI. Nothing reads this folder directly, and it is not part of the
docs site or the PyPI package.

## Files

| File | What it is |
| --- | --- |
| `playnano_logo_text.svg` | Logo with wordmark, 350 × 100 px, **three pages** (below). Needs the Basic font to edit. |
| `playnano_logo_square.svg` | The scan-line play mark alone, 100 × 100 px. Used for icons 48 px and up. |
| `playnano_logo_small_pixelaligned.svg` | Simplified mark for **16 and 32 px only**, with 5 pixel-aligned bars. |
| `palettes/` | Inkscape palettes (`.gpl`) and the `playnano_gold` gradient (see [Palettes](#palettes)). |

Pages in `playnano_logo_text.svg`:

| Page | Background | "play" | "Nano" | Use on |
| --- | --- | --- | --- | --- |
| 1 | transparent | `#1E1E1E` | `#95480F` | light backgrounds |
| 2 | `#1E1E1E` (GUI charcoal) | `#E0E0E0` | `#F4C555` | dark backgrounds |
| 3 | `#FFFFFF` | `#1E1E1E` | `#95480F` | anywhere a solid white box is wanted |

## Published files and where they come from

| Published file | Made from |
| --- | --- |
| `docs/_static/logo/playnano_logo_light.svg` | `playnano_logo_text.svg` page 1, text converted to paths |
| `docs/_static/logo/playnano_logo_dark.svg` | page 2, text converted to paths, **dark background rectangle removed** |
| `docs/_static/logo/playnano_icon.svg` | `playnano_logo_square.svg` |
| `docs/_static/logo/favicon.ico` | the 16, 32 and 48 px icon PNGs combined |
| `src/playnano/resources/icons/playnano_icon_16.png`, `_32.png` | `playnano_logo_small_pixelaligned.svg` |
| `src/playnano/resources/icons/playnano_icon_48.png` … `_512.png` | `playnano_logo_square.svg` |

The README logo and the docs sidebar both use the two `docs/_static/logo/playnano_logo_*.svg` files.
The GUI loads every size in `src/playnano/resources/icons/` (see `ICON_SIZES` in
`src/playnano/gui/main.py`).

## Re-exporting

These commands use the Inkscape 1.4 command line. Run them from the repository root. On Windows
use `inkscape.com` (in `C:\Program Files\Inkscape\bin\`) so messages are printed to the terminal.

**Docs SVGs.** `--export-plain-svg` drops Inkscape's editor data and `--export-text-to-path` makes the
wordmark display correctly without the Basic font installed:

```bash
inkscape branding/playnano_logo_text.svg --export-type=svg --export-plain-svg \
  --export-text-to-path --export-page=1 --export-filename=docs/_static/logo/playnano_logo_light.svg

inkscape branding/playnano_logo_text.svg --export-type=svg --export-plain-svg \
  --export-text-to-path --export-page=2 --export-filename=docs/_static/logo/playnano_logo_dark.svg

inkscape branding/playnano_logo_square.svg --export-type=svg --export-plain-svg \
  --export-area-page --export-filename=docs/_static/logo/playnano_icon.svg
```

Then open `playnano_logo_dark.svg` and **delete the full-page `#1E1E1E` rectangle**, so the dark logo
sits on a transparent background. Furo's dark theme and GitHub's dark mode supply their own background.

**GUI icons.** Export the page at each width:

```bash
for s in 16 32; do
  inkscape branding/playnano_logo_small_pixelaligned.svg --export-type=png --export-area-page \
    --export-width=$s --export-filename=src/playnano/resources/icons/playnano_icon_$s.png
done
for s in 48 64 128 256 512; do
  inkscape branding/playnano_logo_square.svg --export-type=png --export-area-page \
    --export-width=$s --export-filename=src/playnano/resources/icons/playnano_icon_$s.png
done
```

If you add or remove a size, update `ICON_SIZES` in `src/playnano/gui/main.py`. The GUI tests check that
every listed size exists.

**Favicon.** Combine the 16, 32 and 48 px icons with Pillow. Each size is stored as-is, without
resampling, so the pixel-aligned small icons stay sharp:

```python
from PIL import Image

icons = "src/playnano/resources/icons/playnano_icon_{}.png"
Image.open(icons.format(48)).save(
    "docs/_static/logo/favicon.ico",
    sizes=[(16, 16), (32, 32), (48, 48)],
    append_images=[Image.open(icons.format(16)), Image.open(icons.format(32))],
)
```

## Font

The wordmark is set in **Basic** by Eben Sorkin (SIL Open Font License). It is the same font as the GUI
interface text and is bundled at `src/playnano/resources/fonts/basic/basic_regular.ttf`. Install it
before editing `playnano_logo_text.svg`, then restart Inkscape. Basic has a single weight, so "play"
and "Nano" are told apart by colour, not boldness.

**Steps Mono** (`src/playnano/resources/fonts/Steps-Mono/`) is used for timestamps and annotations on
exported videos and works as an accent font, e.g. for version numbers or labels.

## Colours

The logo is drawn from the `playnano_gold` colormap. The supporting colours come from the GUI stylesheet
(`src/playnano/gui/styles/dark_bluegreen.qss`).

| Role | Colour |
| --- | --- |
| Logo gradient (main mark), left → right | `#95480F` → `#E59819` → `#FFEEAA` |
| Small icon gradient (16/32 px), left → right | `#BD7014` → `#E59819` |
| Wordmark on light: "play" / "Nano" | `#1E1E1E` / `#95480F` |
| Wordmark on dark: "play" / "Nano" | `#E0E0E0` / `#F4C555` |
| Dark background (GUI window) | `#1E1E1E` |
| Dark panel (GUI panels) | `#252525` |
| Accent (GUI teal), use sparingly | `#209BA5` (darker: `#16868F`) |

Usage notes:

- Use the small pixel-aligned icon at 32 px and below, and the full mark from 48 px up.
- On dark backgrounds avoid the darkest third of `playnano_gold` (below about `#6F2B0A`); it disappears
  on `#1E1E1E`.
- On white, the palest gold (`#FFEEAA`) disappears; use the page 1 or page 3 versions as drawn.
- Don't put gold text on teal or teal text on gold; both are mid-brightness and hard to read together.

## Palettes

`palettes/` holds Inkscape (GIMP-format) palettes:

| File | Contents |
| --- | --- |
| `playnano_gold-17.gpl` | 17 evenly spaced steps through `playnano_gold`, for picking fills |
| `playnano_gold-256.gpl` | all 256 colormap steps |
| `playNano-GUI.gpl` | every colour in the GUI stylesheet, named by where it is used |
| `playnano_gold-gradient.svg` | `playnano_gold` as a 33-stop linear gradient |

To install the palettes, copy the `.gpl` files to Inkscape's palettes folder and restart Inkscape:

- Windows: `%APPDATA%\inkscape\palettes\`
- Linux: `~/.config/inkscape/palettes/`
- macOS: `~/Library/Application Support/org.inkscape.Inkscape/config/inkscape/palettes/`

They then appear in the palette menu at the bottom of the window and in **Object → Swatches**.
Palettes only hold flat colours: to use the gradient, import `playnano_gold-gradient.svg` and pick
**playnano_gold** from the gradient list in **Fill and Stroke**.
