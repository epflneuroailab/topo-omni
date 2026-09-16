"""

this file contains infrastructure for position embeddings
 - partially built on top of spacetorch (github.com/neuroailab/TDANN)
 - infra for loading / saving positions
 - scripts for spatial loss calculation
 - scripts for pre-optimization

"""

import os

import numpy as np
import pickle as pkl
from dataclasses import dataclass

from tqdm import tqdm

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

### CLASS INFRA - LayerPositions + NetworkPositions ###

@dataclass
class LayerPositions:

    name: str # layer name

    # coordinates is an N x 2 matrix with the x-coordinates of each unit in the first
    # column and the y-coordinates in the second column
    coordinates: torch.Tensor

    # neighborhood_indices is a M x N binary matrix, where there are M neighborhoods
    # may be None when neighborhoods are resolved on the fly from `unit_radii`
    neighborhood_indices: Optional[torch.Tensor]

    # number of neighborhoods to compute / average loss over
    neighborhoods_per_batch: int

    # per-unit neighborhood radius (N,). when set, neighborhoods are computed on the
    # fly from the coordinates instead of read out of `neighborhood_indices`
    unit_radii: Optional[torch.Tensor] = None

    # confine every neighborhood to the layer tile its center sits in. Only meaningful on
    # a tiled layout, where one tile IS one layer: at a whole-tile radius an unscoped
    # neighborhood would spill over the tile border into a DIFFERENT layer's units, and
    # the loss would organize the layer grid rather than the layer. Populate `tile_ids`
    # and `tile_members` with `attach_tile_scoping` before use.
    tile_scoped: bool = False
    tile_ids: Optional[torch.Tensor] = None
    tile_members: Optional[torch.Tensor] = None

    def save(self, save_dir: Path):

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok = True, parents = True)
        path = save_dir / f'{self.name}.pkl'

        with path.open('wb') as f:
            pkl.dump(self, f)

    # load positions from disk
    @classmethod
    def load(cls, path: Path) -> 'LayerPositions':

        path = Path(path)

        assert path.exists(), path
        assert path.suffix == '.pkl', 'invalid file, needs to be pickle'

        with path.open('rb') as f:
            return pkl.load(f)

    # helper to put things on gpu
    def to(self, device):
        self.coordinates = self.coordinates.to(device)

        if self.neighborhood_indices is not None:
            self.neighborhood_indices = self.neighborhood_indices.to(device)

        if self.unit_radii is not None:
            self.unit_radii = self.unit_radii.to(device)

        # getattr, not attribute access: a LayerPositions unpickled from a pre-tiling
        # run has no such field at all, and `.to()` runs on every forward.
        if getattr(self, "tile_ids", None) is not None:
            self.tile_ids = self.tile_ids.to(device)

        if getattr(self, "tile_members", None) is not None:
            self.tile_members = self.tile_members.to(device)

        return self


@dataclass
class NetworkPositions:

    # dictionary of all layer positions
    layer_positions: Dict[str, LayerPositions]
    version: int

    # load ALL positions from disk
    @classmethod
    def load_from_dir(cls, load_dir: Path):

        load_dir = Path(load_dir)

        assert load_dir.is_dir(), load_dir
        layer_files = list(load_dir.glob('*.pkl'))

        d = {}

        for layer_file in layer_files:

            layer_name = layer_file.stem
            d[layer_name] = LayerPositions.load(layer_file)

        version_path = load_dir / 'version.txt'
        version = 1.0

        if version_path.is_file():
            with version_path.open('r') as f:
                version = int(f.readline())

        return cls(version = version, layer_positions = d)

### CORTICAL SHEET LAYOUT + DYNAMIC NEIGHBORHOOD SIZES ###

#: Tiled geometry, chosen so the bands still concatenate: the vision and audio bands are
#: joined on COLUMNS (so they must match), and the encoder block is joined to the thinker on
#: ROWS (so 2*W_enc must equal W_thinker). Encoder: 33 tiles of 40x32 in an 11x3 grid ->
#: 440x96 per modality. Thinker: 36 tiles of 64x32 in a 6x6 grid -> 384x192. Total
#: 824x192 = 158,208 units, identical to the strip layout's 309x512.
#: A layer's tile is 1.20 x 0.96 Lambda (at Lambda=33.2 px) instead of 0.15 x 7.71.
#: CONTINUOUS geometry -- the point of which is a sheet with enough EXTENT to carry a
#: metric, so simulated cortical distance can be compared against macaque V1.
#:
#: The gap to TDANN was never density: per hypercolumn we already had 1103 units/Lambda^2
#: against their 94. It was extent -- a layer spanned 1.16 Lambda^2 where TDANN's spans
#: ~1069, and a map is not visible inside one hypercolumn, nor can distance be calibrated
#: from it. So this matches TDANN's PACKING (~94-100 units per Lambda^2) while giving each
#: layer several hypercolumns across.
#:
#: cortical_dim 4096 -> a 64x64 px tile per layer. With the neighbourhood radius set so
#: Lambda ~ 10 px, that is 6.4 x 6.4 Lambda and 100 units/Lambda^2.
#: Calibration: macaque V1 Lambda ~ 0.75 mm / 10 px  =>  MM_PER_PX = 0.075, so a layer
#: spans 4.8 mm and pairwise sheet distances convert straight to mm.
CONTINUOUS_GEOMETRY = dict(
    n_row=1088, n_col=384, encoder_rows=704,
    num_encoder_layers=33, num_thinker_layers=36,
    enc_tile=(64, 64, 11, 3),      # (tile_rows, tile_cols, tiles_down, tiles_across)
    thk_tile=(64, 64, 6, 6),
    cortical_dim=4096,
    mm_per_px=0.075,
)

MACAQUE_LAMBDA_MM = 0.75           #: macaque V1 orientation column spacing
TARGET_LAMBDA_PX = 10              #: the radius schedule should put Lambda near this


#: STRIP geometry with a 4096-wide adaptor: 816x512 = 417,792 units, the same unit count
#: and the same cortical_dim as CONTINUOUS_GEOMETRY, but each layer laid out as a
#: full-width strip rather than a 2-D tile.
#:
#: WHY: continuous-4096 changed two things at once against the original 309x512 strip --
#: the adaptor went 1280 -> 4096 AND the layout went strip -> tiled. This isolates the
#: first. If tangential structure appears here too, extent was never the blocker and the
#: adaptor width was doing the work; if it appears only under tiling, the 2-D tile is.
#:
#: The row counts are forced, not chosen. `_band_from_layers` in strip mode does
#: `flat.reshape(T, band_rows, band_cols)` on `[T, L*cortical_dim]`, so the band area must
#: equal L * cortical_dim exactly:
#:     encoder  33 * 4096 = 135,168 = 528 rows x 256 cols  (n_col // 2)
#:     thinker  36 * 4096 = 147,456 = 288 rows x 512 cols
#:     -> n_row = 528 + 288 = 816
#: which gives 16 rows per encoder slot and 8 per thinker layer, against 5 and 4 on the
#: original strip.
#:
#: NO mm CALIBRATION. A layer here spans 16 x 256 px, so it has extent along the column
#: axis and almost none across it -- the anisotropy that made Lambda on the original strip
#: a blend of two very different per-axis values. mm_per_px stays None, and distances from
#: this layout are px only.
STRIP_4096_GEOMETRY = dict(
    n_row=816, n_col=512, encoder_rows=528,
    num_encoder_layers=33, num_thinker_layers=36,
    cortical_dim=4096,
)


TILED_GEOMETRY = dict(
    n_row=824, n_col=192, encoder_rows=440,
    num_encoder_layers=33, num_thinker_layers=36,
    enc_tile=(40, 32, 11, 3),      # (tile_rows, tile_cols, tiles_down, tiles_across)
    thk_tile=(64, 32, 6, 6),
)


#: TILED-32: the conv front-end slots are GONE, so an encoder contributes exactly its 32
#: transformer blocks and the sheet carries 2*32 + 36 = 100 layer tiles.
#:
#: WHY drop the front-end: it was only ever added so the two encoder bands had equal row
#: counts (see the front-end-slot note in the docs). Under tiling the bands are tiled
#: independently, so the constraint is 32 == enc_grid, not "both bands or neither" -- the
#: slot costs a tile and an adaptor in the residual stream and buys nothing here.
#:
#: WHY 4608 and not 4096: the tile grids are forced by arithmetic. n_col must satisfy
#: BOTH  n_col = 2 * enc_across * enc_tile_cols  AND  n_col = thk_across * thk_tile_cols
#: with enc_down*enc_across = 32 and thk_down*thk_across = 36. At cortical_dim 4096 and
#: 64-wide tiles that gives n_col = 512, and the only thk_across dividing 36 are 1/2/4 ->
#: a 32x128 thinker tile (4:1), so a thinker patch could only ever be a stripe. At 4608
#: the tiles are 64x72 / 72x64, n_col = 576, and 576/64 = 9 DOES divide 36. Both bands
#: then get near-square (1.12) tiles, which is what a round within-layer patch needs.
#:
#:     encoder  32 tiles of 64x72 in an 8x4 grid -> 512 x 288 per modality
#:     thinker  36 tiles of 72x64 in a 4x9 grid  -> 288 x 576
#:     sheet    800 x 576 = 460,800 = 100 * 4608
#:
#: NO mm CALIBRATION. Lambda is set by the neighborhood radius (measured: Lambda ~ 2.4r),
#: and the radius now varies across depth, so there is no single Lambda to calibrate a
#: millimetre scale against. Distances from this layout are px only.
TILED32_GEOMETRY = dict(
    n_row=800, n_col=576, encoder_rows=512,
    num_encoder_layers=32, num_thinker_layers=36,
    enc_tile=(64, 72, 8, 4),
    thk_tile=(72, 64, 4, 9),
    cortical_dim=4608,
    #: Serpentine, so consecutive layers are always spatially adjacent and a cluster can
    #: grow across a tile border into the next layer. Under raster order layer 3 sits at
    #: the right edge of tile row 0 and layer 4 at the LEFT edge of row 1 -- opposite
    #: corners -- so the depth sequence is torn at every row end and the tile grid becomes
    #: visible in the map. The sheet is meant to be unified; this is what makes the
    #: 2-D tiling compatible with that.
    tile_order="serpentine",
)


def tile_grid_pos(layer: int, tiles_across: int, serpentine: bool) -> Tuple[int, int]:
    """Layer index -> (tile_row, tile_col) in its band's tile grid."""
    r, c = divmod(int(layer), tiles_across)
    if serpentine and r % 2 == 1:
        c = tiles_across - 1 - c
    return r, c


def tile_grid_layer(tile_row, tile_col, tiles_across: int, serpentine: bool):
    """(tile_row, tile_col) -> layer index. Vectorised inverse of `tile_grid_pos`.

    Must stay the exact inverse: `_tile_blocks` and the model's `_band_from_layers` place
    a layer's units using the forward map, while `depth_slot` recovers the depth of a unit
    using this one. If they disagree the sheet is scrambled without any error.
    """
    if serpentine:
        tile_col = torch.where(tile_row % 2 == 1, tiles_across - 1 - tile_col, tile_col)
    return tile_row * tiles_across + tile_col


@dataclass
class SheetLayout:
    """
    Geometry of the unified cortical sheet.

    The row index *is* the depth axis. The vision (left half) and audio (right
    half) encoders share the first `encoder_rows` rows, and the thinker occupies
    the remaining rows across the full width. This mirrors exactly the reshapes
    done in `Qwen2_5OmniThinkerForConditionalGeneration.forward`:

        visual / audio : [T, L, D] -> [T, encoder_rows, n_col // 2]
        thinker        : [T, L, D] -> [T, thinker_rows, n_col]

    so encoder slot `l` lives on rows `l * rows_per_encoder_layer ...` and
    thinker layer `l` on rows `encoder_rows + l * rows_per_thinker_layer ...`.

    `num_encoder_layers` counts sheet *slots*, not transformer blocks. Each
    encoder contributes its conv front-end (audio `conv2`, vision `patch_embed`)
    as slot 0 followed by its 32 blocks, so the count is 33 and the two encoders
    stay row-aligned for the column-wise `cat` that builds the unified sheet.
    Both bands must have the same row count, which is why the front-end slot is
    added to vision and audio together or not at all.
    """

    n_row: int = 309
    n_col: int = 512
    encoder_rows: int = 165
    num_encoder_layers: int = 33
    num_thinker_layers: int = 36
    #: "strip" reproduces the original geometry exactly. "tiled" gives every layer a
    #: 2-D tile instead of a full-width strip -- see TILED_GEOMETRY and `SheetLayout.tiled`.
    mode: str = "strip"
    #: (tile_rows, tile_cols, tiles_down, tiles_across) per band; tiled mode only.
    enc_tile: tuple = None
    thk_tile: tuple = None
    #: units a layer contributes to the sheet. None = the layer's own hidden size (square
    #: adaptor). Set it to widen the adaptor and buy sheet EXTENT -- see CONTINUOUS_GEOMETRY.
    cortical_dim: int = None
    #: mm of simulated cortex per sheet pixel, so distances convert to macaque units.
    #: None = the sheet has no calibrated metric and distances are in px only.
    mm_per_px: float = None
    #: How consecutive layers are laid out over the tile grid; tiled mode only.
    #:   "raster"     -- row-major. Layer l sits at (l // across, l % across), so the last
    #:                   tile of a row and the first of the next are at OPPOSITE ends of
    #:                   the band: consecutive layers can be maximally far apart, and a
    #:                   cluster cannot grow across that break.
    #:   "serpentine" -- boustrophedon; odd rows run right-to-left. Every pair of
    #:                   consecutive layers is then spatially ADJACENT, so a cluster is
    #:                   free to span layers, which is the point of a unified sheet.
    #: Default "raster" so every pre-existing tiled layout is bit-identical.
    tile_order: str = "raster"

    def __post_init__(self):
        if self.mode not in ("strip", "tiled"):
            raise ValueError(f"unknown sheet layout mode {self.mode!r}")
        if self.tile_order not in ("raster", "serpentine"):
            raise ValueError(f"unknown tile_order {self.tile_order!r}")
        self.thinker_rows = self.n_row - self.encoder_rows

        if self.mode == "strip":
            if self.encoder_rows % self.num_encoder_layers != 0:
                raise ValueError(
                    f"{self.encoder_rows} encoder rows do not divide evenly over "
                    f"{self.num_encoder_layers} encoder layers"
                )
            if self.thinker_rows % self.num_thinker_layers != 0:
                raise ValueError(
                    f"{self.thinker_rows} thinker rows do not divide evenly over "
                    f"{self.num_thinker_layers} thinker layers"
                )
            self.rows_per_encoder_layer = self.encoder_rows // self.num_encoder_layers
            self.rows_per_thinker_layer = self.thinker_rows // self.num_thinker_layers
        else:
            eth, etw, enr, enc_ = self.enc_tile
            tth, ttw, tnr, tnc = self.thk_tile
            if enr * enc_ != self.num_encoder_layers:
                raise ValueError(f"encoder tile grid {enr}x{enc_} != "
                                 f"{self.num_encoder_layers} slots")
            if tnr * tnc != self.num_thinker_layers:
                raise ValueError(f"thinker tile grid {tnr}x{tnc} != "
                                 f"{self.num_thinker_layers} layers")
            if enr * eth != self.encoder_rows or 2 * enc_ * etw != self.n_col:
                raise ValueError("encoder tiling does not fill the encoder band")
            if tnr * tth != self.thinker_rows or tnc * ttw != self.n_col:
                raise ValueError("thinker tiling does not fill the thinker band")
            # Depth is no longer the row axis under tiling; these stay defined only so
            # that code reading them fails loudly rather than silently mis-binning.
            self.rows_per_encoder_layer = eth
            self.rows_per_thinker_layer = tth

        self.num_units = self.n_row * self.n_col
        #: units a single layer occupies on the sheet
        self.units_per_layer = (self.enc_tile[0] * self.enc_tile[1] if self.mode == "tiled"
                                else self.rows_per_encoder_layer * (self.n_col // 2))

    def mm(self, px):
        """Sheet distance in px -> mm of simulated cortex. Requires a calibrated layout."""
        if self.mm_per_px is None:
            raise ValueError(f"layout '{self.mode}' has no mm calibration; distances are px")
        return px * self.mm_per_px

    # -- constructors ------------------------------------------------------
    @classmethod
    def strip(cls, **kw):
        """The original geometry: 309x512, each layer a full-width strip."""
        return cls(**kw)

    @classmethod
    def strip_4096(cls, **kw):
        """816x512 strips with a 4096-unit adaptor: the continuous run's adaptor width and
        unit count, but the original strip layout. See STRIP_4096_GEOMETRY."""
        g = dict(STRIP_4096_GEOMETRY); g.update(kw)
        return cls(mode="strip", **g)

    @classmethod
    def continuous(cls, **kw):
        """1088x384 with 64x64 tiles and a 4096-unit adaptor: TDANN-like packing, and a
        calibrated mm scale so sheet distance is comparable to macaque V1."""
        g = dict(CONTINUOUS_GEOMETRY); g.update(kw)
        return cls(mode="tiled", **g)

    @classmethod
    def tiled(cls, **kw):
        """824x192, each layer a 2-D tile. Same unit count as `strip`."""
        g = dict(TILED_GEOMETRY)
        g.update(kw)
        return cls(mode="tiled", **g)

    @classmethod
    def tiled32(cls, **kw):
        """800x576, no conv front-end slot: 32 encoder tiles per modality plus 36 thinker
        tiles, all near-square, with a 4608-unit adaptor. See TILED32_GEOMETRY."""
        g = dict(TILED32_GEOMETRY)
        g.update(kw)
        return cls(mode="tiled", **g)

    @property
    def num_tiles(self) -> int:
        """Distinct layer tiles on the sheet: one per encoder layer PER MODALITY, plus
        one per thinker layer."""
        return 2 * self.num_encoder_layers + self.num_thinker_layers

    # -- depth ------------------------------------------------------------
    def depth_slot(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Per-unit depth index: 0..num_encoder_layers-1 in the encoder bands, then
        num_encoder_layers + thinker layer index. The one place that knows how depth is
        embedded in the sheet, so `strip` and `tiled` differ here and nowhere else."""
        rows = coordinates[:, 0].long()
        cols = coordinates[:, 1].long()
        enc = rows < self.encoder_rows
        slot = torch.empty_like(rows)
        if self.mode == "strip":
            slot[enc] = torch.clamp(rows[enc] // self.rows_per_encoder_layer,
                                    max=self.num_encoder_layers - 1)
            slot[~enc] = self.num_encoder_layers + torch.clamp(
                (rows[~enc] - self.encoder_rows) // self.rows_per_thinker_layer,
                max=self.num_thinker_layers - 1)
        else:
            eth, etw, enr, enc_n = self.enc_tile
            tth, ttw, tnr, tnc = self.thk_tile
            # encoder: tile grid is per modality, so fold the column into the half-band
            serp = self.tile_order == "serpentine"
            half = self.n_col // 2
            ec = torch.remainder(cols[enc], half)
            slot[enc] = torch.clamp(
                tile_grid_layer(rows[enc] // eth, ec // etw, enc_n, serp),
                max=self.num_encoder_layers - 1)
            tr = rows[~enc] - self.encoder_rows
            slot[~enc] = self.num_encoder_layers + torch.clamp(
                tile_grid_layer(tr // tth, cols[~enc] // ttw, tnc, serp),
                max=self.num_thinker_layers - 1)
        return slot


    def tile_id(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Per-unit tile index, unique across modality AND depth: 0..num_tiles-1.

        Deliberately NOT `depth_slot`. That one gives the vision and audio units at a
        given depth the SAME id, which is what the radius schedule wants -- radius keys on
        depth alone. Tile-scoped neighborhoods want the opposite: the vision tile and the
        audio tile at depth 7 are two different patches of simulated cortex sitting in
        different columns, and a neighborhood must never span both. Layout:

            vision encoder layer l  ->  l
            audio  encoder layer l  ->  num_encoder_layers + l
            thinker layer l         ->  2 * num_encoder_layers + l
        """
        if self.mode != "tiled":
            raise ValueError("tile_id is only defined for a tiled layout")
        rows = coordinates[:, 0].long()
        cols = coordinates[:, 1].long()
        enc = rows < self.encoder_rows
        audio = enc & (cols >= self.n_col // 2)
        slot = self.depth_slot(coordinates)
        # thinker slots already come back offset by num_encoder_layers, so one more
        # offset of num_encoder_layers puts them at 2*NE + layer.
        tid = slot + self.num_encoder_layers * (~enc).long()
        tid[audio] = slot[audio] + self.num_encoder_layers
        return tid


#: Selected by $TOPO_SHEET_LAYOUT: "strip" (the default, 309x512, 1280-wide adaptor),
#: "strip-4096" (816x512 strips, 4096-wide adaptor), "tiled", or "continuous". Reverting
#: to the original geometry is unsetting the variable -- no code change, no config edit.
_LAYOUT_MODE = os.environ.get("TOPO_SHEET_LAYOUT", "strip").strip().lower()

#: The geometry every eval script should read its sheet dimensions from. Hard-coding
#: `304`/`512` in analysis code is how a layout change turns into a scrambled map
#: rather than an exception -- a stale literal still reshapes cleanly often enough to
#: look plausible. Import this instead.
DEFAULT_SHEET_LAYOUT = (SheetLayout.continuous() if _LAYOUT_MODE == "continuous"
                        else SheetLayout.tiled32() if _LAYOUT_MODE in ("tiled32",
                                                                       "tiled-32")
                        else SheetLayout.tiled() if _LAYOUT_MODE == "tiled"
                        else SheetLayout.strip_4096() if _LAYOUT_MODE in ("strip-4096",
                                                                          "strip_4096")
                        else SheetLayout.strip())


RADIUS_SCHEDULES = ("linear", "geometric", "area-linear")


@dataclass
class RadiusSchedule:
    """
    Neighborhood radius as a function of depth.

    The radius grows from `radius_min` at the first vision / audio encoder layer
    to `radius_max` at the last one; every thinker layer sits at `radius_max`.
    """

    radius_min: float
    radius_max: float
    schedule: str = "linear"

    def encoder_radii(self, num_layers: int) -> np.ndarray:
        if num_layers <= 1:
            return np.array([float(self.radius_max)])

        t = np.linspace(0.0, 1.0, num_layers)

        if self.schedule == "linear":
            return self.radius_min + (self.radius_max - self.radius_min) * t
        if self.schedule == "geometric":
            return self.radius_min * (self.radius_max / self.radius_min) ** t
        if self.schedule == "area-linear":
            # linear in neighborhood *area*, so radius grows as a square root
            return np.sqrt(
                self.radius_min ** 2 + (self.radius_max ** 2 - self.radius_min ** 2) * t
            )

        raise ValueError(
            f"unknown radius schedule '{self.schedule}', expected one of {RADIUS_SCHEDULES}"
        )


def build_unit_radii(
    coordinates: torch.Tensor,
    layout: SheetLayout,
    schedule: RadiusSchedule,
) -> torch.Tensor:
    """
    Per-unit neighborhood radius, keyed on where the unit sits on the sheet.

    Units in the encoder band get their layer's scheduled radius; everything in
    the thinker band gets `radius_max`.
    """

    rows = coordinates[:, 0].long()
    radii = torch.full((coordinates.shape[0],), float(schedule.radius_max), dtype=torch.float32)

    encoder = rows < layout.encoder_rows
    layer_idx = layout.depth_slot(coordinates)[encoder]

    encoder_radii = torch.from_numpy(schedule.encoder_radii(layout.num_encoder_layers)).float()
    radii[encoder] = encoder_radii[layer_idx]

    return radii


def describe_radius_schedule(layout: SheetLayout, schedule: RadiusSchedule) -> str:
    radii = schedule.encoder_radii(layout.num_encoder_layers)
    per_layer = ", ".join(f"{r:.1f}" for r in radii)

    return (
        f"radius schedule '{schedule.schedule}' "
        f"[{schedule.radius_min:g} -> {schedule.radius_max:g}]\n"
        f"   encoder layers 0..{layout.num_encoder_layers - 1}: {per_layer}\n"
        f"   thinker layers 0..{layout.num_thinker_layers - 1}: "
        f"{schedule.radius_max:g} (constant)"
    )


def _tile_blocks(layout: "SheetLayout"):
    """(row_slice, col_slice) for every layer tile, in depth order per band.

    Encoder tiles are laid out per modality: the vision half fills columns
    [0, n_col/2) and the audio half [n_col/2, n_col), each with its own
    `tiles_down x tiles_across` grid, so a slot index means the same depth in both.
    """
    eth, etw, enr, enc_n = layout.enc_tile
    tth, ttw, tnr, tnc = layout.thk_tile
    half = layout.n_col // 2
    blocks = []
    serp = layout.tile_order == "serpentine"
    for col_off in (0, half):                                   # vision, then audio
        for slot in range(layout.num_encoder_layers):
            r, c = tile_grid_pos(slot, enc_n, serp)
            blocks.append((slice(r * eth, (r + 1) * eth),
                           slice(col_off + c * etw, col_off + (c + 1) * etw)))
    for lay in range(layout.num_thinker_layers):
        r, c = tile_grid_pos(lay, tnc, serp)
        blocks.append((slice(layout.encoder_rows + r * tth,
                             layout.encoder_rows + (r + 1) * tth),
                       slice(c * ttw, (c + 1) * ttw)))
    return blocks


def _shuffle_within_tiles(coordinates, layout: "SheetLayout", rng):
    """Tiled counterpart of the strip shuffle: permute coordinates inside each tile.

    Same principle as the strip version -- a layer's units start scattered over the
    positions allotted to that layer, so none begins at a privileged spot -- but the
    allotment is a compact 2-D tile rather than a full-width strip.
    """
    for row_sl, col_sl in _tile_blocks(layout):
        idx = torch.tensor([r * layout.n_col + c
                            for r in range(row_sl.start, row_sl.stop)
                            for c in range(col_sl.start, col_sl.stop)])
        perm = torch.randperm(len(idx), generator=rng)
        coordinates[idx] = coordinates[idx].clone()[perm]
    return coordinates


def build_sheet_coordinates(
    layout: SheetLayout,
    seed: int = 42,
    thinker_rows_per_block: int = 3,
) -> torch.Tensor:
    """
    Cortical coordinates for every unit, shuffled within each layer's own strip.

    The shuffle is the random initialization of the topography: a layer's units
    are scattered across the rows allotted to that layer, so no unit starts out
    at a privileged position.

    NOTE `thinker_rows_per_block` defaults to 3 to stay bit-identical with the
    coordinates `init_coords.py` has always produced, even though the thinker
    actually occupies `layout.rows_per_thinker_layer` (= 4) rows per layer.
    """

    rng = torch.Generator()
    rng.manual_seed(seed)

    coordinates = torch.Tensor(
        [(i, j) for i in range(layout.n_row) for j in range(layout.n_col)]
    )

    if layout.mode == "tiled":
        return _shuffle_within_tiles(coordinates, layout, rng)

    half = layout.n_col // 2
    parts = [
        (0, layout.encoder_rows, 0, half, layout.rows_per_encoder_layer),          # vision encoder
        (0, layout.encoder_rows, half, layout.n_col, layout.rows_per_encoder_layer),  # audio encoder
        (layout.encoder_rows, layout.n_row, 0, layout.n_col, thinker_rows_per_block),  # thinker
    ]

    for row_start, row_end, col_start, col_end, rows_per_block in parts:
        num_blocks = (row_end - row_start) // rows_per_block

        for block_idx in range(num_blocks):
            br_start = row_start + block_idx * rows_per_block
            br_end = br_start + rows_per_block

            # Collect flat indices of coordinates belonging to this block
            indices = []
            for r in range(br_start, br_end):
                for c in range(col_start, col_end):
                    indices.append(r * layout.n_col + c)

            indices = torch.tensor(indices)

            # Random permutation within this block
            perm = torch.randperm(len(indices), generator=rng)

            original_values = coordinates[indices].clone()
            coordinates[indices] = original_values[perm]

    return coordinates

### SPATIAL LOSS ###

# norm implementation for lp-norm
# for some reason this makes things 20x faster than torch.norm????
def p_norm(positions, p='inf'):

    diff = positions[:, None, :] - positions[None, :, :]

    if p == 1:
        return torch.sum(torch.abs(diff), dim=-1)
    if p == 2:
        return torch.sqrt(torch.sum(diff ** 2, dim=-1))
    if p == 'inf':
        return torch.max(torch.abs(diff), dim=-1)[0]

    raise ValueError(f'norm type {p} not supported')

def get_neighborhood(center, positions, radius = 5, p = 'inf'):
    distances = p_norm(positions, p)
    return distances[center] < radius

def depth_band_ids(coordinates: torch.Tensor, layout: "SheetLayout") -> Tuple[torch.Tensor, int]:
    """Per-unit depth band id: one band per encoder layer, then one per thinker layer.

    Used to stratify neighborhood centers over depth. Uniform sampling over units gives
    each band a share proportional to its unit count, which leaves the earliest encoder
    layers at the mercy of sampling noise; stratifying gives every layer a fixed quota.
    """
    band = layout.depth_slot(coordinates)
    return band, layout.num_encoder_layers + layout.num_thinker_layers


def band_center_allocation(
    weights: torch.Tensor, num_centers: int, min_per_band: int = 1
) -> torch.Tensor:
    """Split `num_centers` across bands proportionally to `weights`.

    Every band is guaranteed `min_per_band` centers first, so no layer can be starved;
    the remainder is handed out by weight with largest-remainder rounding.
    """
    num_bands = weights.numel()
    alloc = torch.full((num_bands,), min_per_band, dtype=torch.long, device=weights.device)

    extra = num_centers - int(alloc.sum())
    if extra <= 0:
        return alloc

    w = weights / weights.sum().clamp_min(1e-12)
    add = torch.floor(w * extra).long()
    rem = extra - int(add.sum())
    if rem > 0:
        frac = w * extra - add.float()
        add[torch.argsort(frac, descending=True)[:rem]] += 1

    return alloc + add


def sample_stratified_centers(
    band_ids: torch.Tensor,
    num_bands: int,
    num_centers: int,
    band_weights: Optional[torch.Tensor] = None,
    candidates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`num_centers` centers spread over depth bands.

    With `band_weights` None every band gets an equal share. Otherwise the share is
    proportional to the weight, which is how shallow encoder layers get oversampled: their
    neighborhoods are tiny under a small radius, so each one gives a very noisy correlation
    estimate, and many of them are needed to average that noise down. Oversampling them is
    close to free -- `local_spatial_loss` is O(size^2) in the neighborhood, so a radius-5
    neighborhood costs ~1/160 of a radius-25 one.
    """
    device = band_ids.device
    if band_weights is None:
        band_weights = torch.ones(num_bands, device=device)

    alloc = band_center_allocation(band_weights, num_centers)
    eligible = torch.ones_like(band_ids, dtype=torch.bool) if candidates is None else candidates

    out = []
    for b in range(num_bands):
        want = int(alloc[b])
        if want == 0:
            continue
        members = ((band_ids == b) & eligible).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        pick = (torch.randint(members.numel(), (want,), device=device)
                if want > members.numel()
                else torch.randperm(members.numel(), device=device)[:want])
        out.append(members[pick])

    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=device)


def inverse_radius_band_weights(
    band_ids: torch.Tensor,
    num_bands: int,
    unit_radii: torch.Tensor,
    exponent: float = 2.0,
) -> torch.Tensor:
    """Band weights proportional to ``1 / radius ** exponent``.

    ``exponent=2`` equalises roughly how many unit-memberships each band contributes per
    step, since a neighborhood holds O(r^2) units. Across the 5->25 radius schedule that
    is a 25:1 tilt towards the shallow, small-radius bands.

    ``exponent=1`` is the gentler tilt: still oversamples the shallow bands to average
    down their noisy correlation estimates, but only 5:1, so the deep large-radius bands
    keep a much larger share of the center budget.
    """
    device = band_ids.device
    r = torch.ones(num_bands, device=device)
    for b in range(num_bands):
        m = band_ids == b
        if m.any():
            r[b] = unit_radii[m].float().mean().clamp_min(1.0)
    return 1.0 / (r ** exponent)


def inverse_size_band_weights(
    band_ids: torch.Tensor, num_bands: int, unit_radii: torch.Tensor
) -> torch.Tensor:
    """Band weights proportional to 1 / neighborhood area, i.e. 1 / radius^2.

    Equalises roughly how many unit-memberships each band contributes per step, so a
    radius-5 band is sampled ~25x more often than a radius-25 one.
    """
    return inverse_radius_band_weights(band_ids, num_bands, unit_radii, exponent=2.0)


def resolve_neighborhoods(
    coordinates: torch.Tensor,
    unit_radii: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int = 32,
) -> List[torch.Tensor]:
    """Neighborhood member indices for explicit `centers`, each using its own radius."""
    coordinates = coordinates.float()
    neighborhoods = []
    for start in range(0, centers.shape[0], chunk_size):
        block = centers[start:start + chunk_size]
        distances = torch.cdist(coordinates[block], coordinates)
        within = distances < unit_radii[block].unsqueeze(1)
        for i in range(block.shape[0]):
            neighborhoods.append(within[i].nonzero(as_tuple=True)[0])
    return neighborhoods


def build_tile_members(coordinates: torch.Tensor, layout: "SheetLayout") -> torch.Tensor:
    """`(num_tiles, units_per_tile)` -- the sheet units belonging to each layer tile.

    Every tile holds exactly `cortical_dim` units by construction, so this is a dense
    tensor rather than a ragged list, and `tile_members[t]` is a plain row lookup.
    """
    tile_ids = layout.tile_id(coordinates)
    counts = torch.bincount(tile_ids, minlength=layout.num_tiles)

    if counts.numel() != layout.num_tiles or not bool((counts == counts[0]).all()):
        raise ValueError(
            f"expected {layout.num_tiles} equally sized tiles, got sizes "
            f"{sorted(set(counts.tolist()))}"
        )

    # stable sort keeps tile t's units contiguous, so the reshape is exact
    order = torch.argsort(tile_ids, stable=True)
    return order.reshape(layout.num_tiles, int(counts[0]))


def attach_tile_scoping(positions: "LayerPositions", layout: "SheetLayout") -> "LayerPositions":
    """Turn on tile-scoped neighborhoods for `positions` and cache the bookkeeping."""
    positions.tile_ids = layout.tile_id(positions.coordinates)
    positions.tile_members = build_tile_members(positions.coordinates, layout)
    positions.tile_scoped = True
    return positions


def resolve_tile_neighborhoods(
    coordinates: torch.Tensor,
    unit_radii: torch.Tensor,
    centers: torch.Tensor,
    tile_ids: torch.Tensor,
    tile_members: torch.Tensor,
    max_units: Optional[int] = None,
) -> Tuple[List[torch.Tensor], int]:
    """Neighborhoods confined to the center's own layer tile.

    Two differences from `resolve_neighborhoods`, both of which matter only once the
    radius is large enough to reach a tile border:

    scoping
        candidates are the `cortical_dim` units of the center's tile rather than all
        `num_units` sheet units. That is the point -- a whole-tile radius must organize
        ONE layer, not blend it with the layer tiled next to it -- and it also makes the
        distance matrix `(chunk, cortical_dim)` instead of `(chunk, num_units)`, i.e. 100x
        smaller here, which is what makes a large radius affordable at all.

    subsampling
        a neighborhood over `max_units` is subsampled HERE rather than downstream. At
        whole-tile radius essentially every neighborhood is oversized, so deferring it
        would materialise `num_centers` full index lists first. A random subset of a
        neighborhood is an unbiased sample of its pairs, so this costs statistical power,
        not correctness -- it is exactly the "sample points from the sheet" formulation,
        applied within one layer.

    Returns `(neighborhoods, n_subsampled)`.
    """
    coordinates = coordinates.float()
    center_tiles = tile_ids[centers]
    out: List[Optional[torch.Tensor]] = [None] * centers.shape[0]
    n_subsampled = 0

    for tile in torch.unique(center_tiles):
        sel = (center_tiles == tile).nonzero(as_tuple=True)[0]
        members = tile_members[tile]                                   # (units_per_tile,)
        distances = torch.cdist(coordinates[centers[sel]], coordinates[members])
        within = distances < unit_radii[centers[sel]].unsqueeze(1)

        for i, slot in enumerate(sel.tolist()):
            idx = members[within[i]]
            if max_units is not None and idx.numel() > max_units:
                perm = torch.randperm(idx.numel(), device=idx.device)[:max_units]
                idx = idx[perm]
                n_subsampled += 1
            out[slot] = idx

    return out, n_subsampled


def local_spatial_loss(activations, positions):
    """
    activations: (B, K)  # B = batch/time samples, K = units/points in neighborhood
    positions:   (K, 2)
    """
    K = activations.shape[1]

    activations = activations.float()
    positions = positions.float()

    device = activations.device

    # Pairwise distance-based similarity D (K, K)
    dist = torch.cdist(positions, positions, p=2)          # Euclidean
    D = 1.0 / (1.0 + dist)

    # Correlation among units across the batch dimension (K, K)
    # activations = activations + 1e-6 * torch.randn_like(activations)
    r = torch.corrcoef(activations.T)

    # Lower-triangular indices (exclude diagonal)
    idx = torch.tril_indices(K, K, offset=-1, device=device)
    pairs_r = r[idx[0], idx[1]]
    pairs_D = D[idx[0], idx[1]]

    return 0.5 * (1.0 - torch.corrcoef(torch.stack([pairs_r, pairs_D], dim=0))[0, 1])

def sample_dynamic_neighborhoods(
    coordinates: torch.Tensor,
    unit_radii: torch.Tensor,
    num_neighborhoods: int,
    chunk_size: int = 32,
) -> List[torch.Tensor]:
    """
    Sample `num_neighborhoods` random centers and resolve each one's neighborhood
    on the fly, using that center's own radius.

    Equivalent to reading rows out of a precomputed neighborhood table, except the
    radius may differ per center and nothing has to be baked out to disk. The
    centers are chunked so the (chunk, num_units) distance matrix stays small.
    """

    num_units = coordinates.shape[0]
    centers = torch.randperm(num_units, device=coordinates.device)[:num_neighborhoods]

    coordinates = coordinates.float()

    neighborhoods = []
    for start in range(0, centers.shape[0], chunk_size):
        block = centers[start:start + chunk_size]

        distances = torch.cdist(coordinates[block], coordinates)   # (chunk, num_units)
        within = distances < unit_radii[block].unsqueeze(1)

        for i in range(block.shape[0]):
            neighborhoods.append(within[i].nonzero(as_tuple=True)[0])

    return neighborhoods

def spatial_loss_fn(
    activations,
    positions,
    accum='mean',
    drop_dead_units: bool = False,
    scale_balance: bool = False,
    stratify_by_depth: bool = False,
    center_allocation: str = "uniform",
    layout: Optional["SheetLayout"] = None,
    dead_eps: float = 1e-6,
    weight_clip: float = 10.0,
    max_units: Optional[int] = None,
    stats: Optional[dict] = None,
):
    """Correlation-vs-distance spatial loss over randomly sampled neighborhoods.

    The three flags are all off by default so an existing config reproduces the original
    behaviour bit-for-bit. What each one fixes:

    drop_dead_units
        ``local_spatial_loss`` runs ``torch.corrcoef`` across the batch dimension, which is
        NaN for a unit whose activation never varies. The caller used to discard the whole
        neighborhood in that case. An image-only batch leaves the entire audio band constant
        (and vice versa), so every neighborhood straddling the vision/audio column split died
        -- measured at 123 of 400, and *all* of them contained at least one dead unit. Masking
        the dead units out of the neighborhood instead keeps the live ones training.

    scale_balance
        The loss is built on correlations, which are scale-invariant, so
        ``dL/dactivation`` scales as ``1 / std(activation)``. Early encoder layers have
        ~1.6x the activation std of late ones and therefore receive ~2x less gradient.
        Weighting each neighborhood by its mean unit std cancels that factor. NOTE: the
        weight is detached, so this reweights gradient magnitude without the chain rule
        undoing it -- standardising the activations in-place would NOT work, since
        d(a/s)/da = 1/s reintroduces exactly the factor being removed.

    stratify_by_depth
        Centers are otherwise drawn uniformly over units, so a layer's share of the
        gradient is proportional to its unit count and subject to sampling noise. This
        gives every encoder and thinker layer a fixed quota per step. It also repairs most
        of the sheet-edge deficit: units in layer 0 sit at depth row 0 and belong to ~21
        neighborhoods per 4000 samples against ~31 for interior layers. The neighborhoods
        there stay geometrically truncated (~980 members vs ~1220) -- that is real, and
        padding the depth axis would fabricate adjacency, so it is left alone.
    """
    device = activations.device
    positions.to(device)

    coords = positions.coordinates                             # (N, 2)
    num_units = coords.shape[0]
    num_centers = positions.neighborhoods_per_batch

    # per-unit scale, detached: used both to find dead units and to reweight
    unit_sd = activations.detach().float().std(0)
    live = unit_sd > dead_eps

    if stratify_by_depth:
        if layout is None:
            raise ValueError("stratify_by_depth=True requires `layout`")
        band_ids, num_bands = depth_band_ids(coords, layout)
        if center_allocation in ("inverse-size", "inverse-radius"):
            if positions.unit_radii is None:
                raise ValueError(
                    f"center_allocation='{center_allocation}' needs dynamic unit_radii"
                )
            # inverse-size is 1/r^2 (equal unit-memberships per band); inverse-radius is
            # the 1/r variant, which leaves the deep large-radius bands a bigger share.
            exponent = 2.0 if center_allocation == "inverse-size" else 1.0
            band_weights = inverse_radius_band_weights(
                band_ids, num_bands, positions.unit_radii, exponent=exponent
            )
        elif center_allocation == "uniform":
            band_weights = None
        else:
            raise ValueError(f"unknown center_allocation '{center_allocation}'")
        # Never spend a center on a dead unit: its neighborhood would be dropped for
        # having < 3 live members. With image-only batches the whole audio band is dead,
        # which otherwise wasted ~36% of the budget.
        centers = sample_stratified_centers(
            band_ids, num_bands, num_centers, band_weights,
            candidates=live if drop_dead_units else None,
        )
    else:
        centers = torch.randperm(num_units, device=device)[:num_centers]

    # --- centre reuse across an accumulation group -------------------------------
    # `spatial_loss_fn` runs once per FORWARD, so with grad-accum G an optimizer step
    # already averages G independent centre draws. `center_reuse = G` instead holds one
    # draw for the whole group, so those G gradients concern the SAME neighbourhoods over
    # different data -- they average into one estimate rather than summing G disjoint ones.
    #
    # This is a trade, not a free win: a step then covers `num_centers` centres with G x
    # the data each, instead of G x num_centers centres with 1x data. It helps if the noise
    # is dominated by the correlation estimate, and hurts if it is dominated by centre
    # coverage. Default 1 reproduces the original per-forward redraw exactly.
    reuse = int(getattr(positions, "center_reuse", 1) or 1)
    if reuse > 1:
        n_calls = getattr(positions, "_center_calls", 0)
        cached = getattr(positions, "_center_cache", None)
        if n_calls % reuse != 0 and cached is not None and cached.numel() == centers.numel():
            centers = cached.to(device)
        else:
            positions._center_cache = centers.detach().clone()
            # First draw of a frozen run: persist it so the eval can restrict its metrics
            # to the units that actually received gradient. Only those neighbourhoods are
            # ever optimised when the centres never change, so a sheet-wide statistic would
            # be diluted by units the loss never touched.
            dump = os.environ.get("TOPO_FROZEN_CENTERS_OUT")
            if dump and not os.path.exists(dump):
                os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
                np.save(dump, centers.detach().cpu().numpy())
                print(f">> froze {centers.numel()} spatial-loss centres -> {dump}")
        positions._center_calls = n_calls + 1

    n_tile_subsampled = 0
    if getattr(positions, "tile_scoped", False):
        # tile-scoped: each neighborhood stays inside its center's own layer tile, and is
        # subsampled to `max_units` there rather than in the loop below.
        if positions.unit_radii is None:
            raise ValueError("tile_scoped neighborhoods require dynamic `unit_radii`")
        if positions.tile_ids is None or positions.tile_members is None:
            raise ValueError("tile_scoped positions need `attach_tile_scoping` first")
        neighborhoods, n_tile_subsampled = resolve_tile_neighborhoods(
            coords, positions.unit_radii, centers,
            positions.tile_ids, positions.tile_members, max_units=max_units,
        )
    elif positions.unit_radii is not None:
        # dynamic neighborhoods: radius grows with depth, resolved on the fly
        neighborhoods = resolve_neighborhoods(coords, positions.unit_radii, centers)
    else:
        # fixed-radius neighborhoods, read out of the precomputed table
        neigh_idx_all = positions.neighborhood_indices         # (M, K) long
        neighborhoods = []
        for i in centers:
            idx = neigh_idx_all[i].squeeze(-1)                 # (K,)
            neighborhoods.append(idx[idx >= 0])

    global_sd = unit_sd[live].mean().clamp_min(dead_eps) if live.any() \
        else torch.ones((), device=device)

    losses, weights = [], []
    n_dropped_small, n_dropped_nan, n_masked_units = 0, 0, 0
    n_subsampled = 0

    for idx in neighborhoods:
        if drop_dead_units:
            before = idx.numel()
            idx = idx[live[idx]]
            n_masked_units += before - idx.numel()

        if idx.numel() < 3:
            # too few units for a meaningful pairwise correlation
            n_dropped_small += 1
            continue

        # Cap the neighborhood. local_spatial_loss builds a (K, K) distance and a (K, K)
        # correlation matrix and keeps both in the autograd graph, so peak memory grows as
        # K^2 per neighborhood. A radius-25 neighborhood holds ~2000 units (4M entries
        # each); with a large center budget that fragments the allocator badly enough to
        # OOM the task loss's logits allocation. A random subset of a neighborhood is an
        # unbiased sample of its pairs, so this costs statistical power, not correctness.
        if max_units is not None and idx.numel() > max_units:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_units]]
            n_subsampled += 1

        loss = local_spatial_loss(activations[:, idx], coords[idx])
        if torch.isnan(loss):
            n_dropped_nan += 1
            continue

        losses.append(loss)
        if scale_balance:
            w = (unit_sd[idx].mean() / global_sd).clamp(1.0 / weight_clip, weight_clip)
            weights.append(w)

    if stats is not None:
        stats.update(
            n_neighborhoods=len(neighborhoods),
            n_kept=len(losses),
            n_dropped_small=n_dropped_small,
            n_dropped_nan=n_dropped_nan,
            n_masked_units=n_masked_units,
            n_subsampled=n_subsampled + n_tile_subsampled,
            n_dead_units=int((~live).sum().item()),
        )

    if not losses:
        # keep the graph alive so DDP still sees a gradient for every parameter
        return activations.sum() * 0.0

    losses = torch.stack(losses)

    if accum == 'mean':
        if scale_balance:
            w = torch.stack(weights)
            return (losses * w).sum() / w.sum().clamp_min(dead_eps)
        return losses.mean()
    elif accum == 'maximum':
        return losses.max()
    else:
        raise ValueError("invalid accumulation function")


# binary mask of points within 'radius' of center
def get_neighborhood(center, distances, radius = 5):
    return distances[center] < radius

# randomly select a center point for a neighborhood (avoiding the edges of the layer)
def get_center(positions, radius, rng=None):

    N = int(np.sqrt(positions.shape[0]))

    mask = (positions[:, 0] >= radius - 1) & (positions[:, 0] <= N - radius) & \
           (positions[:, 1] >= radius - 1) & (positions[:, 1] <= N - radius)

    indices = torch.nonzero(torch.Tensor(mask)).flatten()

    return indices[torch.randint(0, len(indices), (1,), generator = rng).item()]

def neighborhoods_chebyshev_grid(H, W, radius, device=None):
    k = 2 * radius - 1
    pad = radius - 1
    device = device or "cpu"

    idx = torch.arange(H * W, device=device).view(1, 1, H, W).long()

    # pad with -1 so "outside" is clearly invalid
    idx_pad = F.pad(idx, (pad, pad, pad, pad), mode="constant", value=-1)

    patches = F.unfold(idx_pad.float(), kernel_size=(k, k), padding=0, stride=1).long()
    neighborhoods = patches.T  # (H*W, k*k) because we padded and used padding=0

    return neighborhoods  # contains -1 for out-of-bounds

def p_norm_numpy(positions, p = 'inf'):

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    if p == 1:
        return np.sum(np.abs(diff), axis = -1)
    if p == 2:
        return np.sqrt(np.sum(diff ** 2, axis = -1))
    if p == 'inf':
        return np.max(np.abs(diff), axis = -1)

    raise ValueError(f'norm type {p} not supported')

def linf_dist_blocked(positions, block=4000):
    N = positions.shape[0]
    D = np.empty((N, N), dtype=positions.dtype)

    for i in tqdm(range(0, N, block)):
        i2 = min(i + block, N)
        positions_i = positions[i:i2]

        for j in range(0, N, block):
            j2 = min(j + block, N)
            positions_j = positions[j:j2]

            diff = positions_i[:, None, :] - positions_j[None, :, :]
            D[i:i2, j:j2] = np.max(np.abs(diff), axis=-1)

    return D

def local_spatial_loss_numpy(activations, positions):

    num_units = activations.shape[1]
    idx = np.tril_indices(num_units, k = -1)

    D = 1 / (1 + p_norm_numpy(positions, 2))
    r = np.corrcoef(activations.T)

    return 0.5 * (1 - np.corrcoef(r[idx], D[idx])[0][1])

def spatial_loss_numpy(activations, positions):

    cur_loss = []
    neighborhoods = np.random.choice(len(positions.neighborhood_indices), positions.neighborhoods_per_batch, replace = False)

    for i in neighborhoods:
        mask = positions.neighborhood_indices[i].astype(bool)
        cur_loss.append(local_spatial_loss_numpy(activations[:, mask], positions.coordinates[mask]))

    return np.stack(cur_loss).mean()
