"""Fixed-grid, human-inspectable holographic market-state images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import HOLOGRAPHIC_FIELD_NAMES, ProfileConfig
from .data.holographic import HolographicProfileBuilder
from .data.source import DuckDBBarSource
from .types import BarBatch, HolographicState


VISUAL_DATASET_PROTOCOL = "fixed-holographic-human-view-v1"


@dataclass(frozen=True, slots=True)
class VisualDatasetConfig:
    """Immutable visual coordinates shared by every exported trading day."""

    dataset_id: str = "semiconductor-2024-2025"
    start: str = "2024-01-01"
    end: str = "2026-01-01"
    bin_width_bps: float = 10.0
    price_bins_each_side: int = 420
    volume_bins: int = 21
    normalization_lookback_sessions: int = 10
    velocity_aggregation_minutes: int = 10
    feature_clip: float = 4.0
    image_width: int = 960
    image_height: int = 540
    vector_price_block_bins: int = 12
    vector_min_mass: float = 2e-4
    vector_min_magnitude: float = 0.08
    vector_max_pixels: float = 13.0
    density_gamma: float = 0.55
    density_ceiling: float = 0.04

    def __post_init__(self) -> None:
        if not self.dataset_id or "/" in self.dataset_id or "\\" in self.dataset_id:
            raise ValueError("dataset_id must be a simple non-empty name")
        if date.fromisoformat(self.end) <= date.fromisoformat(self.start):
            raise ValueError("end must be later than start")
        if self.bin_width_bps <= 0 or self.price_bins_each_side < 10:
            raise ValueError("fixed price grid must be positive")
        if self.volume_bins < 3 or self.volume_bins % 2 == 0:
            raise ValueError("volume_bins must be an odd integer of at least three")
        if self.normalization_lookback_sessions < 2:
            raise ValueError("normalization lookback must be at least two sessions")
        if self.image_width < 640 or self.image_height < 360:
            raise ValueError("image canvas is too small for readable annotations")
        if self.density_ceiling <= 0 or self.density_gamma <= 0:
            raise ValueError("density encoding parameters must be positive")

    @property
    def price_bins(self) -> int:
        return self.price_bins_each_side * 2

    @property
    def minimum_price_edge(self) -> int:
        return -self.price_bins_each_side

    @property
    def maximum_price_edge(self) -> int:
        return self.price_bins_each_side

    @property
    def log_bin_width(self) -> float:
        return float(np.log1p(self.bin_width_bps / 10_000.0))

    @property
    def minimum_simple_return(self) -> float:
        return float(np.expm1(self.minimum_price_edge * self.log_bin_width))

    @property
    def maximum_simple_return(self) -> float:
        return float(np.expm1(self.maximum_price_edge * self.log_bin_width))


@dataclass(frozen=True, slots=True)
class FixedHolographicRaster:
    """A zero-padded price x relative-volume raster on global coordinates."""

    density: NDArray[np.float32]
    price_velocity: NDArray[np.float32]
    volume_velocity: NDArray[np.float32]
    source_price_bins: int
    copied_price_bins: int
    out_of_range_mass: float


def rasterize_state(
    state: HolographicState, config: VisualDatasetConfig
) -> FixedHolographicRaster:
    """Project a variable-width daily field onto the shared fixed grid."""

    if not np.isclose(state.log_bin_width, config.log_bin_width, rtol=0, atol=1e-12):
        raise ValueError("state and visual grid use different price-bin widths")
    if len(state.volume_coordinates) != config.volume_bins:
        raise ValueError("state and visual grid use different volume-bin counts")
    shape = (config.price_bins, config.volume_bins)
    density = np.zeros(shape, dtype=np.float32)
    price_velocity = np.zeros(shape, dtype=np.float32)
    volume_velocity = np.zeros(shape, dtype=np.float32)

    fixed_first_center = config.minimum_price_edge + 0.5
    destinations = np.rint(state.bin_coordinates - fixed_first_center).astype(np.int64)
    valid = (destinations >= 0) & (destinations < config.price_bins)
    field_density = np.asarray(state.fields[..., 0], dtype=np.float64)
    out_of_range_mass = float(field_density[~valid].sum())
    if np.any(valid):
        destination = destinations[valid]
        source_density = field_density[valid]
        density[destination] = source_density.astype(np.float32)
        safe_density = np.maximum(source_density, 1e-12)
        price_velocity[destination] = (
            np.asarray(state.fields[valid, :, 1], dtype=np.float64) / safe_density
        ).astype(np.float32)
        volume_velocity[destination] = (
            np.asarray(state.fields[valid, :, 2], dtype=np.float64) / safe_density
        ).astype(np.float32)
        support = source_density <= 1e-12
        price_velocity[destination] *= (~support).astype(np.float32)
        volume_velocity[destination] *= (~support).astype(np.float32)
    return FixedHolographicRaster(
        density=density,
        price_velocity=price_velocity,
        volume_velocity=volume_velocity,
        source_price_bins=len(state.bin_coordinates),
        copied_price_bins=int(valid.sum()),
        out_of_range_mass=out_of_range_mass,
    )


def vector_length(magnitude: float, config: VisualDatasetConfig) -> float:
    """Map standardized local speed monotonically to a bounded arrow length."""

    return min(
        config.vector_max_pixels,
        max(0.0, float(magnitude))
        / config.feature_clip
        * config.vector_max_pixels,
    )


def render_state_png(
    state: HolographicState,
    raster: FixedHolographicRaster,
    destination: str | Path,
    config: VisualDatasetConfig,
) -> None:
    """Render one fixed-size annotated image without changing state geometry."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("visual dataset export requires Pillow") from exc

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (config.image_width, config.image_height), "#f6f7f7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    plot_left = 76
    plot_top = 104
    plot_right = config.image_width - 28
    plot_bottom = config.image_height - 82
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    scaled = np.clip(raster.density.T / config.density_ceiling, 0.0, 1.0)
    scaled = np.power(scaled, config.density_gamma)
    gray = np.asarray(np.rint(255.0 - scaled * 222.0), dtype=np.uint8)
    density_image = Image.fromarray(gray, mode="L").resize(
        (plot_width, plot_height), resample=Image.Resampling.NEAREST
    ).convert("RGB")
    image.paste(density_image, (plot_left, plot_top))

    zero_x = _price_to_pixel(0.0, plot_left, plot_width, config)
    zero_y = _volume_to_pixel(0.0, plot_top, plot_height, config)
    draw.line((zero_x, plot_top, zero_x, plot_bottom), fill="#ba4740", width=2)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill="#78858b", width=1)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#68767c")

    _draw_vectors(draw, raster, config, plot_left, plot_top, plot_width, plot_height)
    _draw_axes(draw, font, config, plot_left, plot_top, plot_right, plot_bottom)
    _draw_header(draw, font, state, config)
    _draw_legend(draw, font, config, plot_left, plot_bottom)
    image.save(destination, format="PNG", optimize=True, compress_level=9)


def _draw_header(draw: Any, font: Any, state: HolographicState, config: VisualDatasetConfig) -> None:
    session = str(state.period_start.astype("datetime64[D]"))
    values = (state.start_price, state.high_price, state.low_price, state.end_price)
    changes = tuple(value / state.anchor_price - 1.0 for value in values)
    direction = "#b53d35" if state.end_price >= state.start_price else "#087f5b"
    draw.text((28, 18), f"{state.symbol}   {session}", fill="#172126", font=font)
    draw.text(
        (28, 43),
        f"Prev {state.anchor_price:.3f}    O {values[0]:.3f} ({changes[0]:+.2%})    "
        f"H {values[1]:.3f} ({changes[1]:+.2%})    L {values[2]:.3f} ({changes[2]:+.2%})    "
        f"C {values[3]:.3f} ({changes[3]:+.2%})",
        fill=direction,
        font=font,
    )
    draw.text(
        (28, 68),
        f"{config.bin_width_bps:g} BP/bin | causal {state.normalization_sessions}-session scale | "
        f"bars {state.bar_count} | flow samples {state.flow_samples}",
        fill="#66737a",
        font=font,
    )


def _draw_axes(
    draw: Any,
    font: Any,
    config: VisualDatasetConfig,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    width = right - left
    height = bottom - top
    for percent in (-30, -20, -10, 0, 10, 20, 30, 40):
        value = percent / 100.0
        if not config.minimum_simple_return <= value <= config.maximum_simple_return:
            continue
        x = _price_to_pixel(math.log1p(value), left, width, config)
        draw.line((x, bottom, x, bottom + 5), fill="#68767c")
        label = f"{percent:+d}%" if percent else "0%"
        draw.text((x - 11, bottom + 9), label, fill="#526068", font=font)
    for sigma in (-4, -2, 0, 2, 4):
        y = _volume_to_pixel(sigma / config.feature_clip, top, height, config)
        draw.line((left - 5, y, left, y), fill="#68767c")
        draw.text((28, y - 5), f"{sigma:+d}s", fill="#526068", font=font)
    draw.text((left + width // 2 - 72, bottom + 31), "log price vs previous close", fill="#526068", font=font)
    draw.text((8, top - 18), "relative volume", fill="#526068", font=font)


def _draw_legend(
    draw: Any,
    font: Any,
    config: VisualDatasetConfig,
    left: int,
    bottom: int,
) -> None:
    y = bottom + 53
    for offset in range(100):
        shade = int(255 - 222 * offset / 99)
        draw.line((left + offset, y, left + offset, y + 8), fill=(shade, shade, shade))
    draw.rectangle((left, y, left + 100, y + 8), outline="#68767c")
    draw.text((left + 108, y - 2), "density", fill="#526068", font=font)
    draw.line((left + 196, y + 4, left + 220, y + 4), fill="#137b8f", width=2)
    _draw_arrow_head(draw, left + 196, y + 4, left + 220, y + 4, "#137b8f")
    draw.text((left + 228, y - 2), "direction + standardized speed", fill="#526068", font=font)


def _draw_vectors(
    draw: Any,
    raster: FixedHolographicRaster,
    config: VisualDatasetConfig,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    block = config.vector_price_block_bins
    for start in range(0, config.price_bins, block):
        stop = min(start + block, config.price_bins)
        density = raster.density[start:stop]
        mass = density.sum(axis=0)
        for volume_index in np.flatnonzero(mass >= config.vector_min_mass):
            weights = density[:, volume_index]
            total = float(mass[volume_index])
            u = float(np.sum(weights * raster.price_velocity[start:stop, volume_index]) / total)
            v = float(np.sum(weights * raster.volume_velocity[start:stop, volume_index]) / total)
            magnitude = math.hypot(u, v)
            if magnitude < config.vector_min_magnitude:
                continue
            length = vector_length(magnitude, config)
            center_bin = (start + stop) / 2.0
            x = left + center_bin / config.price_bins * width
            y = top + (config.volume_bins - volume_index - 0.5) / config.volume_bins * height
            dx = u / magnitude * length
            dy = -v / magnitude * length
            x0, y0, x1, y1 = x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2
            color = "#137b8f"
            draw.line((x0, y0, x1, y1), fill=color, width=2)
            _draw_arrow_head(draw, x0, y0, x1, y1, color)


def _draw_arrow_head(draw: Any, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 3.5
    left = (x1 - size * math.cos(angle - 0.6), y1 - size * math.sin(angle - 0.6))
    right = (x1 - size * math.cos(angle + 0.6), y1 - size * math.sin(angle + 0.6))
    draw.polygon(((x1, y1), left, right), fill=color)


def _price_to_pixel(log_return: float, left: int, width: int, config: VisualDatasetConfig) -> float:
    minimum = config.minimum_price_edge * config.log_bin_width
    maximum = config.maximum_price_edge * config.log_bin_width
    return left + (log_return - minimum) / (maximum - minimum) * width


def _volume_to_pixel(value: float, top: int, height: int, config: VisualDatasetConfig) -> float:
    normalized = (value + 1.0) / 2.0
    return top + (1.0 - normalized) * height


class VisualDatasetIndex:
    """Read-only paginated access to one generated visual dataset."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def database_path(self) -> Path:
        return self.root / "index.sqlite3"

    def summary(self) -> dict[str, Any]:
        manifest_path = self.root / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        if not self.database_path.is_file():
            return {**manifest, "available": False, "image_count": 0, "symbols": []}
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(session), MAX(session) FROM images"
            ).fetchone()
            symbols = [item[0] for item in connection.execute("SELECT DISTINCT symbol FROM images ORDER BY symbol")]
        return {
            **manifest,
            "available": True,
            "image_count": int(row[0]),
            "symbol_count": int(row[1]),
            "data_start": row[2],
            "data_end": row[3],
            "symbols": symbols,
        }

    def query(
        self,
        *,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        page: int = 1,
        page_size: int = 24,
    ) -> dict[str, Any]:
        if not self.database_path.is_file():
            return {"items": [], "total": 0, "page": page, "page_size": page_size}
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if start:
            clauses.append("session >= ?")
            params.append(start)
        if end:
            clauses.append("session <= ?")
            params.append(end)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            total = int(connection.execute(f"SELECT COUNT(*) FROM images{where}", params).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM images{where} ORDER BY session DESC, symbol LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def image_path(self, symbol: str, session: str) -> Path | None:
        if not self.database_path.is_file():
            return None
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT image_path FROM images WHERE symbol = ? AND session = ?",
                [symbol, session],
            ).fetchone()
        if row is None:
            return None
        path = (self.root / row[0]).resolve()
        if self.root not in path.parents or not path.is_file():
            return None
        return path


class VisualDatasetExporter:
    def __init__(
        self,
        database_path: str,
        output_root: str | Path,
        config: VisualDatasetConfig,
    ) -> None:
        self.source = DuckDBBarSource(database_path)
        self.database_path = str(Path(database_path).expanduser().resolve())
        self.output_root = Path(output_root).expanduser().resolve()
        self.config = config

    def export(self, symbols: Sequence[str]) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.output_root / "index.sqlite3")
        self._initialize_database(connection)
        started = datetime.now(timezone.utc)
        exported = 0
        skipped = 0
        errors: dict[str, str] = {}
        for index, symbol in enumerate(symbols, start=1):
            try:
                states = self._states(symbol)
                for state in states:
                    session = str(state.period_start.astype("datetime64[D]"))
                    if not self.config.start <= session < self.config.end:
                        continue
                    relative_path = Path("images", symbol, f"{session}.png")
                    target = self.output_root / relative_path
                    raster = rasterize_state(state, self.config)
                    if target.is_file():
                        skipped += 1
                    else:
                        render_state_png(state, raster, target, self.config)
                        exported += 1
                    self._upsert(connection, state, raster, relative_path)
                connection.commit()
                print(
                    f"[{index}/{len(symbols)}] {symbol}: {len(states)} states, "
                    f"exported={exported}, existing={skipped}",
                    flush=True,
                )
            except Exception as exc:  # keep a long sector export resumable
                connection.rollback()
                errors[symbol] = str(exc)
                print(f"[{index}/{len(symbols)}] {symbol}: ERROR {exc}", flush=True)
        self._write_csv(connection)
        count = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        symbol_count = int(connection.execute("SELECT COUNT(DISTINCT symbol) FROM images").fetchone()[0])
        connection.close()
        manifest = {
            "protocol": VISUAL_DATASET_PROTOCOL,
            "dataset_id": self.config.dataset_id,
            "source": self.database_path,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "status": "complete" if not errors else "complete_with_errors",
            "image_count": count,
            "symbol_count": symbol_count,
            "requested_symbols": len(symbols),
            "new_images": exported,
            "existing_images": skipped,
            "errors": errors,
            "tensor": {
                "layout": "price_bin x relative_volume_bin",
                "width": self.config.price_bins,
                "height": self.config.volume_bins,
                "padding": "exact zeros outside each day's occupied support",
            },
            "encoding": {
                "x": "log(price / previous_close)",
                "x_bin_width_bps": self.config.bin_width_bps,
                "x_min_simple_return": self.config.minimum_simple_return,
                "x_max_simple_return": self.config.maximum_simple_return,
                "y": "causally standardized log1p(relative minute volume)",
                "grayscale": "density with fixed ceiling and gamma",
                "density_ceiling": self.config.density_ceiling,
                "density_gamma": self.config.density_gamma,
                "vectors": "density-weighted local price and volume velocity; length is standardized speed",
            },
            "image": {"width": self.config.image_width, "height": self.config.image_height},
            "config": asdict(self.config),
        }
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def _states(self, symbol: str) -> list[HolographicState]:
        warmup_start = f"{date.fromisoformat(self.config.start).year - 1}-10-01"
        bars = self.source.fetch(symbol, "1m", warmup_start, self.config.end)
        profile = ProfileConfig(
            bin_width_bps=self.config.bin_width_bps,
            max_profile_bins=self.config.price_bins,
            relative_volume_lookback_sessions=self.config.normalization_lookback_sessions,
            relative_volume_min_observations=self.config.normalization_lookback_sessions,
            velocity_aggregation_minutes=self.config.velocity_aggregation_minutes,
            holographic_volume_bins=self.config.volume_bins,
            holographic_fields=HOLOGRAPHIC_FIELD_NAMES,
            feature_clip=self.config.feature_clip,
        )
        states = HolographicProfileBuilder(profile).build(bars)
        return self._attach_full_session_ohlc(states, bars)

    @staticmethod
    def _attach_full_session_ohlc(
        states: Sequence[HolographicState], bars: BarBatch
    ) -> list[HolographicState]:
        """Keep flow on continuous trading bars but annotate full-session OHLC."""

        sessions = bars.timestamp.astype("datetime64[D]")
        by_session: dict[np.datetime64, tuple[float, float, float, float]] = {}
        for session in np.unique(sessions):
            indices = np.flatnonzero(sessions == session)
            day = bars.take(indices)
            by_session[session] = (
                float(day.open[0]),
                float(day.high.max()),
                float(day.low.min()),
                float(day.close[-1]),
            )
        result: list[HolographicState] = []
        for state in states:
            session = state.period_start.astype("datetime64[D]")
            ohlc = by_session.get(session)
            result.append(
                state
                if ohlc is None
                else replace(
                    state,
                    start_price=ohlc[0],
                    high_price=ohlc[1],
                    low_price=ohlc[2],
                    end_price=ohlc[3],
                )
            )
        return result

    @staticmethod
    def _initialize_database(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                symbol TEXT NOT NULL,
                session TEXT NOT NULL,
                image_path TEXT NOT NULL,
                previous_close REAL NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                open_return REAL NOT NULL,
                high_return REAL NOT NULL,
                low_return REAL NOT NULL,
                close_return REAL NOT NULL,
                bar_count INTEGER NOT NULL,
                flow_samples INTEGER NOT NULL,
                density_max REAL NOT NULL,
                velocity_max REAL NOT NULL,
                out_of_range_mass REAL NOT NULL,
                source_price_bins INTEGER NOT NULL,
                copied_price_bins INTEGER NOT NULL,
                PRIMARY KEY (symbol, session)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS images_session ON images(session)")

    @staticmethod
    def _upsert(
        connection: sqlite3.Connection,
        state: HolographicState,
        raster: FixedHolographicRaster,
        relative_path: Path,
    ) -> None:
        session = str(state.period_start.astype("datetime64[D]"))
        magnitude = np.hypot(raster.price_velocity, raster.volume_velocity)
        values = (
            state.symbol,
            session,
            relative_path.as_posix(),
            state.anchor_price,
            state.start_price,
            state.high_price,
            state.low_price,
            state.end_price,
            state.start_price / state.anchor_price - 1.0,
            state.high_price / state.anchor_price - 1.0,
            state.low_price / state.anchor_price - 1.0,
            state.end_price / state.anchor_price - 1.0,
            state.bar_count,
            state.flow_samples,
            float(raster.density.max()),
            float(magnitude.max()),
            raster.out_of_range_mass,
            raster.source_price_bins,
            raster.copied_price_bins,
        )
        connection.execute(
            "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )

    def _write_csv(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute("SELECT * FROM images ORDER BY symbol, session")
        with (self.output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(item[0] for item in cursor.description)
            writer.writerows(cursor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bnhmm-visual-dataset")
    parser.add_argument("--duckdb-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--dataset-id", default="semiconductor-2024-2025")
    parser.add_argument("--price-bins-each-side", type=int, default=420)
    parser.add_argument("--volume-bins", type=int, default=21)
    parser.add_argument("--density-ceiling", type=float, default=0.04)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VisualDatasetConfig(
        dataset_id=args.dataset_id,
        start=args.start,
        end=args.end,
        price_bins_each_side=args.price_bins_each_side,
        volume_bins=args.volume_bins,
        density_ceiling=args.density_ceiling,
    )
    source = DuckDBBarSource(args.duckdb_path)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        symbols = source.list_symbols()
    manifest = VisualDatasetExporter(
        args.duckdb_path, args.output_root, config
    ).export(symbols)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not manifest["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
