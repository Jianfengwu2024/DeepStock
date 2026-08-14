"""Discovery and validation of immutable sector DuckDB datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SectorDataset:
    dataset_id: str
    name: str
    database_path: str
    source: str
    source_code: str
    member_count: int
    available_member_count: int
    data_start: str | None
    data_end: str | None
    description: str
    symbols: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return payload


class SectorDatasetCatalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def list(self) -> list[SectorDataset]:
        if not self.root.is_dir():
            return []
        datasets: list[SectorDataset] = []
        for path in sorted(self.root.glob("*.duckdb")):
            try:
                datasets.extend(self._read(path))
            except Exception:
                continue
        return sorted(datasets, key=lambda item: item.name)

    def get(self, dataset_id: str) -> SectorDataset | None:
        return next(
            (item for item in self.list() if item.dataset_id == dataset_id), None
        )

    def _read(self, path: Path) -> list[SectorDataset]:
        resolved = path.resolve()
        if resolved.parent != self.root:
            raise ValueError("sector dataset is outside the configured root")
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sector dataset discovery requires DuckDB") from exc
        with duckdb.connect(str(resolved), read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT dataset_id, name, source, source_code, member_count,
                       available_member_count, data_start, data_end, description
                FROM sector_datasets ORDER BY dataset_id
                """
            ).fetchall()
            result: list[SectorDataset] = []
            for row in rows:
                symbols = tuple(
                    str(item[0])
                    for item in connection.execute(
                        """
                        SELECT symbol FROM sector_dataset_members
                        WHERE dataset_id = ? AND available
                        ORDER BY symbol
                        """,
                        [row[0]],
                    ).fetchall()
                )
                result.append(
                    SectorDataset(
                        dataset_id=str(row[0]),
                        name=str(row[1]),
                        database_path=str(resolved),
                        source=str(row[2]),
                        source_code=str(row[3]),
                        member_count=int(row[4]),
                        available_member_count=int(row[5]),
                        data_start=str(row[6]) if row[6] is not None else None,
                        data_end=str(row[7]) if row[7] is not None else None,
                        description=str(row[8]),
                        symbols=symbols,
                    )
                )
        return result
