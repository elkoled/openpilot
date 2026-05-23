import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
  dongle_id: str
  route_id: str
  branch: str = ""
  rlog_url: str = ""        # if set, extract loads this directly (per-segment)

  @property
  def route_key(self) -> str:
    # canonical key: for per-route, dongle/route; for per-segment, the full segment id
    if self.rlog_url:
      return self.route_id
    return f"{self.dongle_id}/{self.route_id}"


def load_manifest(path: str | Path) -> list[ManifestRow]:
  """Accepts two CSV schemas:
    (A) dongle_id, route_id[, branch]                     # route-level (uses /a auto-fallback)
    (B) segment, dongle, platform, mean_v, max_v,         # per-segment eps_seglist.csv format
        engaged_frac, rlog_url
  """
  rows: list[ManifestRow] = []
  with open(path, newline="") as f:
    reader = csv.DictReader(f)
    fields = set(reader.fieldnames or [])
    if {"rlog_url", "segment", "dongle"}.issubset(fields):
      for raw in reader:
        seg = (raw.get("segment") or "").strip()
        url = (raw.get("rlog_url") or "").strip()
        dongle = (raw.get("dongle") or "").strip()
        if not seg or not url or not dongle:
          continue
        rows.append(ManifestRow(
          dongle_id=dongle,
          route_id=seg,         # carry the full segment id as the identifier
          branch=(raw.get("platform") or "").strip(),
          rlog_url=url,
        ))
    elif {"dongle_id", "route_id"}.issubset(fields):
      for raw in reader:
        dongle = (raw.get("dongle_id") or "").strip()
        route = (raw.get("route_id") or "").strip()
        if not dongle or not route:
          continue
        rows.append(ManifestRow(
          dongle_id=dongle, route_id=route,
          branch=(raw.get("branch") or "").strip(),
        ))
    else:
      raise ValueError(
        f"manifest {path} columns {sorted(fields)} match neither "
        f"(dongle_id,route_id[,branch]) nor (segment,dongle,...,rlog_url)"
      )
  return rows


def already_processed(out_parquet: str | Path) -> set[str]:
  path = Path(out_parquet)
  if not path.exists():
    return set()
  import pyarrow.parquet as pq
  table = pq.read_table(path, columns=["route_key"])
  return set(table.column("route_key").to_pylist())
