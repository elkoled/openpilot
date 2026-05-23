"""Build a dongle-stratified, speed-stratified subsample of eps_seglist.csv.

Round-robin across dongles to guarantee fair coverage even if the run is
truncated by bandwidth or time. Within a dongle, prefer high-speed segments
(mean_v >= 15 m/s ~ highway band) since the user has flagged that as the
problem region.
"""
import argparse
import csv
from collections import defaultdict


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--in-csv', default='/home/batman/eps_seglist.csv')
  ap.add_argument('--out-csv', default='/home/batman/openpilot6/tools/lateral_maneuvers/fleet_residuals/stratified.csv')
  ap.add_argument('--per-dongle', type=int, default=80,
                  help='max segments to draw from each dongle')
  ap.add_argument('--highway-mean-ms', type=float, default=15.0,
                  help='preferred minimum mean_v for the first pass')
  args = ap.parse_args()

  by_dongle: dict[str, list[dict]] = defaultdict(list)
  with open(args.in_csv) as f:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames
    for row in r:
      try:
        row['_mv'] = float(row['mean_v'])
      except Exception:
        row['_mv'] = 0.0
      by_dongle[row['dongle']].append(row)

  # Within each dongle, prefer high-speed first
  for d, rows in by_dongle.items():
    rows.sort(key=lambda r: -r['_mv'])

  # Round-robin across dongles, capping at per-dongle
  out_rows: list[dict] = []
  exhausted = set()
  indices = {d: 0 for d in by_dongle}
  taken = {d: 0 for d in by_dongle}
  while len(exhausted) < len(by_dongle):
    for d in list(by_dongle.keys()):
      if d in exhausted:
        continue
      if taken[d] >= args.per_dongle:
        exhausted.add(d)
        continue
      i = indices[d]
      if i >= len(by_dongle[d]):
        exhausted.add(d)
        continue
      out_rows.append(by_dongle[d][i])
      indices[d] += 1
      taken[d] += 1

  with open(args.out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in out_rows:
      row.pop('_mv', None)
      w.writerow(row)

  print(f"wrote {len(out_rows)} rows to {args.out_csv}")
  for d, n in sorted(taken.items(), key=lambda kv: -kv[1]):
    print(f"  {d}: {n}")


if __name__ == '__main__':
  main()
