import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path

def safe(x, fallback):
    return fallback if x in (None, "", "null") else x

def normalize_license(lic):
    # Keep it simple & consistent for grouping
    if lic is None:
        return "unknown"
    return str(lic).strip().lower() or "unknown"

def pad(s, width):
    s = str(s)
    if len(s) >= width: 
        return s
    return s + " " * (width - len(s))

def make_table(rows, headers):
    # rows: list of lists of strings
    # headers: list of strings
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    line = " | ".join(pad(h, col_widths[i]) for i, h in enumerate(headers))
    sep  = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append(" | ".join(pad(str(cell), col_widths[i]) for i, cell in enumerate(r)))
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Country distributions of licensing practices (counts + proportions).")
    ap.add_argument("input_json", help="Path to hf_modelcard.licensed_v2.country_hq.json (or v2.json if it already has country_hq).")
    ap.add_argument("--min-country-total", type=int, default=1,
                    help="Only show countries with at least this many models (default: 1).")
    ap.add_argument("--top-licenses", type=int, default=0,
                    help="If >0, only print the top N licenses globally (others bucketed into 'other').")
    args = ap.parse_args()

    p = Path(args.input_json)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build country -> license -> count
    country_license_counts = defaultdict(Counter)
    global_license_counts = Counter()

    for rec in data:
        country = safe(rec.get("country_hq"), "Unknown")
        lic_raw = rec.get("license")
        lic = normalize_license(lic_raw)
        country_license_counts[country][lic] += 1
        global_license_counts[lic] += 1

    total_models = sum(global_license_counts.values())

    # Optionally reduce to top N licenses
    if args.top_licenses > 0:
        top = {lic for lic, _ in global_license_counts.most_common(args.top_licenses)}
        def collapse(counter):
            new = Counter()
            other_sum = 0
            for k, v in counter.items():
                if k in top:
                    new[k] = v
                else:
                    other_sum += v
            if other_sum:
                new["other"] = other_sum
            return new
        global_license_counts = collapse(global_license_counts)
        for c in list(country_license_counts.keys()):
            country_license_counts[c] = collapse(country_license_counts[c])

    # Determine all license columns after optional collapse
    all_licenses = sorted(global_license_counts.keys())

    # Filter countries by min total
    def country_total(c):
        return sum(country_license_counts[c].values())

    countries = [c for c in country_license_counts.keys() if country_total(c) >= args.min_country_total]
    countries.sort(key=lambda c: (-country_total(c), c))

    # ---- 1) Counts table
    count_rows = []
    for c in countries:
        row = [c, country_total(c)]
        for lic in all_licenses:
            row.append(country_license_counts[c].get(lic, 0))
        count_rows.append(row)

    headers_counts = ["Country", "Total"] + [lic for lic in all_licenses]
    print("\n=== Country × License: COUNTS ===")
    if count_rows:
        print(make_table(count_rows, headers_counts))
    else:
        print("(no rows after filtering)")

    # ---- 2) Proportions table (row-normalized)
    prop_rows = []
    for c in countries:
        tot = country_total(c)
        row = [c, tot]
        for lic in all_licenses:
            val = country_license_counts[c].get(lic, 0)
            prop = (val / tot) if tot > 0 else 0.0
            row.append(f"{prop:.3f}")
        prop_rows.append(row)

    headers_props = ["Country", "Total"] + [lic for lic in all_licenses]
    print("\n=== Country × License: PROPORTIONS (row-normalized) ===")
    if prop_rows:
        print(make_table(prop_rows, headers_props))
    else:
        print("(no rows after filtering)")

    # ---- 3) Global totals & proportions
    glob_rows = []
    for lic in sorted(global_license_counts.keys(), key=lambda k: (-global_license_counts[k], k)):
        cnt = global_license_counts[lic]
        prop = cnt / total_models if total_models else 0.0
        glob_rows.append([lic, cnt, f"{prop:.3f}"])
    print("\n=== Global License Distribution ===")
    print(make_table(glob_rows, ["License", "Count", "Proportion"]))

    print(f"\nProcessed {total_models} models from: {p}")

if __name__ == "__main__":
    main()
