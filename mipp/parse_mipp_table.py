#!/usr/bin/env python
"""Parse Table I of Paley et al., Phys. Rev. D 90, 032001 (2014) into CSV.

MIPP measurement of pi+/pi- yields per POT off the NuMI *low-energy* target
for 120 GeV/c protons, in bins of (pz, pT).

Usage:
    python parse_mipp_table.py <paper.pdf> <out.csv>
"""
import os
import csv
import re
import sys

import fitz

ROW_RE = re.compile(r"\[(\d+\.\d+),(\d+\.\d+)\)")


def parse(pdf_path):
    doc = fitz.open(pdf_path)
    tokens = []
    for page in range(11, doc.page_count):  # tables live on pages 12-14
        tokens.extend(doc[page].get_text().split())

    rows = []
    i = 0
    while i < len(tokens) - 1:
        m1 = ROW_RE.fullmatch(tokens[i])
        m2 = ROW_RE.fullmatch(tokens[i + 1]) if m1 else None
        if not (m1 and m2):
            i += 1
            continue
        pz_lo, pz_hi = float(m1.group(1)), float(m1.group(2))
        pt_lo, pt_hi = float(m2.group(1)), float(m2.group(2))
        vals = []
        j = i + 2
        while j < len(tokens) and not ROW_RE.fullmatch(tokens[j]):
            try:
                vals.append(float(tokens[j]))
            except ValueError:
                break  # hit page furniture / caption text
            j += 1
        if len(vals) >= 9:
            v = vals[:9]
        elif len(vals) >= 3:
            # bins where only the pi+ yield is reported
            v = vals[:3] + [None] * 6
        else:
            i = j
            continue
        rows.append(dict(
            pz_lo=pz_lo, pz_hi=pz_hi, pt_lo=pt_lo, pt_hi=pt_hi,
            n_pip=v[0], d_pip_stat=v[1], d_pip_syst=v[2],
            n_pim=v[3], d_pim_stat=v[4], d_pim_syst=v[5],
            ratio=v[6], d_ratio_stat=v[7], d_ratio_syst=v[8],
        ))
        i = j
    return rows


def main():
    pdf, out = sys.argv[1], sys.argv[2]
    rows = parse(pdf)
    fields = ["pz_lo", "pz_hi", "pt_lo", "pt_hi",
              "n_pip", "d_pip_stat", "d_pip_syst",
              "n_pim", "d_pim_stat", "d_pim_syst",
              "ratio", "d_ratio_stat", "d_ratio_syst"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    npip = sum(1 for r in rows if r["n_pip"] is not None)
    npim = sum(1 for r in rows if r["n_pim"] is not None)
    print(f"parsed {len(rows)} rows -> {out}")
    print(f"  pi+ measurements: {npip}   (paper quotes 124)")
    print(f"  pi- measurements: {npim}   (paper quotes 119)")
    print(f"  sum N(pi+)/POT over measured bins = "
          f"{sum(r['n_pip'] for r in rows if r['n_pip']):.4f}")
    print(f"  sum N(pi-)/POT over measured bins = "
          f"{sum(r['n_pim'] for r in rows if r['n_pim']):.4f}")
    pz = sorted({(r['pz_lo'], r['pz_hi']) for r in rows})
    print("  pz bins:", pz)


if __name__ == "__main__":
    main()
