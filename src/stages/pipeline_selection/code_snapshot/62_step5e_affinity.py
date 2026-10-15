# -*- coding: utf-8 -*-
"""62_step5e_affinity.py — Stage D+E: build the binary chemical-affinity descriptor
table (A1-A9) from ONE uniform versioned source (matminer Miedema featurizer),
write provenance, run the Ni scan. No model is fit."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5e_common as C

XS = {"ZR": "Zr", "TI": "Ti", "CO": "Co", "SI": "Si", "AG": "Ag", "MO": "Mo"}
BASE = {"CU": "Cu", "CR": "Cr", "NB": "Nb"}


def main():
    import importlib.metadata as mdv
    import warnings
    warnings.filterwarnings("ignore")
    from pymatgen.core.composition import Composition
    from matminer.featurizers.composition.alloy import Miedema

    m = Miedema()
    labels = m.feature_labels()
    assert C.AFFINITY_COLUMN in labels, C.AFFINITY_COLUMN

    rows, prov = [], []
    mm_version = mdv.version("matminer")
    for xe, sym in XS.items():
        vals = {}
        for be, bsym in BASE.items():
            comp = Composition({sym: 0.5, bsym: 0.5})
            v = dict(zip(labels, [float(t) for t in m.featurize(comp)]))
            vals[be] = v[C.AFFINITY_COLUMN]
            for l in labels:  # store all three source quantities for the record
                prov.append({"element": sym, "pair": f"{sym}-{bsym}",
                             "descriptor_name": l, "value": v[l], "unit": "eV/atom",
                             "source": f"matminer=={mm_version} "
                                       f"featurizers.composition.alloy.Miedema",
                             "method": "Miedema model with de Boer et al. 1988 parameter "
                                       "tables (shipped with matminer)",
                             "composition_definition": "equiatomic 50:50 binary",
                             "citation": "de Boer, Boom, Mattens, Miedema, Niessen, "
                                         "Cohesion in Metals, Coord. Chem. Rev. 100 (1988) 665; "
                                         "matminer implementation (Ward et al., Comput. Mater. "
                                         "Sci. 152 (2018) 60)",
                             "retrieval_timestamp": C.utcnow(),
                             "notes": "deterministic package implementation; no manual estimates"})
        hcu, hcr, hnb = vals["CU"], vals["CR"], vals["NB"]
        rows.append({
            "element_code": xe, "element_symbol": sym,
            "A1_H_XCu": hcu, "A2_H_XCr": hcr, "A3_H_XNb": hnb,
            "A4_H_XCr_minus_H_XCu": hcr - hcu,
            "A5_H_XNb_minus_H_XCu": hnb - hcu,
            "A6_H_XNb_minus_H_XCr": hnb - hcr,
            "A7_min": min(hcu, hcr, hnb), "A8_max": max(hcu, hcr, hnb),
            "A9_range": max(hcu, hcr, hnb) - min(hcu, hcr, hnb),
        })
    table = pd.DataFrame(rows)
    assert table["element_code"].tolist() == list(XS.keys())
    num = table.drop(columns=["element_code", "element_symbol"])
    assert np.isfinite(num.to_numpy(float)).all(), "non-finite affinity value"
    assert not num.columns.duplicated().any()

    tpath = os.path.join(C.D_AFF, "STEP5E_AFFINITY_DESCRIPTORS.csv")
    table.to_csv(tpath, index=False)
    ppath = os.path.join(C.D_AFF, "STEP5E_AFFINITY_DESCRIPTOR_PROVENANCE.csv")
    pd.DataFrame(prov).to_csv(ppath, index=False)

    # Ni must never appear: scan the descriptor table and provenance
    hits = C.ni_scan([("STEP5E_AFFINITY_DESCRIPTORS", table),
                      ("PROVENANCE", pd.DataFrame(prov))])
    for f in [tpath, ppath]:
        hits += [f"{os.path.basename(f)}::text::hit{i}"
                 for i, _ in enumerate(C.NI_TOKEN.finditer(
                     open(f, encoding="utf-8").read().upper()))]
    # NIOBIUM cannot match the token-bounded pattern; DOMINANT is absent here
    status = {"status": "AVAILABLE",
              "affinity_column": C.AFFINITY_COLUMN,
              "n_pairs": len(prov) // len(labels),
              "n_x_elements": len(XS),
              "nan": 0,
              "ni_scan_hits": hits,
              "descriptor_table_sha256": C.sha256_file(tpath),
              "provenance_sha256": C.sha256_file(ppath),
              "generated_utc": C.utcnow()}
    assert not hits, f"Ni token found: {hits[:5]}"
    C.write_json(os.path.join(C.D_AFF, "AFFINITY_BRANCH_STATUS.json"), status)
    print(table.to_string(index=False))
    print(f"\nSTAGE D+E: affinity descriptors AVAILABLE ({len(prov)} provenance rows, "
          f"Ni hits 0). A1-A9 built on {C.AFFINITY_COLUMN}.")


if __name__ == "__main__":
    main()
