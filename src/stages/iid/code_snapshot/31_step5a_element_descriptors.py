# -*- coding: utf-8 -*-
"""
31_step5a_element_descriptors.py
STEP5A Stage F: development-element descriptor source.

Builds ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.csv for exactly the 9 development
elements (CU CR NB ZR TI CO SI AG MO). Ni is NEVER queried.

Source policy (task book sec.29-30): version-pinned Python package databases
only; one single source per property for ALL development elements; no web
scraping.

Sources:
  1. matminer 0.10.1 MagpieData (magpie.data.json elemental repository)
     -> AtomicNumber, AtomicWeight, CovalentRadius, Electronegativity(Pauling),
        VEC(NValence), MeltingT, FirstIonizationEnergy, ElectronAffinity,
        AtomicVolume_GS (GSvolume_pa)
  2. mendeleev 1.2.0 (built-in elemental database)
     -> MetallicRadius(atomic_radius), Density, MendeleevNumber
  3. matminer 0.10.1 CohesiveEnergyData
     -> CohesiveEnergy

Outputs:
  01_feature_sources/ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.csv
  01_feature_sources/ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet
  01_feature_sources/ELEMENT_DESCRIPTOR_PROVENANCE.md
"""
import importlib.metadata as md_ver
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
from pymatgen.core.periodic_table import Element

STEP5A = r"E:\Cu_CALPHAD_ML\04_ML\STEP5A_JMST_ML_FOUNDATION_V2"
OUT = os.path.join(STEP5A, "01_feature_sources")

DEV_ELEMENTS = ["CU", "CR", "NB", "ZR", "TI", "CO", "SI", "AG", "MO"]
PMG_SYMBOL = {"CU": "Cu", "CR": "Cr", "NB": "Nb", "ZR": "Zr", "TI": "Ti",
              "CO": "Co", "SI": "Si", "AG": "Ag", "MO": "Mo"}

assert "NI" not in DEV_ELEMENTS and "Ni" not in PMG_SYMBOL.values()

MATMINER_V = md_ver.version("matminer")
MENDELEEV_V = md_ver.version("mendeleev")
PYMATGEN_V = md_ver.version("pymatgen")

# (property, source_key, package, field_name, unit, transform)
PROPERTIES = [
    ("AtomicNumber",         "magpie",   f"matminer=={MATMINER_V} MagpieData", "Number",                 "-",       "none"),
    ("AtomicWeight",         "magpie",   f"matminer=={MATMINER_V} MagpieData", "AtomicWeight",           "amu",     "none"),
    ("CovalentRadius",       "magpie",   f"matminer=={MATMINER_V} MagpieData", "CovalentRadius",         "pm",      "none"),
    ("Electronegativity",    "magpie",   f"matminer=={MATMINER_V} MagpieData", "Electronegativity",      "Pauling", "none"),
    ("VEC",                  "magpie",   f"matminer=={MATMINER_V} MagpieData", "NValence",               "count",   "none (total valence electrons s+p+d+f of the neutral atom)"),
    ("MeltingT",             "magpie",   f"matminer=={MATMINER_V} MagpieData", "MeltingT",               "K",       "none"),
    ("FirstIonizationEnergy","magpie",   f"matminer=={MATMINER_V} MagpieData", "FirstIonizationEnergy",  "eV",      "none"),
    ("ElectronAffinity",     "magpie",   f"matminer=={MATMINER_V} MagpieData", "ElectronAffinity",       "eV",      "none"),
    ("AtomicVolume_GS",      "magpie",   f"matminer=={MATMINER_V} MagpieData", "GSvolume_pa",            "A^3/atom","none (DFT ground-state volume per atom from the Magpie repository)"),
    ("MetallicRadius",       "mendeleev",f"mendeleev=={MENDELEEV_V}",          "atomic_radius",          "pm",      "none"),
    ("Density",              "mendeleev",f"mendeleev=={MENDELEEV_V}",          "density",                "g/cm^3",  "none (solid, ambient)"),
    ("MendeleevNumber",      "mendeleev",f"mendeleev=={MENDELEEV_V}",          "mendeleev_number",       "-",       "none"),
    ("CohesiveEnergy",       "cohesive", f"matminer=={MATMINER_V} CohesiveEnergyData", "cohesive energy", "eV/atom", "none"),
]


def main():
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    from matminer.utils.data import MagpieData, CohesiveEnergyData
    from mendeleev import element as mend_element

    magpie = MagpieData()
    cohesive = CohesiveEnergyData()

    rows = {e: {"element": e} for e in DEV_ELEMENTS}
    prov = []
    dropped = []
    for prop, src, pkg, field, unit, transform in PROPERTIES:
        vals = {}
        for e in DEV_ELEMENTS:
            sym = PMG_SYMBOL[e]
            if src == "magpie":
                v = magpie.get_elemental_property(Element(sym), field)
            elif src == "cohesive":
                v = cohesive.get_elemental_property(sym, field)
            else:
                v = getattr(mend_element(sym), field)
            vals[e] = v
        missing = [e for e, v in vals.items() if v is None or (isinstance(v, float) and v != v)]
        included = len(missing) == 0
        if included:
            for e, v in vals.items():
                rows[e][prop] = float(v)
        else:
            dropped.append((prop, missing))
        prov.append({
            "property": prop, "source": src, "package": pkg, "package_version":
                {"magpie": MATMINER_V, "cohesive": MATMINER_V, "mendeleev": MENDELEEV_V}[src],
            "field_name": field, "unit": unit, "missing_values": ",".join(missing) if missing else "none",
            "transformation": transform, "included": included,
            "exclusion_reason": "" if included else f"missing for {missing}",
        })

    table = pd.DataFrame([rows[e] for e in DEV_ELEMENTS])
    os.makedirs(OUT, exist_ok=True)
    table.to_csv(os.path.join(OUT, "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.csv"), index=False)
    table.to_parquet(os.path.join(OUT, "ELEMENT_DESCRIPTOR_TABLE_DEV6_V1.parquet"), index=False)

    prov_df = pd.DataFrame(prov)
    prov_df.to_csv(os.path.join(OUT, "ELEMENT_DESCRIPTOR_PROVENANCE.csv"), index=False)

    with open(os.path.join(OUT, "ELEMENT_DESCRIPTOR_PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write("# ELEMENT DESCRIPTOR PROVENANCE (Step5A Stage F)\n\n")
        f.write(f"generated_utc: {ts}\n\n")
        f.write(f"Development elements (exactly 9): {', '.join(DEV_ELEMENTS)}\n\n")
        f.write("The sealed prospective element (nickel, Z=28) was NEVER queried against any "
                "source in this script (hard-coded DEV_ELEMENTS list; assertion at module top).\n\n")
        f.write("Package versions:\n\n")
        f.write(f"- matminer {MATMINER_V} (MagpieData = magpie.data.json elemental repository; CohesiveEnergyData)\n")
        f.write(f"- mendeleev {MENDELEEV_V} (built-in versioned elemental database)\n")
        f.write(f"- pymatgen {PYMATGEN_V} (Element objects only, as lookup keys)\n\n")
        f.write("Policy: one single source per property, identical for all 9 development elements "
                "(task book sec.29). No web-scraped tables (sec.30). Properties with any missing "
                "value are dropped entirely, never hand-estimated (sec.33).\n\n")
        f.write(prov_df.to_markdown(index=False))
        f.write("\n\nDropped properties: " + (", ".join(p for p, _ in dropped) if dropped else "none") + "\n")

    # Ni token scan of this stage's outputs
    ni_hits = 0
    for fn in os.listdir(OUT):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p) and fn.endswith((".csv", ".md", ".json")):
            txt = open(p, encoding="utf-8", errors="ignore").read()
            ni_hits += len(re.findall(r"(?<![A-Za-z])(?:NI|Ni)(?![A-Za-z])", txt)) - \
                       len(re.findall(r"NIOBIUM|DOMINANT", txt, re.IGNORECASE))
    print(f"Ni token scan on 01_feature_sources outputs: {ni_hits} hits (must be 0)")
    assert ni_hits == 0, "Ni token found in Stage F outputs!"

    print(table.to_string(index=False))
    print(f"\nSTAGE F: {len(table.columns)-1} properties x {len(table)} elements written. PASS")

    with open(os.path.join(OUT, "STAGE_F_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": ts, "n_elements": len(DEV_ELEMENTS),
                   "elements": DEV_ELEMENTS, "n_properties": len(table.columns) - 1,
                   "properties": [c for c in table.columns if c != "element"],
                   "dropped_properties": dropped, "matminer": MATMINER_V,
                   "mendeleev": MENDELEEV_V, "pymatgen": PYMATGEN_V,
                   "ni_scan_hits": ni_hits}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
