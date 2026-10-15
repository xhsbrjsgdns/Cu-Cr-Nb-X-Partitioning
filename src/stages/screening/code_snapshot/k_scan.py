# -*- coding: utf-8 -*-
"""
k_scan.py
Shared text scanners for the STEP5K integrity audit.

Two lessons from earlier stages are built in here rather than re-learned:

1. A forbidden phrase split across a line break is still the phrase. The scan
   therefore normalises all whitespace to single spaces before matching, so
   "highest-\\nstrength" is caught exactly as "highest-strength" is.

2. Some documents must quote the forbidden vocabulary in order to disclaim it.
   Those passages are wrapped in explicit DISCLAIMER markers and removed before the
   scan, so a document that says "this is not an optimal alloy" does not trip the
   guard that exists to stop it claiming one. Nothing else is exempt: the markers
   must be present in the file, not inferred.
"""
import json
import os
import re

D_OPEN = "<!-- DISCLAIMER_BLOCK_START -->"
D_CLOSE = "<!-- DISCLAIMER_BLOCK_END -->"
NI_TOKEN = re.compile(r"(?<![A-Za-z])NI(?![A-Za-z])")


def strip_disclaimers(text):
    return re.sub(re.escape(D_OPEN) + r".*?" + re.escape(D_CLOSE), " ", text,
                  flags=re.S)


def normalise(text):
    return re.sub(r"\s+", " ", text)


def _walk_json(node, path=""):
    """Yield (key_path, string_value) for every string in a parsed JSON document."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_json(v, "%s/%s" % (path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_json(v, "%s/%d" % (path, i))
    elif isinstance(node, str):
        yield path, node


def forbidden_scan(paths, phrases, declaration_keys=("forbidden",)):
    """Return [(file, phrase, context)] for every forbidden phrase that is ASSERTED.

    A phrase is not asserted when it appears inside a marked disclaimer block, or -
    in a JSON document - underneath a key that declares the vocabulary itself (the
    screening rules freeze lists the forbidden words in order to forbid them; a text
    scan reads that as five violations). JSON files are therefore parsed and walked
    by key path rather than grepped, so the exemption is exactly the declaration
    subtree and nothing else: a forbidden phrase anywhere else in the same file is
    still a finding.
    """
    hits = []
    for p in paths:
        if not os.path.exists(p):
            continue
        low = p.lower()
        if low.endswith(".json"):
            try:
                doc = json.load(open(p, encoding="utf-8"))
            except Exception:                                     # noqa: BLE001
                doc = None
            if doc is not None:
                for keypath, val in _walk_json(doc):
                    if any(d.lower() in keypath.lower() for d in declaration_keys):
                        continue
                    t = normalise(strip_disclaimers(val)).lower()
                    for ph in phrases:
                        q = normalise(ph).lower()
                        i = t.find(q)
                        if i >= 0:
                            hits.append((os.path.basename(p), ph,
                                         "%s :: %s" % (keypath,
                                                       t[max(0, i - 60):i + len(q) + 60])))
                continue
        elif not low.endswith((".md", ".txt", ".csv")):
            continue
        raw = open(p, encoding="utf-8", errors="ignore").read()
        txt = normalise(strip_disclaimers(raw)).lower()
        for ph in phrases:
            q = normalise(ph).lower()
            i = txt.find(q)
            if i >= 0:
                hits.append((os.path.basename(p), ph,
                             txt[max(0, i - 70):i + len(q) + 70]))
    return hits


def ni_scan(paths, allow_substrings=()):
    """Find the token NI in STEP5K outputs. Bookkeeping columns and sentences that
    legitimately DISCUSS the Ni blind test are allowed through by explicit substring,
    which is recorded in the audit rather than hidden in a regex."""
    hits = []
    for p in paths:
        if not os.path.exists(p) or not p.lower().endswith((".md", ".txt", ".json",
                                                            ".csv")):
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
        for m in NI_TOKEN.finditer(txt):
            ctx = normalise(txt[max(0, m.start() - 80):m.end() + 80])
            if any(a.lower() in ctx.lower() for a in allow_substrings):
                continue
            hits.append((os.path.basename(p), ctx))
    return hits


def walk_files(dirs, exts=(".md", ".txt", ".json", ".csv"), exclude_dirs=()):
    """Collect scannable files under `dirs`, recursively.

    `exclude_dirs` is honoured even when an ancestor of an excluded directory is
    also listed - passing the project root alongside an excluded subdirectory would
    otherwise silently re-include everything, which is exactly how an exclusion
    quietly stops working.
    """
    ex = tuple(os.path.abspath(d) + os.sep for d in exclude_dirs)
    out = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            ap = os.path.abspath(root) + os.sep
            if any(ap.startswith(e) or ap == e for e in ex):
                continue
            for f in files:
                if f.lower().endswith(exts):
                    out.add(os.path.abspath(os.path.join(root, f)))
    return sorted(out)
