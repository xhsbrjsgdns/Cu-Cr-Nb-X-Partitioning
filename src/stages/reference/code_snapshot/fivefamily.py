# -*- coding: utf-8 -*-
"""fivefamily.py - Step 3B phase-family scheme: FCC / BCC / C15 / OTHER_SOLID / LIQUID.

Rule (spec section 1):
    FCC_L12*      -> FCC
    BCC_B2*       -> BCC
    C15_LAVES*    -> C15
    LIQUID*       -> LIQUID
    all other stable non-liquid phases (CR3SI, CU4TI1, future intermetallics) -> OTHER_SOLID
Raw phase instance names are never modified or dropped.
"""

import re

FAMILY_MAP_5 = {
    "FCC_L12": "FCC",
    "BCC_B2": "BCC",
    "C15_LAVES": "C15",
    "LIQUID": "LIQUID",
}
FAMILY_ORDER_5 = ["FCC", "BCC", "C15", "OTHER_SOLID", "LIQUID"]
SOLID_FAMILIES = ["FCC", "BCC", "C15", "OTHER_SOLID"]


def base_name(raw):
    return re.sub(r"#\d+$", "", str(raw))


def family_of_5(base):
    base = base_name(base)
    return FAMILY_MAP_5.get(base, "OTHER_SOLID")
