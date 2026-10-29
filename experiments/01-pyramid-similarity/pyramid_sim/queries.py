"""Pre-registered queries, match definitions, and method parameters.

Written and committed before any similarity map was computed. Exemplars were picked
from WorldCover 2021 and the Sentinel-2 2024-05-20 true-colour image only, and each
one was checked to satisfy its own match definition.

Window shares are over centred windows: s16 = 16 x 16 px (160 m), s128 = 128 x 128 px
(1.28 km). "inverse" is the definition used to score the negated contrast query
("opposite pattern").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .scoring import BUILT, GRASS, GREEN, OPEN, TREE, WATER, Shares


@dataclass(frozen=True)
class Query:
    key: str
    name: str
    row: int  # AOI pixel row / col of the exemplar
    col: int
    place: str
    definition: str
    inverse_definition: str
    match: Callable[[Shares], np.ndarray]
    inverse: Callable[[Shares], np.ndarray]


def _q1(S):
    return S.is_(TREE) & (S(TREE, 16) >= 0.5) & (S(OPEN, 128) >= 0.5)


def _q1_inv(S):
    return S.is_(OPEN) & (S(OPEN, 16) >= 0.5) & (S(TREE, 128) >= 0.5)


def _q2(S):
    return S.is_(GRASS) & (S((GRASS, TREE), 16) >= 0.5) & (S(BUILT, 128) >= 0.5)


def _q3(S):
    return S.is_(BUILT) & (S(BUILT, 16) >= 0.25) & (S(GREEN, 128) >= 0.8)


def _q4(S):
    return (
        S.is_(GRASS)
        & (S(GRASS, 16) >= 0.25)
        & (S(TREE, 16) >= 0.25)
        & (S(TREE, 128) >= 0.25)
        & (S(BUILT, 128) <= 0.1)
    )


def _q4_inv(S):
    return (
        S.is_(TREE)
        & (S(GRASS, 16) >= 0.25)
        & (S(TREE, 16) >= 0.25)
        & (S(GRASS, 128) >= 0.25)
        & (S(BUILT, 128) <= 0.1)
    )


def _q5(S):
    return S.is_(BUILT) & (S(BUILT, 16) >= 0.5) & (S(WATER, 16) == 0) & (S(WATER, 128) >= 0.05)


def _q5_inv(S):
    return S.is_(WATER) & (S(WATER, 16) >= 0.5) & (S(BUILT, 128) >= 0.3)


QUERIES = [
    Query(
        "Q1", "woodland clump in open parkland", 1144, 1432,
        "plantation (~300 x 500 m) in open grassland, Richmond Park",
        "pixel tree; s16 tree >= 50 %; s128 grassland + cropland >= 50 %",
        "pixel grass/crop; s16 grass + crop >= 50 %; s128 tree >= 50 % (clearing in woodland)",
        _q1, _q1_inv,
    ),
    Query(
        "Q2", "small green space in dense residential", 392, 1192,
        "small park (~200 x 400 m) among terraced streets, Brentford/Ealing",
        "pixel grassland; s16 grass + tree >= 50 %; s128 built-up >= 50 %",
        "= Q3 definition (built pocket in large green space)",
        _q2, _q3,
    ),
    Query(
        "Q3", "built-up pocket inside large green space", 1103, 1422,
        "White Lodge (Royal Ballet School) buildings, Richmond Park",
        "pixel built-up; s16 built-up >= 25 %; s128 tree + shrub + grass + crop + wetland >= 80 %",
        "= Q2 definition (small green space in dense residential)",
        _q3, _q2,
    ),
    Query(
        "Q4", "open grassland at a woodland edge", 1160, 1256,
        "grass beside Spankers Hill / Sidmouth woods, Richmond Park",
        "pixel grassland; s16 grass >= 25 % and tree >= 25 %; s128 tree >= 25 % and built-up <= 10 %",
        "pixel tree; s16 grass >= 25 % and tree >= 25 %; s128 grass >= 25 % and built-up <= 10 %",
        _q4, _q4_inv,
    ),
    Query(
        "Q5", "ordinary housing whose wider context is the river", 630, 1300,
        "terraced housing ~200 m from the Thames, Strand-on-the-Green (Chiswick)",
        "pixel built-up; s16 built-up >= 50 % and water = 0; s128 water >= 5 %",
        "pixel water; s16 water >= 50 %; s128 built-up >= 30 % (water inside a built area)",
        _q5, _q5_inv,
    ),
]

# --- method parameters (identical for every query) ------------------------------
LEVELS = list(range(1, 11))  # 10 m ... 5.12 km, for every method
EQUAL_W = {k: 1 / len(LEVELS) for k in LEVELS}
# Focal + context: half the weight on the pixel, half spread over 320 m ... 5.12 km.
# Levels 2-5 (20-160 m) lie inside the model's own 16 x 16 px window, so the pixel
# vector already encodes them; the context the model cannot see starts at level 6.
FOCAL_CONTEXT_W = {1: 0.5, **{k: 0.1 for k in range(6, 11)}}
PAIR_LEVELS = (5, 8)  # "two conditions on the means": 160 m and 1.28 km (the scoring scales)
BAND = (5, 8)  # contrast mean_5 - mean_8 = slot5 + slot6 + slot7
SLOT = 6  # single dyadic slot: 320 m vs 640 m, the middle of the 160 m -> 1.28 km span
EXCLUDE_LEVEL = 5  # the query's own 160 m cell is removed from every ranking
KS = (50, 200)
