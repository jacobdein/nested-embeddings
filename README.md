# nested-embeddings

Embeddings, embedded in their landscape.

Pixel embeddings from Earth-observation foundation models (OlmoEarth, AlphaEarth) see
only the context their model gave them, from about 160 m up to about 1.3 km. Many
landscape questions need more: a forest island in grassland, a housing estate inside
woodland. This repo holds research proofs of concept that add that context *outside*
the model. Each one aggregates pixel embeddings into nested multiscale structures and
asks what that buys for search, mapping and scale analysis. The aggregation follows the
logic of the Moellering & Tobler (1972) scale-variance decomposition (see
[`scalevar`](https://github.com/jacobdein/scale-variance)).

These are experiments, not a library.

## Experiments

| # | question | status |
|---|---|---|
| [01](experiments/01-pyramid-similarity/) | A dyadic pyramid of mean embeddings: does storing it as **means** or as **differences between levels** change which places rank as "like this"? OlmoEarth v1.3, SW London. | done |

Each experiment folder is self-contained, with its own `pyproject.toml` (run with `uv`),
README, scripts and committed figures. Downloaded data goes to a gitignored `data/` cache.
