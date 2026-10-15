# production_freeze

PRIMARY/SECONDARY checkpoints, scalers, feature schema, split IDs and anchor-sequence freezes before Ni reveal.

These are the actual archived stage sources. Run them through `src/reproduce.py prepare` and `run` so their historical paths resolve in a separate workspace. Paired inputs and saved outputs are indexed under `data/production_freeze/`. Data are not modified in the release. Training and thermodynamic stages can be expensive.
