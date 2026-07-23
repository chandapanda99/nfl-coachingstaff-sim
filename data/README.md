# Scenario artifacts and attribution

The real-data scenario builder reads play-by-play releases maintained by the
[nflverse project](https://github.com/nflverse). nflverse documents the relevant
play-by-play data as CC-BY-4.0. Generated `benchmark-v1.jsonl` and
`quickstart-v1.jsonl` files retain that license and attribution.

The checked-in `demo-v1.jsonl` pack is synthetic, contains no nflverse records,
and is Apache-2.0. It exists only so the application and harness work offline.

## JSONL schema

Each line contains:

- `schema_version`: public scenario schema version.
- `scenario_id`: stable identifier derived from game and play identifiers.
- `state`: pre-play game state, with scores and win probability from the
  possession team's perspective.
- `ep_baseline`: expected EPA for each legal decision from the training-only
  bucketed baseline.
- `source` and `source_license`: artifact provenance.

Historical actions, EPA, WPA, descriptions, and post-play fields are deliberately
excluded from scenario payloads.

`manifest-v1.json` records training/evaluation seasons, scenario count, source,
license, and SHA-256 hashes for released real-data packs.
