# Deterministic slow-cache key audit

## Scope

The isolated change updates `cosmosis/runtime/pipeline.py` so that slow-subspace
cache keys include a deterministic SHA-256 digest of the slow parameter values
and the relevant pipeline context.

The context is built after Fast/Slow analysis, so automatic and explicit
boundaries use the same rule. Configuration for fast modules is excluded.
`values` and `priors` files contribute content digests, preventing reuse after
an in-place file edit. Secret-like option names are represented as
`<redacted>`.

## Verification

- Pipeline and parameter regression tests: passed.
- Explicit boundary context test: passed.
- Automatic boundary context test: passed.
- External values-file content invalidation: passed.
- Real CLASS v3.2.5 output comparison: passed with exact sampled-array equality.
- Massive-neutrino, nonlinear, and CDM+baryon output paths: passed.
- Long real-CLASS revisit sequence: passed.
- Four cosmology combinations with cache capacity three: passed; final cache size remained three.
- Fast-only parameter change: CLASS was reused while the fast likelihood changed.
- Combined relevant suite: 19 passed.
- Writable Linux core test suite with CAMB and GetDist: 54 passed when
  optional sampler tests were excluded.
- MultiNest and PolyChord built successfully in the writable Linux environment.
- Optional sampler suite after those builds: 38 passed, 4 skipped.
- Full `cosmosis/test` suite in the writable Linux environment: 106 passed,
  4 skipped.

## Measured overhead

Deterministic key generation was approximately 4.5 microseconds per key in a
10,000-key microbenchmark. A 1 MB external-file digest was approximately 0.53
milliseconds and is computed only during pipeline setup. A representative
CLASS evaluation was approximately 0.28 seconds.

## Current status

All work remains in the isolated worktree. The main checkout is unchanged and
no commit, push, or pull request has been made.

## Remaining review item

The tested Linux environment includes CAMB, GetDist, MultiNest, and PolyChord.
The remaining four skipped tests are explicitly marked skips by the project
test suite; no cache-key regression was observed.
