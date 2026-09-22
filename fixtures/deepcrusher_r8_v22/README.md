# Frozen DeepCrusher R8 v22 benchmark fixture

This directory contains the exact operational OpenPPL source used by SpinCore's
canonical DeepCrusher benchmark preparation.

- upstream repository: `pmartins87/DeepCrusher`
- upstream branch: `r8-v22-stable-20260914`
- upstream filename: `DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_ASCII_20260914.txt`
- upstream Git blob: `6af89462eb6e6ad79f9eca8b3d7dde8153046347`
- required SHA256: `0113badc99727a7dd47c02448d4d042b5e008534cd63fd79a461a72b24eeb68d`

The copy is vendored because the DeepCrusher repository is not accessible to
the SpinCore GitHub Actions token.  The benchmark preflight verifies the exact
filename and SHA256 before any use, so this fixture must never be edited in
place.  A future DeepCrusher opponent version requires a new fixture directory,
new hashes and a separately recorded benchmark result.
