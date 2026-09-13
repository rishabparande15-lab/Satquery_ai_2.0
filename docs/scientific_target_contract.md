# Scientific Target Contract

## Token-level coverage task

`X` is frozen CROMA `[N,225,768]`. Token `i = 15r + c` maps row-major to reference pixels `[8r:8(r+1), 8c:8(c+1)]`. `Y[n,i,k]` is the fraction of those 64 pixels assigned to class `k`, giving `[N,225,19]`. Excluded pixels are tracked separately and never reassigned.

## Scene-level SatQuery coverage task

`X` is one representation per area: physical `[N,62]`, optical/SAR/joint CROMA GAP `[N,768]`, or physical plus joint GAP `[N,830]`. `Y[n,k]` is total class-`k` pixels divided by total pixels assigned to any of the 19 classes. It has shape `[N,19]`, sums to one, and measures labelled-pixel scene coverage. Excluded pixels are omitted. This is the fresh smoke task and is distinct from the historical token-level experiment.
