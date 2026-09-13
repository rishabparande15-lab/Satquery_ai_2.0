# Pixel-to-CROMA token alignment

The established input is 120×120 and CROMA returns 225 tokens in a 15×15 layout. Therefore each token covers exactly 8 rows × 8 columns (80×80 m on the common 10 m grid). Arrays are north-up; row 0 is north and column 0 is west.

For token index `i` in row-major order:

- `token_row = i // 15`, `token_col = i % 15`
- pixel rows `[8*token_row, 8*(token_row+1))`
- pixel columns `[8*token_col, 8*(token_col+1))`
- token 0 is northwest, token 14 northeast, token 210 southwest, token 224 southeast

Aggregation reshapes `[120,120] → [15,8,15,8]`, transposes to `[15,15,8,8]`, then flattens tokens row-major. Each token stores mean, std, min, max, median, p25, p75 and valid ratio. Tests use synthetic row/column-coded blocks and verify indices 0, 1 and 15 exactly. This matches the half-open row-major convention in `src.region_grounding` and the 64-pixel target denominator in the pinned target manifest.

Result: **PASS** for geometry, orientation and deterministic aggregation. Important limitation: CROMA implementation semantics are spatial tokens on this layout, but learned receptive fields may extend beyond a nominal 8×8 block. “Same geographic anchor” is justified; “token contains information only from that block” is not.
