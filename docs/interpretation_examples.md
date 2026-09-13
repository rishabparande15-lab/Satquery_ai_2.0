# Pass 5E Interpretation Examples

These examples document adapter behavior; the real outputs are generated in `artifacts/interpretation/61_39`.

| Question | Route | Expected behavior |
|---|---|---|
| What is present in this image? | Scene claims | Emit accepted vegetation, water, and SAR evidence; state that sensor views are complementary, not joint prediction |
| Is there vegetation? | Vegetation | Emit only the validated NDVI-supported optical claim |
| Where is the vegetation? | Vegetation + geometry | Add deterministic region count and geometry-derived scene position |
| Is there water? | Water | Emit only the validated NDWI-supported optical claim |
| Where is the water? | Water + geometry | Add location only when referenced region geometry exists |
| What does the SAR image show? | SAR | Emit only VV-minus-VH-supported surface-variation language |
| Is the scene homogeneous? | Heterogeneity | Return unavailable unless the exact validated heterogeneity claim exists |
| What evidence supports that? | Evidence details | Emit accepted claims; technical mode adds feature/token/region references |
| How many houses are present? | Rejected | Return the fixed unavailable response with zero claims |
| What exact road is this? | Rejected | Return the fixed unavailable response with zero claims |

For sample `61_39`, the generated simple answer reports moderate vegetation-related optical evidence, moderate water-related optical evidence, and moderate SAR polarization-related variation. It does not say forest, river, building, road, object count, exact identity, calibrated confidence, or joint semantic prediction.
