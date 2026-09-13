# Pass 5D per-class analysis

Per-class values are scene-fraction MAE on the same 200 held-out areas. “Support” is the number of test areas with nonzero class fraction, not independent pixel count.

| Class | Support | A MAE pp | B MAE pp | Change pp |
|---|---:|---:|---:|---:|
| Marine waters | 6 | 1.1597 | 0.8230 | −0.3368 |
| Pastures | 76 | 8.6846 | 8.3502 | −0.3344 |
| Mixed forest | 79 | 12.3073 | 12.0285 | −0.2789 |
| Natural grassland/sparsely vegetated | 7 | 2.0942 | 1.9108 | −0.1835 |
| Inland waters | 21 | 1.8134 | 1.6518 | −0.1616 |
| Urban fabric | 55 | 3.0366 | 2.9259 | −0.1108 |
| Agro-forestry | 7 | 2.2937 | 2.2015 | −0.0922 |
| Transitional woodland/shrub | 61 | 5.7693 | 5.6846 | −0.0847 |
| Permanent crops | 4 | 1.9575 | 1.8922 | −0.0653 |
| Coastal wetlands | 0 | 0.6921 | 0.6891 | −0.0030 |
| Beaches/dunes/sands | 1 | 0.7011 | 0.7071 | +0.0060 |
| Moors/heathland/sclerophyllous vegetation | 4 | 1.1843 | 1.2147 | +0.0304 |
| Inland wetlands | 2 | 1.1491 | 1.2036 | +0.0545 |
| Agriculture with natural vegetation | 54 | 6.3310 | 6.4253 | +0.0943 |
| Industrial/commercial | 13 | 1.4893 | 1.5953 | +0.1060 |
| Coniferous forest | 45 | 8.0498 | 8.1759 | +0.1260 |
| Arable land | 103 | 11.5219 | 11.6518 | +0.1299 |
| Broad-leaved forest | 96 | 13.1955 | 13.3638 | +0.1683 |
| Complex cultivation patterns | 50 | 9.3762 | 9.5844 | +0.2082 |

Nine classes improved and nine worsened; coastal wetlands improved numerically despite zero positive-support areas and cannot be interpreted. The largest supported gains were pastures and mixed forest. The largest well-supported regressions were complex cultivation, broad-leaved forest and arable land. Marine waters has the largest numerical gain but only six positive areas.

Dominant-class recall changed notably for urban fabric (50%→75%, support 8), arable land (70.5%→61.4%, support 44), pastures (73.1%→76.9%, support 26), coniferous forest (90.9%→81.8%, support 11), and marine waters (66.7%→100%, support 3). Small supports make several changes unstable. Overall dominant accuracy fell from 59.5% to 58.5%.
