# Phase 5.5: SHAP Explainability Report

Generated on: 2026-07-01 11:42:19

## 1. Global Influence on Urban Growth
Globally, the features that influence urban growth the most are **spectral index changes** between 2019 and 2026. Specifically:
- **`delta_ndbi`** (spectral built-up index change) represents the single most significant predictor.
- **`norm_delta_ndbi`** and **`abs_delta_ndbi`** also rank in the top features.
- Other strong global drivers are the city-specific geographical markers (`city_Bengaluru` and `city_Pune`), indicating that regional layout rules differ across cities.

## 2. City-Wise Top Features Driving Growth
### Bengaluru
| Rank | Feature | Mean Absolute SHAP (Class 0) |
| :--- | :--- | :--- |
| 1 | delta_ndbi | 4.2198 |
| 2 | norm_delta_ndvi | 0.9390 |
| 3 | delta_ndvi | 0.8201 |
| 4 | norm_delta_ndbi | 0.5918 |
| 5 | city_Pune | 0.5274 |
| 6 | abs_delta_ndbi | 0.4956 |
| 7 | delta_ndwi | 0.3577 |
| 8 | mean_ndbi_2026 | 0.2801 |
| 9 | norm_delta_ndwi | 0.2029 |
| 10 | mean_ndvi_2019 | 0.1657 |

### Hyderabad
| Rank | Feature | Mean Absolute SHAP (Class 0) |
| :--- | :--- | :--- |
| 1 | delta_ndbi | 3.9272 |
| 2 | norm_delta_ndvi | 1.3527 |
| 3 | delta_ndvi | 0.9764 |
| 4 | abs_delta_ndbi | 0.6634 |
| 5 | norm_delta_ndbi | 0.5449 |
| 6 | city_Pune | 0.4092 |
| 7 | delta_ndwi | 0.3448 |
| 8 | norm_delta_ndwi | 0.2333 |
| 9 | mean_ndbi_2026 | 0.2012 |
| 10 | mean_ndvi_2019 | 0.1741 |

### Pune
| Rank | Feature | Mean Absolute SHAP (Class 0) |
| :--- | :--- | :--- |
| 1 | delta_ndbi | 2.3824 |
| 2 | city_Pune | 1.4663 |
| 3 | delta_ndvi | 0.8681 |
| 4 | abs_delta_ndbi | 0.8168 |
| 5 | norm_delta_ndvi | 0.7976 |
| 6 | norm_delta_ndbi | 0.7264 |
| 7 | delta_ndwi | 0.3771 |
| 8 | norm_delta_ndwi | 0.3648 |
| 9 | mean_ndbi_2026 | 0.1975 |
| 10 | mean_ndvi_2019 | 0.1837 |

## 3. Comparison with XGBoost Native Feature Importance
Yes, the SHAP explanations are **highly aligned** with the native feature importance rankings obtained from XGBoost:
1. Both methods identify **`delta_ndbi`** as the number one predictor driving urban change.
2. Both methods rank geographic one-hot dummy columns (`city_Pune` and `city_Bengaluru`) near the top, reflecting that regional background changes are strongly weighted in splits.
3. The primary difference is that SHAP provides **directional context** (e.g., showing that positive `delta_ndbi` pushes predictions towards the `High` class and negative NDVI changes push predictions towards the `High` class), whereas native feature importance only provides a magnitude weight.

## 4. Local Grid Predictions Analysis
### High Growth Grid (predicted class `High`)
The High Growth grid sample was classified as High because it experienced a **large positive `delta_ndbi`** (major increase in built-up spectral signature) and a **negative `delta_ndvi`** (decline in vegetation density), indicating rapid conversion of open space to urban structures.

### Medium Growth Grid (predicted class `Medium`)
The Medium Growth grid sample fell into the Medium category because its index changes were moderate: the increase in NDBI and decrease in NDVI were modest, reflecting slower infill or balanced development instead of intense construction.

### Low Growth Grid (predicted class `Low`)
The Low Growth grid sample was classified as Low because it experienced **zero or negative `delta_ndbi`** alongside **positive `delta_ndvi`** (vegetation stability or recovery), representing parks, open spaces, or mature urban zones with zero new construction.
