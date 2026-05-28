# Chicago crashes experiments

Simple scripts to fetch data, train models, and compare MAE/RMSE for daily crash counts per community area.

## Layout
- data/raw: raw CSV from the city API
- data/processed: prepared panel + community area boundaries
- scripts: training and data preparation
- outputs: metrics and model artifacts
- notebooks: results visualization

## Quick start
1. Fetch data
   - python scripts/fetch_data.py --start 2023-01-01 --end 2023-12-31
2. Train models
   - python scripts/train_all.py
   - python scripts/train_xgboost.py --cv-splits 4 --test-area-frac 0.2
   - python scripts/train_georegression.py --models strf,stst,gwr --cv-splits 4 --test-area-frac 0.2
   - python scripts/train_kriging.py --cv-splits 4 --test-area-frac 0.2
   - python scripts/train_graph_model.py --cv-splits 4 --test-area-frac 0.2
3. View results
   - Open notebooks/results.ipynb

## Notes
- GeoRegression models require `pip install georegression`.
- Kriging baseline requires `pip install pykrige`.
- Graph baseline requires `pip install torch torch-geometric torch-geometric-temporal`.
- Spatial joins require geopandas with a spatial index backend (rtree or pygeos).
