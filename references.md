# References

Sources to cite on the References & Code slide.

## Dataset

- Zillow Prize: Zillow's Home Value Prediction (Zestimate), Kaggle competition, 2017.
  https://www.kaggle.com/competitions/zillow-prize-1/overview

## Code

- Project repository (data prep, models, figures):
  https://github.com/lambo-n/zillow-property-value-ml

## GPU acceleration

- NVIDIA cuML `cuml.accel`: runs unchanged scikit-learn code on the GPU. Used here for
  LinearRegression, KNeighborsRegressor, RandomForestRegressor, StandardScaler and
  OneHotEncoder (cuml-cu13 26.10). MLPRegressor isn't supported and runs on the CPU.
  - Overview: https://docs.nvidia.com/cuml/latest/cuml-accel/
  - Supported estimators and limitations: https://docs.nvidia.com/cuml/latest/cuml-accel/compatibility/
  - Note for the slides: cuML's Random Forest uses a different algorithm from
    scikit-learn's, so results won't match scikit-learn numerically.
