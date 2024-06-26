# Breast Cancer Detection API

Deployed at: https://breast-cancer-detection-tu59.onrender.com
Api docs can be found at: https://breast-cancer-detection-tu59.onrender.com/docs

Example api call with cURL:

```bash
curl -H 'Content-Type: application/json' -d '{"data":{"radius_mean": 18.94, "texture_mean": 21.31, "perimeter_mean": 123.6, "area_mean": 1130.0, "smoothness_mean": 0.09009, "compactness_mean": 0.1029, "concavity_mean": 0.108, "concave points_mean": 0.07951, "symmetry_mean": 0.1582, "fractal_dimension_mean": 0.05461, "radius_se": 0.7888, "texture_se": 0.7975, "perimeter_se": 5.486, "area_se": 96.05, "smoothness_se": 0.004444, "compactness_se": 0.01652, "concavity_se": 0.02269, "concave points_se": 0.0137, "symmetry_se": 0.01386, "fractal_dimension_se": 0.001698, "radius_worst": 24.86, "texture_worst": 26.58, "perimeter_worst": 165.9, "area_worst": 1866.0, "smoothness_worst": 0.1193, "compactness_worst": 0.2336, "concavity_worst": 0.2687, "concave points_worst": 0.1789, "symmetry_worst": 0.2551, "fractal_dimension_worst": 0.06589}}' -X POST https://breast-cancer-detection-tu59.onrender.com/predict
```