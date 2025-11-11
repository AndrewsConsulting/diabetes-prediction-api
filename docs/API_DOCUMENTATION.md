# API Documentation

## Endpoints

### GET /health
Health check
```bash
curl http://localhost:8080/health
```

### GET /status
Get model status
```bash
curl http://localhost:8080/status
```

### POST /train
Train model with CSV
```bash
curl -X POST -F "file=@diabetes_data.csv" http://localhost:8080/train
```

CSV must have columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

### POST /predict
Batch predictions
```bash
curl -X POST -F "file=@test_data.csv" http://localhost:8080/predict
```

CSV must have same columns but WITHOUT Outcome column

### POST /predict_single
Single patient prediction
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}' \
  http://localhost:8080/predict_single
```

### GET /analysis
View full analysis and visualizations
```
http://localhost:8080/analysis
```

## Response Examples

### Successful Prediction
```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.78,
  "risk_level": "High",
  "diagnosis": "Diabetes"
}
```

### Training Complete
```json
{
  "success": true,
  "results": {
    "best_model": "ENSEMBLE",
    "threshold": 0.25,
    "best_config": {
      "fn": 0,
      "tp": 89,
      "sensitivity": 1.0,
      "specificity": 0.835,
      "auc": 0.942
    }
  }
}
```

## Error Response
```json
{
  "error": "Model not trained yet",
  "success": false
}
```

## Data Format

### Required Columns
- Pregnancies (0-17)
- Glucose (0-200)
- BloodPressure (0-122)
- SkinThickness (0-99)
- Insulin (0-846)
- BMI (0-67)
- DiabetesPedigreeFunction (0-2.4)
- Age (21-81)
- Outcome (0 or 1) - For training only

### Optional Columns
- PhysicalActivityLevel (0-4)
- FamilyHistory (None/Moderate/Strong)
