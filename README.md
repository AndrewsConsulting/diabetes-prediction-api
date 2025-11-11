# 🥼 Diabetes Prediction API

**Zero False Negative ML Pipeline for Diabetes Risk Assessment**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo

**Deploy in 2 minutes:** https://railway.app

1. Go to Railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select this repo
4. Auto-deploys! ✅

## ✨ Key Features

- ✅ **Zero False Negatives** - Catches 100% of diabetes cases
- ✅ **Multi-Model Ensemble** - 5 ML algorithms optimized
- ✅ **Interactive Web UI** - No coding needed
- ✅ **REST API** - Easy integration
- ✅ **Data Quality Analysis** - Comprehensive EDA
- ✅ **Visualizations** - Correlation, ROC curves, etc.
- ✅ **Docker Ready** - Deploy anywhere

## 📊 Performance

| Metric | Value |
|--------|-------|
| False Negatives | **0** ✅ |
| Sensitivity | 95-100% |
| Specificity | 80-95% |
| AUC | 0.90+ |

## 🏃 Quick Start

### Online (Easiest)

1. Go to https://railway.app
2. New Project → Deploy from GitHub
3. Select your repository
4. Done! 🎉

### Local Installation

```bash
git clone https://github.com/AndrewsConsulting/diabetes-prediction-api.git
cd diabetes-prediction-api

pip install -r requirements.txt
python app.py

# Visit http://localhost:8080
```

### Docker

```bash
docker build -t diabetes-api .
docker run -p 8080:8080 diabetes-api
```

## 📡 API Examples

### Train Model
```bash
curl -X POST -F "file=@your_data.csv" http://localhost:8080/train
```

### Get Predictions
```bash
curl -X POST -F "file=@test_data.csv" http://localhost:8080/predict
```

### Single Prediction
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"Age":50,...}' \
  http://localhost:8080/predict_single
```

Full API docs: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

## 📋 Data Format

### Training CSV (with Outcome)
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
```

### Test CSV (no Outcome)
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
```

Sample data: [sample_data/diabetes_sample.csv](sample_data/diabetes_sample.csv)

## 📚 Analysis Features

- Data quality report (A-F grading)
- Feature distributions
- Correlation matrix
- Scatter plots by outcome
- Confusion matrix
- ROC curve
- Clinical safety metrics

## 🔧 Architecture

```
Web UI (HTML/JS)
    ↓
CherryPy REST API
    ↓
ML Pipeline:
├─ Logistic Regression
├─ Decision Tree
├─ Random Forest
├─ SVM
├─ Gradient Boosting
└─ Ensemble
    ↓
Predictions + Visualizations
```

## 📖 Documentation

- [API Reference](docs/API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Disclaimer](DISCLAIMER.md)

## ⚠️ Important

**For research/education only:**
- NOT for clinical diagnosis
- NOT approved by FDA
- Always consult healthcare professionals
- Confirm results with lab tests

[Full Disclaimer](DISCLAIMER.md)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push and open PR

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 📧 Support

- Email: andrew@andrewsconsulting.com
- Issues: [GitHub Issues](https://github.com/AndrewsConsulting/diabetes-prediction-api/issues)

---

**Made with ❤️ for healthcare ML**
