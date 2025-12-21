# Deployment Guide for Agro Guard

## Files Overview

### For Deployment (Streamlit Cloud / Public)
- **Main App**: `app_deploy.py` 
- **Requirements**: `requirements_deploy.txt`
- **Purpose**: Clean deployment without Firebase dependencies

### For Local/Presentation (With Hardware)
- **Main App**: `webapp.py`
- **Requirements**: `requirements.txt` 
- **Purpose**: Full functionality with Firebase IoT integration

---

## Quick Deploy to Streamlit Cloud

### 1. Push to GitHub
```bash
git add app_deploy.py requirements_deploy.txt
git commit -m "Add deployment version"
git push
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `Sobhagyaverma/Smart-Plant-Care-System`
4. **Main file path**: `app_deploy.py`
5. **Python version**: 3.9
6. Click "Deploy"

### 3. No Secrets Needed!
- ✅ No Firebase configuration required
- ✅ No API keys needed
- ✅ Works out of the box

---

## What Works in Deployment Version

### ✅ Fully Functional
- **Disease Detection**: 100% working
  - Upload images
  - Camera input
  - AI predictions with 95.85% accuracy
  - Treatment recommendations
  - Confidence scores

### 📊 Simulation Mode
- **Smart Watering**: Demo mode
  - Simulated sensor data (moisture, temp, humidity)
  - Interactive charts
  - Pump control simulation
  - Activity logging
  - No Firebase errors or warnings

---

## Differences Between Versions

| Feature | `webapp.py` (Local) | `app_deploy.py` (Deploy) |
|---------|---------------------|--------------------------|
| Disease Detection | ✅ Full | ✅ Full |
| Firebase Integration | ✅ Yes | ❌ No |
| IoT Live Data | ✅ Real hardware | 📊 Simulated |
| Dependencies | Includes firebase-admin | Clean (no Firebase) |
| Errors/Warnings | None | None |
| Use Case | Presentation with hardware | Public deployment |

---

## Testing Locally

Before deploying, test the deployment version:

```bash
streamlit run app_deploy.py
```

You should see:
- ✅ Disease Detection working perfectly
- 📊 Smart Watering in "SIMULATION MODE" 
- ❌ No Firebase errors
- ❌ No missing module warnings

---

## Deployment Checklist

- [ ] Test `app_deploy.py` locally
- [ ] Verify model files are in repo:
  - `cnn_mobilenetv2.keras`
  - `scaler.pkl`
  - `svm_linear.pkl`
  - `labels.txt`
- [ ] Verify image files:
  - `accuracy_curve.png`
  - `loss_curve.png`
  - `confusion_matrix.png`
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Test deployed app

---

## Troubleshooting

### If models don't load:
- Check file sizes (GitHub has 100MB limit per file)
- Use Git LFS for large files if needed

### If app crashes:
- Check Streamlit Cloud logs
- Verify all dependencies in `requirements_deploy.txt`

---

## For Presentations

Use `webapp.py` with Firebase when demonstrating with real hardware:

```bash
streamlit run webapp.py
```

This version connects to your ESP32 and shows real-time IoT data!
