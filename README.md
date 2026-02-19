# POSHEM SPOTIFY MUSIC MOOD CLASSIFICATION

**Supervised Learning for Spotify Music Classification Using Audio Features**

Predict song mood (Sad, Happy, Energetic, Calm) based on Spotify's audio characteristics using machine learning.

---

## 🛠️ LIBRARIES & TOOLS

**Data Processing:**
- pandas, numpy

**Visualization:**
- matplotlib, seaborn

**Machine Learning:**
- scikit-learn (RandomForest, DecisionTree, SVM, LogisticRegression)
- LightGBM
- joblib (model persistence)

**Deployment:**
- Streamlit (interactive web app)

---

## 🔧 CHALLENGES FIXED

### Challenge 1: Severe Class Imbalance
**Problem:** Dataset had unequal distribution across 4 mood categories  
**Solution:** Applied oversampling to balance all classes to equal representation  
**Result:** Prevented model bias toward majority class, improved recall across all moods

### Challenge 2: Feature Scaling Inconsistency
**Problem:** Audio features on different scales (tempo: 0-250, valence: 0-1, loudness: -60 to 5)  
**Solution:** Applied StandardScaler to normalize all features  
**Result:** 15% accuracy improvement, especially for distance-based models (SVM)

### Challenge 3: Model Selection Uncertainty
**Problem:** Unclear which algorithm works best for audio feature classification  
**Solution:** Trained 5 different models and compared with cross-validation  
**Result:** Random Forest achieved 97.6% accuracy (best performer)

### Challenge 4: Overfitting Risk
**Problem:** High accuracy on train set, needed generalization verification  
**Solution:** Implemented 5-fold stratified cross-validation  
**Result:** CV accuracy: 97.6% ± 0.06% - confirmed model generalizes well

---

## 📊 DATASET

**Source:** 278k Emotion Labeled Spotify Songs (Kaggle)  
**Records:** 277,938 songs  
**Features:** 11 audio characteristics  
**Target:** 4 mood categories (0=Sad, 1=Happy, 2=Energetic, 3=Calm)

**Audio Features:**
- duration (ms) - Song length
- danceability - Suitability for dancing (0-1)
- energy - Intensity and activity (0-1)
- loudness - Overall volume in dB
- speechiness - Presence of spoken words (0-1)
- acousticness - Acoustic probability (0-1)
- instrumentalness - Likelihood of no vocals (0-1)
- liveness - Live audience presence (0-1)
- valence - Musical positivity (0-1)
- tempo - Beats per minute
- spec_rate - Spectral audio rate

---

## 🚀 HOW TO RUN

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm streamlit joblib
```

### Run Notebook Analysis
```bash
jupyter notebook spotify_music.ipynb
```

### Run Streamlit App
```bash
streamlit run spotify_classifier_app.py
```

Make sure `trained_models.pkl` and `scaler.pkl` are in the same directory.

---

## 📈 MODEL PERFORMANCE

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | CV Mean ± Std |
|-------|----------|-----------|---------|----------|---------|---------------|
| **Random Forest** | **97.6%** | **97.6%** | **97.6%** | **97.6%** | **99.9%** | **97.6% ± 0.06%** |
| LightGBM | 96.9% | 96.9% | 96.9% | 96.9% | 97.3% | 96.8% ± 0.07% |
| Logistic Regression | 96.0% | 96.0% | 96.0% | 96.0% | 99.7% | 96.0% ± 0.06% |
| Decision Tree | 96.0% | 96.0% | 96.0% | 96.0% | 99.0% | 96.0% ± 0.06% |
| SVM | 81.8% | 81.5% | 81.8% | 81.5% | 97.0% | 81.7% ± 0.05% |

**Winner:** Random Forest with 97.6% accuracy and 99.9% ROC AUC

---

## 🔍 KEY INSIGHTS

### What Defines Each Mood?

**Sad Songs:**
- Low energy (0.2-0.4)
- Low valence (0.1-0.3)
- High acousticness (0.5-0.8)
- Slower tempo (60-90 BPM)

**Happy Songs:**
- High energy (0.6-0.9)
- High valence (0.7-0.9)
- High danceability (0.6-0.8)
- Moderate tempo (110-130 BPM)

**Energetic Songs:**
- Very high energy (0.8-1.0)
- High loudness (-5 to 0 dB)
- Fast tempo (130-180 BPM)
- Low acousticness (0.0-0.3)

**Calm Songs:**
- Low energy (0.1-0.4)
- High acousticness (0.6-0.9)
- Low loudness (-20 to -10 dB)
- Slow tempo (60-100 BPM)

---

## 🎯 REAL-WORLD APPLICATIONS

### Music Recommendation Systems
- Spotify/Apple Music: Suggest songs matching user's current mood
- Playlist curation: Auto-generate mood-based playlists
- Smart speakers: "Play me something energetic"

### Music Therapy
- Mental health apps: Recommend calming music for anxiety
- Workout apps: High-energy playlists for exercise
- Sleep apps: Calm songs for bedtime routines

### Marketing & Retail
- Store ambiance: Auto-select music matching brand mood
- Advertising: Match song mood to commercial tone
- Events: DJ assistance for mood-appropriate selections

### Music Production
- Artists: Analyze their songs' emotional impact
- Producers: Ensure intended mood is achieved
- A&R: Discover similar-mood artists for label signing

---

## 📁 PROJECT STRUCTURE

```
Spotify_Music_Classification/
├── spotify_music.ipynb              # Complete analysis notebook
├── spotify_music.py                 # Streamlit web app
├── 278k_song_labelled.csv           # Dataset
├── trained_models.pkl               # Saved models
├── scaler.pkl                       # StandardScaler
├── README.md                        # This file
└── Spotify_Classification_Report.pdf  # Detailed report
```

---

## 📊 ANALYSIS WORKFLOW

1. **Exploratory Data Analysis**
   - Feature distributions
   - Correlation analysis
   - Class balance inspection

2. **Data Preprocessing**
   - Handle missing values (none found)
   - Balance classes using oversampling
   - Feature scaling with StandardScaler
   - 80/20 train-test split

3. **Model Training**
   - Trained 5 models: Random Forest, LightGBM, Logistic Regression, Decision Tree, SVM
   - Each model learns audio feature → mood patterns

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC AUC for multi-class classification
   - 5-fold stratified cross-validation
   - Confusion matrix analysis

5. **Model Deployment**
   - Saved best model (Random Forest)
   - Built Streamlit web interface
   - Real-time predictions

---

## 💡 WHAT I LEARNED

**Technical Skills:**
- Multi-class classification with imbalanced data
- Comparing multiple ML algorithms systematically
- Cross-validation for robust evaluation
- Model persistence and deployment
- Interactive UI development with Streamlit

**Data Science Process:**
- Importance of class balancing
- Feature scaling impact on model performance
- Model selection based on multiple metrics
- Interpreting audio features for business insights

**Real-World Impact:**
- How audio features correlate with human emotion perception
- Practical applications in music tech industry
- Deployment considerations for ML models

---

## 🚀 FUTURE ENHANCEMENTS

- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- SHAP values for model explainability
- Add genre as additional feature
- Deploy as REST API
- Real-time Spotify API integration
- Mood transition predictions (sad → happy songs)
- Batch processing for playlist analysis

---

## 📄 DELIVERABLES

1. Jupyter Notebook with complete analysis
2. Streamlit web application
3. Trained models and scaler (pickled)
4. Comprehensive report (DOCX)
5. README documentation

---

## 🎓 PROJECT CONTEXT

**Program:** 30 Days Data Challenge @ Poshem Technologies Institute  
**Project Type:** Supervised Machine Learning  
**Duration:** Week 3  
**Dataset Size:** 277,938 songs  
**Best Model:** Random Forest (97.6% accuracy)

---

## 👤 AUTHOR

**Name:** [Your Name]  
**Program:** Poshem Technologies Institute - Data Analytics Track  
**Date:** February 2026

---

## 📞 CONTACT

For questions about this project:
- Review the Jupyter notebook for detailed methodology
- Check the Streamlit app for interactive predictions
- Read the comprehensive report for business insights

---

<!-- ## ⚖️ LICENSE & DATA USAGE

Dataset: 278k Emotion Labeled Spotify Songs from Kaggle  
Purpose: Educational project for machine learning practice  
Models: Available for academic and portfolio use

---

## ✅ PROJECT STATUS

**Status:** ✅ Completed  
**Model Accuracy:** 97.6%  
**Deployment:** Streamlit App Ready  
**Documentation:** Comprehensive  
**Real-World Ready:** Yes

---

**Last Updated:** February 2026  
**Version:** 1.0 -->
