import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Spotify Mood Classifier", page_icon="🎵", layout="wide")

st.title("🎵 Spotify Music Mood Classifier")
st.markdown("Predict song mood based on audio features using machine learning")

st.markdown("---")

@st.cache_resource
def load_models():
    models = joblib.load("trained_models.pkl")
    scaler = joblib.load("scaler.pkl")
    return models, scaler

try:
    models, scaler = load_models()
    st.success("✓ Models loaded successfully")
except:
    st.error("Error loading models. Make sure trained_models.pkl and scaler.pkl are in the same directory.")
    st.stop()

label_maps = {0: 'Sad 😢', 1: 'Happy 😊', 2: 'Energetic ⚡', 3: 'Calm 😌'}
reverse_map = {'Sad 😢': 0, 'Happy 😊': 1, 'Energetic ⚡': 2, 'Calm 😌': 3}

model_options = list(models.keys())
selected_model = st.sidebar.selectbox("Select Model", model_options, index=model_options.index('Random Forest'))

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.markdown(f"""
**Random Forest:** 97.6% accuracy  
**LightGBM:** 96.9% accuracy  
**Logistic Regression:** 96.0% accuracy  
**Decision Tree:** 96.0% accuracy  
**SVM:** 81.8% accuracy
""")

st.markdown("### Enter Song Audio Features")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Duration (ms)", min_value=0.0, max_value=1000000.0, value=200000.0, 
                                help="Song length in milliseconds")
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 
                              help="How suitable the track is for dancing (0-1)")
    energy = st.slider("Energy", 0.0, 1.0, 0.5, 
                       help="Perceived intensity and activity (0-1)")
    loudness = st.slider("Loudness (dB)", -60.0, 5.0, -5.0, 
                         help="Overall loudness in decibels")
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 
                            help="Presence of spoken words (0-1)")
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 
                             help="Likelihood the track is acoustic (0-1)")

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5, 
                                  help="Likelihood of no vocals (0-1)")
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 
                         help="Presence of a live audience (0-1)")
    valence = st.slider("Valence", 0.0, 1.0, 0.5, 
                        help="Musical positivity conveyed (0-1)")
    tempo = st.number_input("Tempo (BPM)", min_value=0.0, max_value=250.0, value=120.0, 
                            help="Beats per minute")
    spec_rate = st.number_input("Spectral Rate", min_value=0.0, max_value=0.00001, value=0.0000003, 
                                 format="%.10f", help="Spectral audio rate feature")

if st.button("🎯 Predict Mood", type="primary"):
    inputs = np.array([[duration, danceability, energy, loudness, speechiness, 
                        acousticness, instrumentalness, liveness, valence, tempo, spec_rate]])
    
    scaled_inputs = scaler.transform(inputs)
    
    model = models[selected_model]
    prediction = model.predict(scaled_inputs)[0]
    
    predicted_mood = label_maps[prediction]
    
    st.markdown("---")
    st.markdown("### 🎭 Prediction Result")
    
    col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
    with col_result2:
        st.markdown(f"<h1 style='text-align: center; color: #1DB954;'>{predicted_mood}</h1>", 
                    unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Model: {selected_model}</p>", 
                    unsafe_allow_html=True)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(scaled_inputs)[0]
        st.markdown("### Confidence Scores")
        
        confidence_df = pd.DataFrame({
            'Mood': [label_maps[i] for i in range(4)],
            'Confidence': [f"{p*100:.1f}%" for p in proba]
        })
        st.dataframe(confidence_df, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("### 💡 Feature Examples")

with st.expander("Click to see typical values for each mood"):
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("**😢 Sad Songs:**")
        st.markdown("• Low Energy (0.2-0.4)")
        st.markdown("• Low Valence (0.1-0.3)")
        st.markdown("• High Acousticness (0.5-0.8)")
        st.markdown("• Slow Tempo (60-90 BPM)")
        
        st.markdown("**😊 Happy Songs:**")
        st.markdown("• High Energy (0.6-0.9)")
        st.markdown("• High Valence (0.7-0.9)")
        st.markdown("• High Danceability (0.6-0.8)")
        st.markdown("• Moderate Tempo (110-130 BPM)")
    
    with example_col2:
        st.markdown("**⚡ Energetic Songs:**")
        st.markdown("• Very High Energy (0.8-1.0)")
        st.markdown("• High Loudness (-5 to 0 dB)")
        st.markdown("• High Tempo (130-180 BPM)")
        st.markdown("• Low Acousticness (0.0-0.3)")
        
        st.markdown("**😌 Calm Songs:**")
        st.markdown("• Low Energy (0.1-0.4)")
        st.markdown("• High Acousticness (0.6-0.9)")
        st.markdown("• Low Loudness (-20 to -10 dB)")
        st.markdown("• Slow Tempo (60-100 BPM)")

st.markdown("---")
st.markdown("### 📊 About This Model")
st.markdown("""
This classifier was trained on 277,938 Spotify songs with labeled moods. The model learns patterns 
in audio features to predict whether a song is Sad, Happy, Energetic, or Calm.

**Best Model:** Random Forest achieved 97.6% accuracy with 5-fold cross-validation.
""")

st.markdown("---")
st.caption("Built with Streamlit • Trained on Spotify Audio Features")