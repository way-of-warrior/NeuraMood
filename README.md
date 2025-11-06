# 🧠 NeuraMood  
### *Real-time EEG Emotion Classification using Deep Learning*  

![NeuraMood Header](assets/banner.png)

---

## 🌟 Overview  
**NeuraMood** is a Streamlit web application that classifies human emotions from EEG (electroencephalography) data using a deep learning model.  
It provides real-time predictions, interactive visualizations, and an elegant UI for showcasing EEG-based emotion research.

The project demonstrates:
- EEG data preprocessing & normalization  
- Deep Neural Network for emotion classification  
- Interactive web visualization (confidence bars, radar charts, confusion matrix)  
- A clean, event-ready Streamlit interface with background video and animations  

---

## 🚀 Live Demo  
👉 [Streamlit App (once deployed)](https://share.streamlit.io/)  

---

## 🧩 Features  

| Feature | Description |
|----------|-------------|
| 🎛️ **Interactive Web UI** | Built using Streamlit with a modern dark theme and background video. |
| 🧠 **Deep Learning Model** | Neural Network trained on EEG features (TensorFlow / Keras). |
| 📊 **Visual Insights** | Emotion distribution, confidence charts, and confusion matrix visualization. |
| 📂 **Upload Support** | Upload CSVs for single or batch predictions. |
| ⚡ **Real-time Feedback** | Instant inference with progress spinners and loader animations. |

---

## 🧪 Tech Stack  

| Layer | Technology |
|--------|-------------|
| Frontend | Streamlit (UI + deployment) |
| Backend | Python 3.10, TensorFlow / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |

---

## 🧬 Dataset  
The EEG data was used for emotion classification and contains extracted statistical and frequency-domain features.  

Example dataset used:  
> [EEG Brainwave Dataset: Feeling Emotions (Kaggle)](https://www.kaggle.com/datasets/berkeley-biosense/eeg-brainwave-dataset-feeling-emotions)

---

## 🧠 Model Architecture  

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

- **Optimizer:** Adam  
- **Loss:** Sparse categorical crossentropy  
- **Accuracy:** ~97–98% on test data  

---

## 💻 Run Locally  

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/NeuraMood.git
cd NeuraMood
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app
```bash
streamlit run NeuraMood.py
```

### 5️⃣ (Optional) Deploy on Streamlit Cloud
- Push the repo to GitHub.
- Go to [share.streamlit.io](https://share.streamlit.io).
- Choose your repo and main file (`NeuraMood.py`).
- Click **Deploy** 🚀  

---

## 🎥 UI Enhancements  

- Background looping video (`assets/bg.mp4`)  
- Page loader animation (CSS spinner or Lottie JSON)  
- Spinners for model loading and predictions  
- Collapsed sidebar by default (`initial_sidebar_state="collapsed"`)  

---

## 🧾 References & Acknowledgments  

Model architecture and methodology were inspired by:  
> [Vidhi1290 / Deep-Learning-for-EEG-Emotion-Classification](https://github.com/Vidhi1290/Deep-Learning-for-EEG-Emotion-Classification)  

Special thanks to the open-source community for providing EEG datasets and visualization tools.

---

## 🧑‍💻 Authors  

**Your Name**  
🎓 Data Science / Machine Learning Researcher  
📧 your.email@example.com  
🌐 [LinkedIn / Portfolio link]  

---

## 📜 License  
This project is licensed under the MIT License – feel free to use and modify with attribution.  

---

### ❤️ Credits  
Developed with passion and caffeine ☕ — *for neuroscience, emotion recognition, and AI research.*
