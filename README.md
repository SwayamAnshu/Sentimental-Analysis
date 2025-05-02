# Sentiment Analysis using Deep Learning (SNN, CNN, LSTM with GloVe Embeddings)

1. This project implements a sentiment analysis system on a labeled dataset of tweets. The models used include:
- Simple Neural Network (SNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

We utilize pre-trained GloVe word embeddings for word representation and analyze the model performance using metrics like accuracy, loss curves, confusion matrix, and classification reports.

---

## 📁 Project Structure

.
├── data/
│ └── file.csv # Labeled tweet dataset
├── embeddings/
│ └── glove.6B.100d.txt # Pretrained GloVe vectors
├── embedding_matrix.csv # Extracted embedding matrix for model training
├── sentiment_analysis.py # Main training and evaluation script
├── requirements.txt # Required Python packages
└── README.md # Project documentation


---

## 🧠 Model Architectures

- **SNN (Simple Neural Network)**: Embedding → Flatten → Dense → Dropout → Output
- **CNN**: Embedding → Conv1D → GlobalMaxPooling1D → Dense → Dropout → Output
- **LSTM**: Embedding → LSTM → Dense → Dropout → Output

---

## 📊 Dataset

- The dataset `file.csv` contains two columns:
  - `tweets`: Text data.
  - `labels`: One of `good`, `neutral`, or `bad`.

Label encoding:
- `good` → 1
- `neutral` → 2
- `bad` → 0

---

## 🔧 Preprocessing Steps

- Lowercasing text
- Removing HTML tags, URLs, punctuation, and stopwords
- Tokenization and padding to fixed length (100)
- Word embeddings using 100-dimensional GloVe vectors

---

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/yourusername/sentiment-analysis-glove.git
cd sentiment-analysis-glove

2. Install dependencies
   pip install -r requirements.txt

3. Add Data and Embeddings
   -> Place your file.csv under the data/ directory.
   -> Download GloVe embeddings and place glove.6B.100d.txt under embeddings/.
4. Run the script
   python sentiment_analysis.py
   
📈 Results
Each model's performance is evaluated using:

Accuracy and Loss curves (training vs validation)

Confusion Matrix

Precision, Recall, and F1-score (classification report)

📌 Notes
GloVe vectors are loaded from glove.6B.100d.txt

The model saves the best weights using ModelCheckpoint during training

📜 License
This project is licensed under the MIT License.

🙌 Acknowledgements
GloVe Embeddings

TensorFlow

Keras
