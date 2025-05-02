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

🔄 Project Workflow Diagram -->

Raw Tweets (CSV)
      │
      ▼
[ Preprocessing ]
   - Clean text
   - Remove stopwords
   - Tokenize & Pad
   - Label Encoding
      │
      ▼
[ GloVe Embedding Matrix ]
      │
      ▼
[ Choose Model ]
   ┌────────────┬────────────┬────────────┐
   │    SNN     │    CNN     │   LSTM     │
   └────────────┴────────────┴────────────┘
      │
      ▼
[ Model Training & Validation ]
      │
      ▼
[ Evaluation ]
   - Accuracy & Loss Graphs
   - Confusion Matrix
   - Classification Report


🧠  Model Architecture Diagrams
1. Simple Neural Network (SNN)

  Input (100 tokens)
  
      │
      
[ Embedding Layer (GloVe) ]

      │
      
[ Flatten ]

      │
      
[ Dense (128) + ReLU ]

      │
      
[ Dropout (0.5) ]

      │
      
[ Dense (3) + Softmax ]


2. Convolutional Neural Network (CNN)

  Input (100 tokens)
  
      │
      
[ Embedding Layer (GloVe) ]

      │
      
[ Conv1D (128 filters, kernel=5) ]

      │
      
[ GlobalMaxPooling1D ]

      │
      
[ Dense (128) + ReLU ]

      │
      
[ Dropout (0.5) ]

      │
      
[ Dense (3) + Softmax ]


3. Long Short-Term Memory (LSTM)

  Input (100 tokens)
  
      │
      
[ Embedding Layer (GloVe) ]

      │
      
[ LSTM (128 units) ]

      │
      
[ Dense (128) + ReLU ]

      │
      
[ Dropout (0.5) ]

      │
      
[ Dense (3) + Softmax ]





🚀 Full Project Workflow
1. Data Collection & Loading
Source: train.csv and test.csv containing tweet text and sentiment labels.

Format:

Columns: text, label

Labels: 0 (negative), 1 (neutral), 2 (positive)

2. Data Preprocessing
Text Cleaning:

Remove HTML tags

Remove special characters and punctuation

Convert text to lowercase

Tokenization:

Use Tokenizer from Keras to convert text into integer sequences.

Padding:

Pad sequences to a maximum length (e.g., 100 tokens).

Label Encoding:

Convert categorical labels (0,1,2) into one-hot encoded vectors.

Split:

Split dataset into train and validation sets.

3. Embedding Layer
Pre-trained Embeddings:

Load GloVe (Global Vectors) embeddings (glove.6B.100d.txt).

Embedding Matrix:

Create an embedding matrix matching each word index to its GloVe vector.

Embedding Layer:

Use this matrix in a non-trainable Keras Embedding layer.

4. Model Building
Choose one or multiple architectures:

✅ Simple Neural Network (SNN)
Embedding → Flatten → Dense → Dropout → Dense (Softmax)

✅ Convolutional Neural Network (CNN)
Embedding → Conv1D → GlobalMaxPooling1D → Dense → Dropout → Dense (Softmax)

✅ Long Short-Term Memory (LSTM)
Embedding → LSTM → Dense → Dropout → Dense (Softmax)

5. Model Compilation
Loss: categorical_crossentropy

Optimizer: adam

Metrics: accuracy

6. Training
Train model using the fit() function.

Use EarlyStopping or ModelCheckpoint if needed.

7. Evaluation
Evaluate model on validation and test sets.

Metrics:

Accuracy

Precision, Recall, F1-Score (via classification_report)

Confusion Matrix

Visualizations:

Training vs Validation Accuracy/Loss

Confusion Matrix Heatmap

8. Inference
Load saved model.

Pass new tweet(s) through the preprocessing pipeline.

Predict sentiment: negative, neutral, or positive.

9. Export & Deployment (Optional)
Save model using .h5 or .pb

Create an API using Flask or FastAPI

Deploy on a server or frontend app




🧭 Summary Diagram (Workflow Outline)

               +----------------+
               |  Load Dataset  |
               +-------+--------+
                       |
               +-------v--------+
               | Preprocess Text|
               +-------+--------+
                       |
               +-------v--------+
               | Tokenize & Pad |
               +-------+--------+
                       |
               +-------v--------+
               | Encode Labels  |
               +-------+--------+
                       |
               +-------v------------------------+
               | Load GloVe & Create Embedding  |
               +-------+------------------------+
                       |
               +-------v-----------------------------+
               | Choose Model: SNN / CNN / LSTM      |
               +-------+-----------------------------+
                       |
               +-------v--------+
               |   Train Model  |
               +-------+--------+
                       |
               +-------v--------+
               |  Evaluate Model|
               +-------+--------+
                       |
               +-------v--------+
               | Make Predictions|
               +----------------+

