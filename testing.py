import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords

# Load the saved model
model = load_model(f"./c1_lstm_model_acc_0.822.keras")  
# Function to clean and preprocess the input text
def remove(text):
    pattern = re.compile(r'https?://\S+|www\.\S+|<[^\>]+>|\n')
    return pattern.sub('', text)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove links and html tags
    text = remove(text)
    # Remove non-alphabetical characters and numbers
    text = re.sub('[^a-zA-Z]', " ", text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    # Remove extra spaces
    text = re.sub(r'\s+', " ", text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Initialize the tokenizer (same one used during training)
word_tokenizer = Tokenizer()
# Fit tokenizer on training data if not previously done
# word_tokenizer.fit_on_texts(training_text_data)  # Assume you have this data

# Load the tokenizer's word index from a previously saved file if needed
# word_tokenizer.word_index = loaded_word_index  # Adjust according to how you've saved the word index

# Example of user input and prediction
def predict_sentiment(input_text):
    # Preprocess input text
    processed_text = preprocess_text(input_text)
    # Convert to sequence
    seq = word_tokenizer.texts_to_sequences([processed_text])
    # Pad the sequence to match model input length
    padded_seq = pad_sequences(seq, padding='post', maxlen=100)
    
    # Predict using the LSTM model
    prediction = model.predict(padded_seq)
    
    # Convert prediction to sentiment (0: "bad", 1: "good", 2: "neutral")
    sentiment = np.argmax(prediction)
    
    # Map the prediction to the corresponding sentiment label
    sentiment_labels = {0: "bad", 1: "good", 2: "neutral"}
    return sentiment_labels[sentiment]

# Console interaction for input
if __name__ == "__main__":
    print("Sentiment Analysis (Enter 'quit' to exit)")
    
    while True:
        user_input = input("Enter a tweet: ")
        if user_input.lower() == 'quit':
            break
        
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment: {sentiment}")
