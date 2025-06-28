import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

# Download NLTK resources
nltk.download('punkt')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. DATA PREPARATION
# For this example, we'll use a small dataset of news articles
# You can replace this with loading your own dataset
def load_sample_data(n_samples=200):
    """
    Generate or load sample data for demonstration
    In a real scenario, you would load your dataset here
    """
    # For demonstration purposes, let's create synthetic data
    # In practice, you would load your dataset from a file
    
    # Let's simulate loading a small news dataset
    articles = []
    summaries = []
    
    # Load CNN/DM dataset or similar
    # For now, let's create synthetic data
    for i in range(n_samples):
        article_length = np.random.randint(100, 500)
        article = f"This is a sample news article {i}. " * (article_length // len(f"This is a sample news article {i}. "))
        
        summary_length = np.random.randint(20, 50)
        summary = f"Summary of article {i}. " * (summary_length // len(f"Summary of article {i}. "))
        
        articles.append(article)
        summaries.append(summary)
        
    return articles, summaries

# Load or generate sample data
articles, summaries = load_sample_data(200)

# 2. TEXT PREPROCESSING
def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Add start and end tokens to target sequences
def add_tokens(text):
    return 'START ' + text + ' END'

# Preprocess articles and summaries
articles_processed = [preprocess_text(article) for article in articles]
summaries_processed = [add_tokens(preprocess_text(summary)) for summary in summaries]

# 3. TOKENIZATION AND SEQUENCE PREPARATION
# Set maximum lengths
max_article_len = 200  # Maximum length for articles
max_summary_len = 50   # Maximum length for summaries

# Create tokenizers
article_tokenizer = Tokenizer(oov_token="<OOV>")
article_tokenizer.fit_on_texts(articles_processed)
article_vocab_size = len(article_tokenizer.word_index) + 1

summary_tokenizer = Tokenizer(oov_token="<OOV>")
summary_tokenizer.fit_on_texts(summaries_processed)
summary_vocab_size = len(summary_tokenizer.word_index) + 1

# Convert text to sequences
article_sequences = article_tokenizer.texts_to_sequences(articles_processed)
summary_sequences = summary_tokenizer.texts_to_sequences(summaries_processed)

# Pad sequences
article_padded = pad_sequences(article_sequences, maxlen=max_article_len, padding='post')
summary_padded = pad_sequences(summary_sequences, maxlen=max_summary_len, padding='post')

# Create decoder input and target data
decoder_input_data = summary_padded[:, :-1]  # Remove the last token
decoder_target_data = summary_padded[:, 1:]  # Remove the first token (START)

# Split the data
X_train, X_test, y_train_in, y_test_in, y_train_target, y_test_target = train_test_split(
    article_padded, decoder_input_data, decoder_target_data, test_size=0.2, random_state=42
)

# 4. BUILDING THE SEQ2SEQ MODEL
def build_seq2seq_model(article_vocab_size, summary_vocab_size, 
                        max_article_len, max_summary_len,
                        embedding_dim=128, lstm_units=256):
    """
    Build an encoder-decoder LSTM model for text summarization
    """
    # Encoder
    encoder_inputs = Input(shape=(max_article_len,), name='encoder_inputs')
    encoder_embedding = Embedding(input_dim=article_vocab_size, output_dim=embedding_dim, 
                                 mask_zero=True, name='encoder_embedding')(encoder_inputs)
    
    encoder_lstm = LSTM(lstm_units, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(max_summary_len - 1,), name='decoder_inputs')
    decoder_embedding = Embedding(input_dim=summary_vocab_size, output_dim=embedding_dim, 
                                 mask_zero=True, name='decoder_embedding')(decoder_inputs)
    
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    decoder_dense = Dense(summary_vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define encoder model for inference
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Define decoder model for inference
    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model

# Build the model
model, encoder_model, decoder_model = build_seq2seq_model(
    article_vocab_size, summary_vocab_size, max_article_len, max_summary_len)

# Print model summary
model.summary()

# 5. TRAINING THE MODEL
# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_train, y_train_in], y_train_target,
    validation_data=([X_test, y_test_in], y_test_target),
    batch_size=64,
    epochs=10,
    callbacks=[early_stopping]
)

# 6. INFERENCE FUNCTION
def decode_sequence(input_seq):
    """
    Use the trained encoder and decoder models to generate a summary
    """
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1 with only the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = summary_tokenizer.word_index['start']
    
    # Output sequence
    decoded_sentence = ''
    
    # Maximum length of generated summary
    max_summary_length = 50
    
    stop_condition = False
    i = 0
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Convert the token to a word
        sampled_word = ''
        for word, index in summary_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        # Exit condition: either hit max length or find stop token
        if sampled_word == 'end' or i >= max_summary_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            
        # Update the target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
        
        i += 1
    
    return decoded_sentence

# 7. EVALUATE THE MODEL
def evaluate_model(encoder_model, decoder_model, X_test, test_articles, test_summaries, n_samples=10):
    """
    Evaluate the model by generating summaries for test articles
    and comparing them with the ground truth
    """
    generated_summaries = []
    reference_summaries = []
    
    for i in range(min(n_samples, len(X_test))):
        # Get the article and original summary
        article = test_articles[i]
        original_summary = test_summaries[i].replace('START ', '').replace(' END', '')
        
        # Generate summary using our model
        input_seq = X_test[i:i+1]
        generated_summary = decode_sequence(input_seq).strip()
        
        # Store results
        generated_summaries.append(generated_summary)
        reference_summaries.append([original_summary.split()])
        
        print(f"Article: {article[:100]}...")
        print(f"Original Summary: {original_summary}")
        print(f"Generated Summary: {generated_summary}")
        print("-" * 80)
    
    # Calculate BLEU score with smoothing to handle zero n-gram matches
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(reference_summaries, [s.split() for s in generated_summaries], smoothing_function=smoothie)
    
    # Calculate ROUGE score
    rouge = Rouge()
    # Convert reference summaries to proper format for ROUGE
    reference_summaries_text = [' '.join(r[0]) for r in reference_summaries]
    rouge_scores = rouge.get_scores(generated_summaries, reference_summaries_text, avg=True)
    
    print(f"BLEU Score: {bleu_score}")
    print(f"ROUGE Scores: {rouge_scores}")
    
    return bleu_score, rouge_scores

# 8. VISUALIZE TRAINING HISTORY
def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# Get a few test articles and summaries for evaluation
test_articles = [articles_processed[i] for i in range(len(X_test))]
test_summaries = [summaries_processed[i] for i in range(len(X_test))]

# Evaluate the model
bleu_score, rouge_scores = evaluate_model(
    encoder_model, decoder_model, X_test, test_articles, test_summaries)

# 9. SAVE THE MODEL
model.save('seq2seq_summarization_model.keras')
print("Model saved!")

# 10. CONCLUSION
print("Summary of Results:")
print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"BLEU Score: {bleu_score:.4f}")
print(f"ROUGE-1 F1 Score: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2 F1 Score: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L F1 Score: {rouge_scores['rouge-l']['f']:.4f}")