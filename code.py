import random
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout

# Function definitions here...

file_path = '/content/The Crow and The Pitcher story.txt'
num_questions = 5
sentences = extract_sentences(file_path)
mcq_questions = generate_mcq(sentences, num_questions)
texts = [question[0] for question in mcq_questions]
max_seq_length = 35  # Maximum sequence length for padding
padded_sequences, tokenizer = prepare_data(texts, max_seq_length)
vocab_size = len(tokenizer.word_index) + 1
model = create_model(vocab_size, max_seq_length)
model.fit(padded_sequences, np.zeros((padded_sequences.shape[0], 1)), epochs=10, verbose=1)

# Generate and print questions
generated_questions = []
for _ in range(num_questions):
    random_sentence = random.choice(sentences)
    words = random_sentence.split()
    if not words:
        continue
    blank_word = random.choice(words)
    options = [word for word in words if word != blank_word]
    options = random.sample(options, min(3, len(options)))  # Select at most 3 unique options
    options.append(blank_word)  # Add the blank word as an option
    random.shuffle(options)  # Shuffle the options
    generated_question = random_sentence.replace(blank_word, '____'), options, blank_word
    generated_questions.append(generated_question)

print("\nGenerated Questions:")
for i, (question, options, _) in enumerate(generated_questions):
    print(f"{i+1}. {question}")
    for j, option in enumerate(options):
        print(f"   {chr(ord('A')+j)}. {option}")
    print()

