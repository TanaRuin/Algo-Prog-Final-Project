import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nltk_utils import tokenize, stem
from model import NeuralNet

# Define a class for the Chat Classifier
class ChatClassifier:
    def __init__(self, intents_file='intents.json', model_file='data.pth'):
        # Initialize the class with default file names and placeholders for data
        self.intents_file = intents_file
        self.model_file = model_file
        self.intents = None
        self.tags = None
        self.model = None
        self.vectorizer = None
        self.X_train = None
        self.y_train = None
        self.batch_size = None

    def load_intents(self):
        # Load intents from a JSON file
        with open(self.intents_file, 'r') as f:
            self.intents = json.load(f)

    def process_intents(self):
        # Process intents data to prepare it for training
        corpus = []
        for intent in self.intents['intents']:
            tag = intent['tag']
            patterns = intent['patterns']
            # Extend the corpus with tokenized patterns and associated tags
            corpus.extend([(pattern, tag) for pattern in patterns])
        # Get unique tags
        self.tags = list(set(tag for _, tag in corpus))
        
        # Separate patterns and tags
        X, y = zip(*corpus)
        # Tokenize patterns
        X = [' '.join(tokenize(pattern)) for pattern in X]

        # Use CountVectorizer to convert patterns to a bag-of-words representation
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        X = self.vectorizer.fit_transform(X).toarray()
        # Index the tags for training
        y = [self.tags.index(tag) for tag in y]

        X = np.array(X)
        y = np.array(y)

        return X, y

    def train_model(self, input_size=0, hidden_size=8, output_size=0,
                    num_epochs=1000, batch_size=8, learning_rate=0.001):
        # Train the neural network model
        input_size = input_size or self.X_train.shape[1]
        output_size = output_size or len(set(self.tags))

        # Initialize the model, loss function, and optimizer
        model = NeuralNet(input_size, hidden_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Create a PyTorch DataLoader for training data
        train_dataset = ChatDataset(self.X_train, self.y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Training loop
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print loss at regular intervals
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save the trained model
        self.model = model

    def save_model(self):
        # Save the trained model's state, input size, hidden size, output size, and tags to a file
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.model.input_size,
            "hidden_size": self.model.hidden_size,
            "output_size": self.model.output_size,
            "tags": self.tags,
        }
        torch.save(data, self.model_file)
        print(f'Training complete. Model saved to {self.model_file}')

    def train_and_save_model(self):
        # Load intents, process them, and train the model
        self.load_intents()
        self.X_train, self.y_train = self.process_intents()
        self.batch_size = len(self.X_train) if self.batch_size is None else self.batch_size
        self.train_model()

    def visualize_training(self, num_epochs=2000):
        # Visualize the training process by printing the loss at regular intervals
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        train_dataset = ChatDataset(self.X_train, self.y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Define a custom dataset for PyTorch's DataLoader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = X.shape[0]
        self.x_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Usage
classifier = ChatClassifier()
classifier.train_and_save_model()

# Visualize training process
classifier.visualize_training()
