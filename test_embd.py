# import os
# os.environ['GENSIM_DATA_DIR'] = '/home/isel-har/goinfre/gensim'
# from tools.preprocessor import NLProcessor
import pandas as pd
import numpy as np
from tools.cnn import TextCNN
import torch

classes_ = torch.load("data/classes.pt", weights_only=False)
model  = torch.load("models/cnn.pkl", weights_only=False)
X_test = torch.load("data/X_test.pt", weights_only=True)

model.eval()

X_test = X_test.permute(0, 2, 1)

intents = []

with torch.no_grad():
    outputs = model(X_test)
    probs = torch.softmax(outputs, dim=1)
    confidences, indices = torch.max(probs, dim=1)

    intents = [
        "Fallback Intent" if conf <= 0.35  else classes_[idx]
        for conf, idx in zip(confidences.tolist(), indices.tolist())
    ]

print("_" * 60)

df = pd.read_csv("data/true_labels.csv")
y_true = df['intent'].values.tolist()

known_correct = 0
known_total = 0

fallback_correct = 0
fallback_total = 0

for pred, true in zip(intents, y_true):
    if true == "Fallback Intent":
        fallback_total += 1
        if pred == true:
            fallback_correct += 1
    else:
        known_total += 1
        if pred == true:
            known_correct += 1

print("Known accuracy:", known_correct / known_total)
print("Fallback accuracy:", fallback_correct / fallback_total)