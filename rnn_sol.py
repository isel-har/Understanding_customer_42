from torch.utils.data import DataLoader, TensorDataset
from tools.earlystopping import EarlyStopping
from tools.train import train
# from transformers import BertTokenizer
# from tools.bert import IntentClassifier
from sklearn.metrics import accuracy_score
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from tools.cnn import TextCNN
from tools.preprocessor import NLProcessor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


X_train, y_train = torch.load('data/X_train.pt', weights_only=True), torch.load('data/y_train.pt', weights_only=True)
X_val, y_val = torch.load('data/X_val.pt', weights_only=True), torch.load('data/y_val.pt', weights_only=True)
classes_ = torch.load('data/classes.pt', weights_only=False)


HP_BATCH_SIZE = 32
dropout_rate = 0.2
patience = 5
lr = 0.001
weight_decay = 1e-4
last_val_acc = 0.0
scheduler_factor = 0.5
num_epochs = 100
kernel_sizes = [3, 5, 7]


X_train = X_train.permute(0, 2, 1)
X_val   = X_val.permute(0, 2, 1)


_, embed_dim, seq_len = X_train.shape 

dataloader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=HP_BATCH_SIZE
)

model = TextCNN(
    embed_dim=embed_dim,
    out_channels=128,
    output_size=len(classes_),
    kernel_sizes=kernel_sizes,
    dropout_rate=dropout_rate,
)
 
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=scheduler_factor, patience=patience
)

earlystopping = EarlyStopping(patience=patience)


for epoch in range(num_epochs):
    model.train()
    avg_train_loss, train_acc = train(model, criterion, optimizer, dataloader)

    print(f"epoch {epoch + 1}/{num_epochs} | train loss : {avg_train_loss:.4f} | train acc : {train_acc:.3f}")

    model.eval()
    val_loss, val_acc = EarlyStopping.compute_val_metrics(model, criterion, X_val, y_val)
    scheduler.step(val_loss)
    print(f"val loss : {val_loss:.4f} | val acc : {val_acc:.3f}")

    if earlystopping(val_loss, model):
        last_val_acc = val_acc
        print(f"Early stopping triggered. Best Val Acc: {last_val_acc:.3f}")
        break

print("_" * 60)


df_test = pd.read_csv("data/data_test.csv")
processor = NLProcessor(use_stopwords=True)
X_test = torch.from_numpy(processor.transform(df_test['text']).astype(np.float32))
X_test = X_test.permute(0, 2, 1)

intents = []
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    probs = torch.softmax(outputs, dim=1)
    confidences, indices = torch.max(probs, dim=1)

    intents = [
        "Fallback Intent" if conf <= 0.35  else classes_[idx]
        for conf, idx in zip(confidences.tolist(), indices.tolist())
    ]

intent_df = pd.DataFrame({'text': df_test['text'], 'intent': intents})
intent_df.to_csv("intents.csv")

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes_)}


y_test = df_test['intent'].map(class_to_idx).fillna(-1).astype(int).values
y_pred = intent_df['intent'].map(class_to_idx).fillna(-1).astype(int).values

print("CNN accuracy score in test set:", accuracy_score(y_true=y_test, y_pred=y_pred))


# torch.save(model, "models/cnn.pkl")
# print("Text CNN model saved as models/cnn.pkl")

# with open("diaries.txt", 'a') as f:

#     f.write(f"""
#     "Model Architecture:"
#         {model}
#     {("=" * 60)}
#     early stopping patience : {patience}
#     learning rate : {lr}
#     epochs : {num_epochs}
#     regularization L2 : {weight_decay}
#     lr scheduler used
#     dropout rate : {dropout_rate}
#     batch size : {batch_size}
#     validation accuracy reached : {last_val_acc:.2f}
#     scheduler factor : {scheduler_factor}
#     """
#     )
#     f.write("_" * 100)
