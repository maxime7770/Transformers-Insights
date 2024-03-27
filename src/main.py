import torch
from src.model.model import TransformerEncoder, ClassifierHead
from tqdm import tqdm
from src.utils import balance_and_create_dataset, plot_animation
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap

num_classes = 2
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 512
sequence_length = 256
dropout = 0.4
num_epochs = 20
batch_size = 32

loss_function = torch.nn.CrossEntropyLoss()


dataset = load_dataset("imdb")
dataset = balance_and_create_dataset(dataset, 1200, 200)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=sequence_length)

def encode_examples(examples):
    # print list of tokens
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=sequence_length)

tokenized_datasets = dataset.map(encode_examples, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# print first id of input_ids for several examples
print(tokenized_datasets['train']['input_ids'][0])
print(tokenized_datasets['train']['input_ids'][1])
print(tokenized_datasets['train']['input_ids'][2])

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=True)

vocab_size = tokenizer.vocab_size

encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_sequence_length=sequence_length)
classifier = ClassifierHead(d_model, num_classes)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)



def collect_embeddings(encoder, dataloader):
    encoder.eval()  # Ensure the model is in evaluation mode to collect embeddings
    all_embeddings = []
    all_labels = []
    with torch.no_grad():  # Disable gradient computation for embedding collection
        for batch in tqdm(dataloader, desc="Collecting embeddings"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.unsqueeze(-1)
            attention_mask = attention_mask & attention_mask.transpose(1, 2)
            labels = batch['label']
            all_labels.extend(labels.tolist())

            output = encoder(input_ids, attention_mask)
            # Collect the mean embedding for each sequence in the batch
            batch_embeddings = [embedding[0, :].cpu().numpy() for embedding in output]
            all_embeddings.extend(batch_embeddings)
    encoder.train()  # Revert model back to training mode
    return np.array(all_embeddings), all_labels

dic_embeddings = dict()

def train(dataloader, encoder, classifier, optimizer, loss_function, num_epochs):
    for epoch in range(num_epochs):        
        # Collect and store embeddings before each epoch starts for visualizaiton purposes
        all_embeddings, all_labels = collect_embeddings(encoder, dataloader)
        reduced_embeddings = visualize_embeddings(all_embeddings, all_labels, epoch, show=False)
        dic_embeddings[epoch] = [reduced_embeddings, all_labels]
        
        encoder.train()
        classifier.train()
        correct_predictions = 0
        total_predictions = 0
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.unsqueeze(-1)
            attention_mask = attention_mask & attention_mask.transpose(1, 2)
            labels = batch['label']
            optimizer.zero_grad()
            output = encoder(input_ids, attention_mask)
            classification = classifier(output)
            loss = loss_function(classification, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(classification, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)
        
        epoch_accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch} Training Accuracy: {epoch_accuracy:.4f}')


def test(dataloader, encoder, classifier):
    '''Return accuracy on the test data'''
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.unsqueeze(-1)
            attention_mask = attention_mask & attention_mask.transpose(1, 2)
            labels = batch['label']
            output = encoder(input_ids, attention_mask)
            classification = classifier(output)
            predictions = torch.argmax(classification, dim=1)
            correct += torch.sum(predictions == labels).item()
            total += len(labels)
    print('Test accuracy:', correct / total)
 

def visualize_embeddings(embeddings, labels, epoch, show=False):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    if not show:
        return embeddings_2d
    else:
        plt.figure(figsize=(10, 6))
        # scatter points with 2 different colors based on the labels
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        # save picture
        plt.savefig(f'assets/epoch_{epoch}.png')
        return embeddings_2d

def visualize_embeddings_umap(embeddings, labels, epoch, show=False):
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42)
    embeddings_2d = reducer.fit_transform(np.array(embeddings))
    if not show:
        return embeddings_2d
    else:
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        
        # Save the plot
        plt.savefig(f'assets/epoch_{epoch}_umap.png')
        return embeddings_2d


if __name__ == "__main__":

    train(train_dataloader, encoder, classifier, optimizer, loss_function, num_epochs)
    test(test_dataloader, encoder, classifier)

    plot_animation(dic_embeddings)
