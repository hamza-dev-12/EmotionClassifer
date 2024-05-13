import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from emotionclass import Emotion_Sentence_Classifier

class Inference:
    def __init__(self):
        self.model_name = 'roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.attributes = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        self.model_path = "sentiment_model.pth"
        self.config = {
            'model_name': 'distilroberta-base',
            'n_labels': len(self.attributes),
            'batch_size': 128,
            'lr': 1.5e-6,
            'n_epochs': 10
        }

        self.model = Emotion_Sentence_Classifier(self.config)

    def run(self, sentence):
        x = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            _, output = self.model(torch.tensor(x['input_ids']).to('cpu'), torch.tensor(x['attention_mask']).to('cpu'), None)
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).int()

            emotion_labels = []
            for index, el in enumerate(predictions[0]):
                if el == 1:
                    emotion_labels.append(self.attributes[index])

        return emotion_labels
