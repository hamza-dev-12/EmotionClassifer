import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from emotionclass import Emotion_Sentence_Classifier

# Load tokenizer and model
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
attributes = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
model_path = "sentiment_model.pth"
config = {
    'model_name': 'distilroberta-base',
    'n_labels': len(attributes),
    'batch_size': 128,
    'lr': 1.5e-6,
    'n_epochs': 10
}

model = Emotion_Sentence_Classifier(config)

def inference(sentence):
  x = tokenizer(sentence)
  with torch.no_grad():
    _ , output = model(torch.tensor([x['input_ids']]).to('cpu'),torch.tensor([x['attention_mask']]).to('cpu'),None)
    prediction = torch.where(F.sigmoid(output) > 0.5,torch.tensor(1),torch.tensor(0))
    em = []
    for index , el in enumerate(prediction[0]):
      if el == 1:
        em.append(attributes[index])
    return em
  
sentence = "I am very sad today."
predicted_emotions = inference(sentence)
print("Predicted emotions:", predicted_emotions)