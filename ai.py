```
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalysis:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=1).numpy()[0]

    def classify(self, text):
        scores = self.predict(text)
        if scores[0] > scores[1]:
            return 'Negative'
        else:
            return 'Positive'

if __name__ == '__main__':
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    analyzer = SentimentAnalysis(model_name)
    text = 'I love this product!'
    print(analyzer.classify(text))
```