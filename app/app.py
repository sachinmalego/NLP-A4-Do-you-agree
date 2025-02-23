from flask import Flask, render_template, request
from transformers import AutoTokenizer, BertModel
from bert import *
import torch
import torchtext

app = Flask(__name__)

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

#load your model of S_BERT
model.load_state_dict(torch.load("pickle/S_BERT.pth", map_location=device))
model.eval()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Hyperparameters for model inference
        sentence_a = request.form['prompt_a']
        sentence_b = request.form['prompt_b']
        score = calculate_similarity(model, tokenizer, sentence_a, sentence_b, device)

        # Classify the score
        label = classify_similarity_label(score)

        return render_template('home.html', sentence_a=sentence_a, sentence_b=sentence_b, output=round(score, 4), label=label)
    else:
        return render_template('home.html', sentence_a="", sentence_b="", output=None, label=None)

if __name__ == '__main__':
    app.run(debug=True)