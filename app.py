from flask import Flask
from flask import request, render_template
import torch 
from transformers import BertTokenizer,BertForTokenClassification
import pandas as pd
import numpy as np
data = pd.read_csv("model/tags.csv")
tag_values = data["tags"].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=False)
sm = torch.nn.Softmax(dim=2)
model = BertForTokenClassification.from_pretrained("bert-base-cased",
        num_labels=len(tag_values),
        output_attentions = False,
        output_hidden_states= False)
model.cuda;
model.eval();
model.load_state_dict(torch.load('model/NER_classifier.pt'))

app = Flask(__name__)




@app.route("/",methods=['GET','POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        text = request.form['text']
        tokenized_text = tokenizer.encode(text)
        model.cuda();
        input_ids = torch.tensor([tokenized_text]).cuda()
        with torch.no_grad():
            output = model(input_ids)
        label_confidence_score = np.max(sm(output[0]).to('cpu').numpy(),axis=2)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
# join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels, scores = [], [], []
        for token, label_idx,score in zip(tokens, label_indices[0],label_confidence_score[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
                scores.append(score)
        return render_template('prediction.html',result= zip(new_tokens,new_labels,scores))




if __name__ == "__main__":
#    app.run(host='0.0.0.0',port=8080,debug=True)
    app.run(host='0.0.0.0',port=8080)
