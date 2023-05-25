import gradio as gr
from PyShbak.Processor import General_Processor
import pickle

# Load the TfidfVectorizer model from the pickle file
filename = "tfidf_vectorizer.pkl"
with open(filename, "rb") as file:
    vectorizer = pickle.load(file)
filename = "model.pkl"
with open(filename, "rb") as file:
    model = pickle.load(file)

def greet(input_txt:str) -> int:
    out_text = General_Processor.remove_emojis(input_txt)
    out_text = General_Processor.remove_hasgtag(out_text)
    out_text = General_Processor.remove_links(out_text)
    out_text = General_Processor.remove_mentions(out_text)
    out_text = General_Processor.remove_punctation(self=0, text = out_text)
    out_text = General_Processor.remove_whitespace(out_text)
    out_text = out_text.lower()
    vectorized_new_data = vectorizer.transform([out_text])
    prediction = model.predict(vectorized_new_data)
    if prediction[0] == 0:
    	ans = "Negative"
    else:
    	ans = "Positive"
    return ans


demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()
