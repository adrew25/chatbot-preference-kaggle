from transformers import DistilBertModel, DistilBertTokenizerFast

# Load the model
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-multilingual-cased"
)

model.save_pretrained("src/models/saved_pths/distilbert-full-model")
tokenizer.save_pretrained("src/models/saved_pths/distilbert-full-tokenizer")
