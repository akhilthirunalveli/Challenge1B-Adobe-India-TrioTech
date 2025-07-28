from sentence_transformers import SentenceTransformer
import os

# The model we want to use offline
model_name = 'all-MiniLM-L6-v2'
# The local directory to save it to
save_path = './model_cache'

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"📥 Downloading model: {model_name}")
print(f"💾 Saving to: {os.path.abspath(save_path)}")

# Download the model from Hugging Face and save it
model = SentenceTransformer(model_name)
model.save(save_path)

print("✅ Model downloaded and saved successfully.")