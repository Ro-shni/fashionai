import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

dataset_folder = 'Data'
chroma_client = chromadb.PersistentClient(path="Vector_database")
from PIL import Image
import numpy as np

class CustomImageLoader:
    def __call__(self, uris):
        images = []
        for uri in uris:
            img = Image.open(uri)
            img = img.resize((224, 224))  # Resize to 224x224 or any other smaller size
            images.append(np.array(img))
        return images

# Update the image loader in your code
image_loader = CustomImageLoader()

CLIP = OpenCLIPEmbeddingFunction()

image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function = CLIP, data_loader = image_loader)

ids = []
uris = []

for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
        file_path = os.path.join(dataset_folder, filename)
        
        ids.append(str(i))
        uris.append(file_path)

image_vdb.add(
    ids=ids,
    uris=uris
)

print("Images stored to the Vector database.")