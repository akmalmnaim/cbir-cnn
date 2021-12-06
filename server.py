import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

class FeatureExtractor:
    def __init__(self):
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    def extract(self, img):
        img = img.resize((224,224)).convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/CitraBatik") / (feature_path.stem +".jpg"))
features = np.array(features)

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query_img"]

        img = Image.open(file.stream)
        uploaded_img_path = "static/uploads/" + \
            datetime.now().isoformat().replace(":",".") + "_" + file.filename
        img.save(uploaded_img_path)

        #run search
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template("index1.html", query_path=uploaded_img_path, scores =scores) 
    else:    
        return render_template("index1.html")



from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from vit import VisionTransformer
import numpy as np



if __name__ == '__main__':
    app.run()