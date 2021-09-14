from flask import Flask, request, jsonify
from PIL import Image

def create_app(classifier):
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    
    def predict():
        img_file = request.files["img"]

        if img_file.filename == "":
            return "Bad Request",400
        
        img = Image.open(img_file)

        result = classifier.predict(img)

        return jsonify({
            "result" : result
        })
        

    return app