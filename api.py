from importlib import import_module
import logging

import flask 
from flask import jsonify, request
import base64
from PIL import Image
import requests, os
from pix2tex import theMainFunction

app = flask.Flask(__name__)

def decode_image(image_string):
    ext = "jpg"
    if 'base64' in image_string:
        ext,image_string = image_string.split("base64")
        image_string = image_string[1:]
        ext = ext.split("/")[1][:-1]
    base64_bytes = base64.b64decode(image_string)
    with open(f"download_image.{ext}", "wb") as image:
        image.write(base64_bytes)

@app.route("/api", methods= ["POST"])
def end():
    decode_image(request.json["image"])
    theMainFunction()
    # exec("pix2tex.py")
    # os.system("& C:/Users/92331/AppData/Local/Programs/Python/Python38/python.exe d:/Math-Search/He2LaTeX/pix2tex.py")
    with open("D:/Math-Search/He2LaTeX/demofile2.txt", "r") as f:
        data = f.read()
    return jsonify({'LaTex': data})

if __name__ == "__main__":
    app.run()