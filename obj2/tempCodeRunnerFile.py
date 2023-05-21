
import cv2
import os
from os import listdir

app = Flask(__name__)

@app.route('/')
def main():
	return render_template("scrapform.html")


