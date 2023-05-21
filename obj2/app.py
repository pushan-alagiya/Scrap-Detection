from distutils.log import debug
from fileinput import filename
from flask import *
import cv2
import os
from os import listdir

app = Flask(__name__)

@app.route('/')
def main():
	return render_template("scrapform.html")



@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		name = request.form['name']
		f.save(f.filename)
		print(f.filename)
		img = cv2.imread(f.filename)
		classNames = []
		identified = []
		pricing = {
			"LAPTOP": "300",
			"MOBILE": "200",
			"CLOCK": "40",
			"MICROVAWE"	: "500",
			"CELL PHONE" : "200",
			"OVEN" : "400",
			"REFRIGERATOR" : "2000",
			"KEYBOARD" : "50",
			"TV" : "4000"
		}
		classFile = 'coco.names'
		with open(classFile,'rt') as f:
			classNames = f.read().rstrip('\n').split('\n')
			
		configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
		weightsPath = 'frozen_inference_graph.pb'
		net = cv2.dnn_DetectionModel(weightsPath,configPath)
		net.setInputSize(320,320)
		net.setInputScale(1.0/ 127.5)
		net.setInputMean((127.5, 127.5, 127.5))
		net.setInputSwapRB(True)

		classIds, confs, bbox = net.detect(img,confThreshold=0.5)

		if len(classIds) != 0:
			for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
				cv2.rectangle(img,box,color=(0,255,0),thickness=2)
				cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
				cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
				identified.append(classNames[classId-1].upper())
				
		cv2.imwrite("ack.png",img)
		print(identified)

	# Loop through the items in identified and calculate the total price and show the price of each item in front of it
	total = 0
	final = []
	no_item = len(identified)
	for item in identified:
		if item in pricing:
			final.append(pricing[item])
			total += int(pricing[item])
		else:
			continue
	print(total)
	print(final)
	print(no_item)

	return render_template("ack2.html",options=identified, total=total, final = final,name=name, no_item=no_item)

if __name__ == '__main__':
	app.run(debug=True)
