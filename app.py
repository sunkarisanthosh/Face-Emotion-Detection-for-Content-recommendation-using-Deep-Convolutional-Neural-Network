from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
	found = False

	cap = cv2.VideoCapture(0)
	while not(found):
		_, frm = cap.read()
		gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

		faces = cascade.detectMultiScale(gray, 1.4, 1)

		for x,y,w,h in faces:
			found = True
			roi = gray[y:y+h, x:x+w]
			cv2.imwrite("static/face.jpg", roi)

	roi = cv2.resize(roi, (48,48))

	roi = roi/255.0
	
	roi = np.reshape(roi, (1,48,48,1))

	prediction = model.predict(roi)

	print(prediction)

	prediction = np.argmax(prediction)
	prediction = label_map[prediction]

	cap.release()
	if(prediction == "Angry" or prediction == "Disgust"):
		link  = f"https://www.youtube.com/results?search_query=content+on+neutralizing+{prediction}+mood"
	elif (prediction == "Fear"):
		link = f"https://www.youtube.com/results?search_query=Motivational+videos+to+neutralize+{prediction}"
	elif (prediction == "Happy" or prediction == "Neutral" or prediction == "Sad"):
		link = f"https://www.youtube.com/results?search_query=comedy+content"
	elif (prediction == "Surprise"):
		link = f"https://www.youtube.com/results?search_query=intresting+facts+that+{prediction}+you"
	webbrowser.open(link)

	return render_template("emotion_detect.html", data=prediction, link=link)

if __name__ == "__main__":
	app.run(debug=True)