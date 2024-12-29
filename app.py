from flask import Flask, render_template, request, redirect, url_for, flash
import os
import base64
from PIL import Image
import io
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Create directories
image_dir = "student_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Database to store face embeddings (in a real app, use a proper database)
face_database = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            if not name:
                flash('Please enter a name', 'error')
                return redirect(url_for('enroll'))
            
            # Get the captured image data
            image_data = request.form.get('image_data')
            if not image_data:
                flash('No image captured', 'error')
                return redirect(url_for('enroll'))
            
            # Convert base64 image to file
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save image
            image_path = os.path.join(image_dir, f"{name.replace(' ', '_')}.jpg")
            image.save(image_path)
            
            # Get face embedding
            try:
                embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", detector_backend='opencv')
                face_database[name] = {
                    'embedding': embedding[0]['embedding'],
                    'image_path': image_path
                }
                flash(f'Successfully enrolled {name}!', 'success')
            except Exception as e:
                os.remove(image_path)  # Clean up the image if face detection fails
                flash('No face detected in the image. Please try again.', 'error')
                return redirect(url_for('enroll'))
                
        except Exception as e:
            flash(f'Error during enrollment: {str(e)}', 'error')
            return redirect(url_for('enroll'))
            
        return redirect(url_for('enroll'))
    
    return render_template('enroll.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/report')
def report():
    return render_template('report.html')

if __name__ == '__main__':
    app.run(debug=True)