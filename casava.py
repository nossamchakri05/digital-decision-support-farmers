import os
from flask import Flask, request, render_template, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the pre-trained model
model = load_model('final_cassava_model.h5')

# Define the image size expected by the model (adjust if needed)
IMAGE_SIZE = (224, 224)  # Example size; modify based on your model's input size

# Preprocessing function to resize and normalize the image
def preprocess_image(image_path):
    img = Image.open(image_path)  # Open the image file
    img = img.resize(IMAGE_SIZE)  # Resize to match model input size
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Disease prediction route
@app.route('/diseaseout', methods=['POST'])
def diseaseout():
    if 'file' not in request.files:
        flash("No file part", "danger")
        return redirect(url_for('disease_page'))

    file = request.files['file']

    if file.filename == '':
        flash("No selected file", "danger")
        return redirect(url_for('disease_page'))

    try:
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Save the uploaded file temporarily
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Preprocess the image and make a prediction
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)

        # Example mapping, update based on your class names
        class_names = [
            'Healthy', 'Cassava Mosaic Disease', 'Cassava Bacterial Blight',
            'Cassava Brown Streak Disease', 'Cassava Green Mottle'
        ]
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]

        # Tips based on prediction
        disease_tips = {
            'Cassava Bacterial Blight': [
                "Plant disease-resistant cassava varieties that can withstand bacterial infections.",
                "Sterilize farming tools after use to prevent the transfer of bacteria between plants.",
                "Remove and destroy any infected plants to stop the bacteria from spreading further.",
                "Avoid planting cassava during heavy rainy seasons, as moisture can facilitate bacterial growth.",
                "Improve drainage in fields to prevent waterlogging, which creates conditions favorable for bacterial proliferation.",
                "Regularly monitor fields and apply copper-based bactericides if signs of infection appear."
            ],
            'Cassava Brown Streak Disease': [
                "Use virus-free planting materials to reduce the initial source of infection in the field.",
                "Apply insecticides or biological agents to control whiteflies, the primary vectors of CBSD.",
                "Immediately uproot and burn any plants showing clear CBSD symptoms to limit spread.",
                "Avoid planting cassava in areas where CBSD outbreaks have been previously reported.",
                "Introduce intercropping systems with non-host crops to reduce vector populations and virus spread.",
                "Promote soil health by applying organic compost to strengthen cassava plants against disease stress."
            ],
            'Cassava Green Mottle': [
                "Collect and safely destroy infected plant debris to eliminate sources of reinfection.",
                "Apply fungicides or bactericides specific to cassava pathogens to control the spread of CGM.",
                "Maintain optimal soil conditions by using proper irrigation techniques to prevent water stagnation.",
                "Remove and replace highly susceptible varieties with those more tolerant to CGM.",
                "Strengthen plant defenses by ensuring proper nutrient application, particularly nitrogen and potassium.",
                "Regularly monitor the field for signs of disease and treat early using recommended agricultural practices."
            ],
            'Cassava Mosaic Disease': [
                "Remove and destroy infected plants immediately upon detection to prevent disease spread.",
                "Use systemic insecticides to reduce whitefly populations, which transmit CMD.",
                "Grow CMD-resistant cassava varieties available from agricultural extension services.",
                "Avoid using planting materials from fields where CMD symptoms were present in previous seasons.",
                "Employ cultural practices like maintaining plant spacing to reduce vector density and movement.",
                "Promote natural biological control by encouraging predators of whiteflies in cassava fields."
            ],
            'Healthy': [
                "Routinely inspect cassava plants for any early signs of disease and act promptly.",
                "Always start planting with certified, disease-free cuttings to ensure a healthy crop.",
                "Keep weeds under control to minimize habitat for vectors and secondary pathogens.",
                "Use fertilizers judiciously to maintain soil fertility and strengthen plant resilience against stress.",
                "Rotate cassava with non-host crops every season to break disease cycles in the soil.",
                "Educate farmers and workers on disease identification and management to prevent accidental spread."
            ]
        }

        farmer_tips = disease_tips.get(predicted_label, [])

        # Render the prediction output in `diseasepredout.html`
        return render_template('diseasepredout.html', predicted_label=predicted_label, farmer_tips=farmer_tips)

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for('disease_page'))

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
