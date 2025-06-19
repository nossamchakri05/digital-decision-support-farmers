from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Database connection configuration
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Nossam@2005'
DB_NAME = 'project'

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print("Database connection successful.")
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to the database: {err}")
        return None

# Load and preprocess yield model references globally
yield_data_path = 'synthetic_crop_yield_dataset.csv'  # Update path if necessary
yield_data = pd.read_csv(yield_data_path)

label_encoders = {}
categorical_columns = ['Type of Soil', 'Season', 'Type of Seeds',
                       'Type of Transplanting Method', 'Type of Irrigation Method',
                       'Type of Fertilizers Used', 'Yield Category']

for column in categorical_columns:
    le = LabelEncoder()
    yield_data[column] = le.fit_transform(yield_data[column])
    label_encoders[column] = le

# Fit scaler for numerical columns
scaler = StandardScaler()
numerical_columns = ['Area Ploughed (in acres)', 'Average Rainfall (in mm)']
scaler.fit(yield_data[numerical_columns])

# Load saved yield prediction model globally
yield_model_path = "cropyield.pkl"  # Update path if necessary
loaded_yield_model = joblib.load(yield_model_path)

# Load and preprocess subsidy model references globally
subsidy_data_path = 'synthetic_farmer_subsidy_dataset_with_names.csv'  # Update path if necessary
subsidy_data = pd.read_csv(subsidy_data_path)

subsidy_label_encoders = {}
categorical_cols = subsidy_data.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop('Subsidy Eligibility')

for col in categorical_cols:
    le = LabelEncoder()
    subsidy_data[col] = le.fit_transform(subsidy_data[col])
    subsidy_label_encoders[col] = le

subsidy_scaler = StandardScaler()
numerical_cols = ['Land Owned (in acres)', 'Annual Income (in INR)', 'Bank Loan (in INR)']
subsidy_scaler.fit(subsidy_data[numerical_cols])

# Load the subsidy prediction model globally
subsidy_model_path = 'best_subsidy_model_oversampling.pkl'  # Update path if necessary
subsidy_model = joblib.load(subsidy_model_path)

# Routes for registration, login, and main page
@app.route('/')
def default_page():
    return render_template('reg.html')

@app.route('/reg', methods=['GET', 'POST'])
def reg_page():
    return render_template('reg.html')

@app.route('/registation', methods=['GET', 'POST'])
def registration_page():
    if request.method == 'POST':
        full_name = request.form['name1']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password1']
        confirm_password = request.form['confirmPassword']
        aadhar_number = request.form['aadhar']
        contact_number = request.form['contactNumber']

        # Validation
        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('registration_page'))

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                query = """
                INSERT INTO users (full_name, username, email, password, aadhar_number, contact_number)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (full_name, username, email, hashed_password, aadhar_number, contact_number))
                conn.commit()
                cursor.close()
                conn.close()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('reg_page'))
            else:
                flash("Database connection failed.", "danger")
                return redirect(url_for('registration_page'))
        except mysql.connector.Error as err:
            flash(f"Error: {err}", "danger")
            return redirect(url_for('registration_page'))

    return render_template('registation.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['user_id']
                session['username'] = user['username']
                flash("Login successful!", "success")
                return redirect(url_for('main_page'))
            else:
                flash("Invalid username or password", "danger")
                return redirect(url_for('reg_page'))
        else:
            flash("Database connection failed.", "danger")
            return redirect(url_for('reg_page'))
    except mysql.connector.Error as err:
        flash(f"Error: {err}", "danger")
        return redirect(url_for('reg_page'))

@app.route('/main')
def main_page():
    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('reg_page'))
    return render_template('main.html')

# Yield Prediction Page
@app.route('/yields')
def yield_page():
    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('reg_page'))
    return render_template('yieldpredict.html')

# Disease Prediction Page
@app.route('/disease')
def disease_page():
    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('reg_page'))
    return render_template('diseasepredict.html')

# Subsidy Page
@app.route('/subsidy')
def subsidy_page():
    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('reg_page'))
    return render_template('subsidy.html')

# Yield Prediction Route
@app.route('/submit_yield', methods=['POST'])
def submit_yield():
    # Collect input from the form
    soil = request.form.get('soil')  # "Type of Soil"
    area = float(request.form.get('area'))  # "Area Ploughed (in acres)"
    rain = float(request.form.get('rain'))  # "Average Rainfall (in mm)"
    season = request.form.get('season')  # "Season"
    seed = request.form.get('seed')  # "Type of Seeds"
    transplant = request.form.get('transplant')  # "Type of Transplanting Method"
    irrigation = request.form.get('irrigation')  # "Type of Irrigation Method"
    fertilizer = request.form.get('fertilizer')  # "Type of Fertilizers Used"

    # Encode categorical features
    encoded_input = [
        label_encoders['Type of Soil'].transform([soil])[0],
        label_encoders['Season'].transform([season])[0],
        label_encoders['Type of Seeds'].transform([seed])[0],
        label_encoders['Type of Transplanting Method'].transform([transplant])[0],
        label_encoders['Type of Irrigation Method'].transform([irrigation])[0],
        label_encoders['Type of Fertilizers Used'].transform([fertilizer])[0]
    ]

    # Scale numerical features
    scaled_values = scaler.transform([[area, rain]])[0]

    # Combine encoded categorical and scaled numerical features
    final_input = np.hstack((scaled_values, encoded_input))

    # Predict yield category
    prediction = loaded_yield_model.predict([final_input])[0]

    # Decode the predicted category back to its label
    predicted_yield_category = label_encoders['Yield Category'].inverse_transform([prediction])[0]

    # Insert the prediction details into the database
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            query = """
            INSERT INTO project.yield_prediction_details (user_id, area_ploughed, soil_type, average_rainfall, season, seed_type, transplanting_method, irrigation_method, fertilizer_type, prediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session['user_id'], area, soil, rain, season, seed, transplant, irrigation, fertilizer, predicted_yield_category))
            conn.commit()
            cursor.close()
            conn.close()
        else:
            flash("Database connection failed.", "danger")
            return redirect(url_for('yield_page'))
    except mysql.connector.Error as err:
        flash(f"Error: {err}", "danger")
        return redirect(url_for('yield_page'))

    # Define suggestions based on prediction
    suggestions = {
        "Low": [
            "Conduct soil testing and correct deficiencies with organic manure and fertilizers.",
            "Implement irrigation systems, such as rainwater harvesting or drip irrigation, to ensure consistent water supply.",
            "Use disease-resistant, high-yielding seed varieties tailored to the local environment.",
            "Educate farmers on seed treatment, crop rotation, and basic farming practices.",
            "Address pest and disease issues using Integrated Pest Management (IPM) techniques.",
            "Enhance soil structure by incorporating green manure and cover crops.",
            "Optimize planting density and spacing to prevent overcrowding and ensure healthy growth.",
            "Provide farmer training programs to build knowledge on effective farming methods."
        ],
        "Medium": [
            "Apply balanced fertilizers (NPK) based on soil test results to prevent nutrient imbalances.",
            "Regularly monitor and control weeds to reduce competition for nutrients and water.",
            "Adopt efficient irrigation methods, like drip or sprinkler systems, to optimize water use.",
            "Incorporate intercropping or multiple cropping systems to maximize land productivity.",
            "Improve post-harvest storage facilities to reduce losses and maintain crop quality.",
            "Use weather forecasts and digital tools to plan farming activities effectively.",
            "Ensure proper timing of sowing, irrigation, and harvesting to avoid yield losses.",
            "Train farmers on advanced techniques like zero-tillage and mulching to conserve resources."
        ],
                "High": [
            "Integrate precision farming technologies, such as sensors and drones, for real-time crop monitoring.",
            "Rotate crops with nitrogen-fixing legumes to maintain soil health and fertility.",
            "Regularly enrich soil with compost, biofertilizers, or other organic inputs.",
            "Use advanced farm machinery to improve efficiency in planting, irrigation, and harvesting.",
            "Ensure proper pest and disease management using eco-friendly and targeted methods.",
            "Implement smart irrigation systems with IoT devices to minimize water wastage.",
            "Focus on continuous training for farmers to keep up with evolving agricultural practices.",
            "Diversify crops by including high-value or niche crops for additional income streams."
        ],
        "Very High": [
            "Promote sustainable farming techniques like agroforestry, crop rotation, and organic farming.",
            "Collaborate with research institutions to adopt the latest farming technologies and seed varieties.",
            "Focus on quality improvement to meet export standards and increase market value.",
            "Use digital tools and AI-based platforms to optimize farm management and decision-making.",
            "Implement water conservation techniques, such as precision irrigation and moisture retention methods.",
            "Introduce value-added processes like on-site grading, packaging, and processing to enhance profits.",
            "Reduce risks from climate variability with crop insurance and adaptive farming methods.",
            "Regularly reassess strategies to improve productivity while maintaining environmental sustainability."
        ]
    }

    farmer_suggestions = suggestions.get(predicted_yield_category, [])

    return render_template('yieldpredout.html', predicted_yield_category=predicted_yield_category, farmer_suggestions=farmer_suggestions)

# Subsidy Prediction Route
@app.route('/subsidyout', methods=['POST'])
def subsidyout():
    schemes = {
        "PM Krishi Sinchayee Yojana": [
            "Visit the official website of PMKSY or contact the local agriculture department.",
            "Check eligibility criteria (e.g., farmer category, land ownership).",
            "Prepare necessary documents: Aadhaar card, land ownership proof, and bank details.",
            "Fill out the application form available at the nearest agriculture office or online.",
            "Submit the form along with required documents to the local implementing agency.",
            "Follow up with the agriculture office for updates on the status of the application.",
            "Utilize the subsidy for drip irrigation, sprinklers, or water conservation systems as approved."
        ],
        "Kisan Credit Card Subsidy": [
            "Visit the nearest bank or cooperative society participating in the scheme.",
            "Check eligibility (farmers engaged in agriculture, animal husbandry, or fisheries).",
            "Collect and fill out the KCC application form with details about landholding and farming activities.",
            "Attach necessary documents: Aadhaar, land records, income certificate, and bank passbook.",
            "Submit the completed application to the bank along with required documents.",
            "Attend the verification process conducted by the bank’s field officers.",
            "Upon approval, receive the KCC, which provides credit at subsidized interest rates."
        ],
        "NFSM": [
            "Contact the district agriculture office or Krishi Vigyan Kendra (KVK) for guidance.",
            "Check eligibility for schemes under NFSM (focused on crops like rice, wheat, pulses).",
            "Obtain and fill the NFSM application form for the desired component (e.g., seed distribution, machinery).",
            "Attach supporting documents: Aadhaar, land records, and bank account details.",
            "Submit the application at the local agriculture office or online portal, if available.",
            "Follow up for verification of the application by authorities.",
            "Avail benefits like seeds, training, or equipment support under the NFSM program."
        ],
        "Soil Health Card Scheme": [
            "Visit the nearest agriculture department or Krishi Vigyan Kendra (KVK).",
            "Register under the Soil Health Card Scheme by providing basic farmer and land details.",
            "Submit soil samples to the designated laboratory for analysis.",
            "Provide necessary documents: Aadhaar card, land records, and contact details.",
            "Wait for the laboratory to analyze the soil samples.",
            "Receive the Soil Health Card, which provides recommendations for fertilizers and crop patterns.",
            "Use the card's advice to optimize soil health and improve crop yield."
        ],
        "RKVY": [
            "Visit the state agriculture department or check the RKVY website for details.",
            "Identify the components of RKVY that suit your requirements (e.g., infrastructure development).",
            "Collect and fill out the RKVY application form for the desired subsidy program.",
            "Attach necessary documents: Aadhaar, land ownership proof, and bank details.",
            "Submit the application to the designated nodal officer at the district or state level.",
            "Follow up for verification and approval by the authorities.",
            "Once approved, use the subsidy for agricultural infrastructure, inputs, or other approved activities."
        ]
    }

    # Collect input from the form
    own = request.form.get('own')
    landowned = float(request.form.get('landowned'))
    region = request.form.get('region')
    fert = request.form.get('fertilizer')
    equipment = request.form.get('equipment')
    income = float(request.form.get('income'))
    loan = float(request.form.get('loan'))
    livestock = request.form.get('livestock')
    prevsub = request.form.get('prevsub')

    # Print all the input values for debugging
    print("Received Inputs:")
    print(f"Land Ownership: {own}")
    print(f"Land Owned (in acres): {landowned}")
    print(f"Region: {region}")
    print(f"Fertilizer Type: {fert}")
    print(f"Farming Equipment Ownership: {equipment}")
    print(f"Annual Income: {income}")
    print(f"Bank Loan (in INR): {loan}")
    print(f"Livestock Ownership: {livestock}")
    print(f"Previous Subsidies: {prevsub}")

    # Prepare the input for prediction
    sample_input = {
        'Ownership of Land': [own],
        'Land Owned (in acres)': [landowned],
        'Location': [region],
        'Types of Fertilizers Used': [fert],
        'Owning Farming Equipment': [equipment],
        'Annual Income (in INR)': [income],
        'Bank Loan (in INR)': [loan],
        'Ownership of Livestock': [livestock],
        'Previous Subsidy Eligibility': [prevsub]
    }

    input_df = pd.DataFrame(sample_input)

    # Encode categorical variables
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = subsidy_label_encoders[col].transform(input_df[col])

    # Scale numerical variables
    input_df[numerical_cols] = subsidy_scaler.transform(input_df[numerical_cols])

    # Prediction using the preloaded model
    try:
        prediction = subsidy_model.predict(input_df)
        print("Prediction successful:", prediction)
    except ValueError as e:
        flash(f"Prediction failed: {e}", "danger")
        print(f"Prediction error: {e}")
        return redirect(url_for('subsidy_page'))

    # Decode prediction
    subsidy_label_encoder = LabelEncoder()
    subsidy_data['Subsidy Eligibility'] = subsidy_label_encoder.fit_transform(subsidy_data['Subsidy Eligibility'])
    predicted_category = subsidy_label_encoder.inverse_transform(prediction)[0]

    # Print prediction result
    print(f"Predicted Category: {predicted_category}")

    # Get farmer scheme information
    farmer_scheme = schemes.get(predicted_category, [])

    # Insert the subsidy details into the database
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            query = """
            INSERT INTO project.subsidy_queries (user_id, land_ownership, land_owned_acres, location, fertilizer_type,
                equipment_ownership, annual_income, loan_balance, irrigation_facility, livestock_ownership, previous_subsidies, predicted_output)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session['user_id'], own, landowned, region, fert, equipment, income, loan, 'N/A', livestock, prevsub, predicted_category))
            conn.commit()
            cursor.close()
            conn.close()
            print("Subsidy data successfully inserted.")
        else:
            flash("Database connection failed.", "danger")
            return redirect(url_for('subsidy_page'))
    except mysql.connector.Error as err:
        flash(f"Error: {err}", "danger")
        return redirect(url_for('subsidy_page'))

    # Render the output page with the prediction result
    return render_template('subsidypredout.html', predicted_category=predicted_category, farmer_scheme=farmer_scheme)

if __name__ == '__main__':
    app.run(debug=True)

