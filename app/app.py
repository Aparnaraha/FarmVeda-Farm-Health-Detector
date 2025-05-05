from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import config
import pickle
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.soil_model import predict_soil
from markupsafe import Markup
import io
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic
from functools import wraps

# ==============================================================================================

# ------------------------- DATABASE CONFIGURATION -------------------------------------------
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database to store users and contact inquiries
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'kqrM+BuveC9CIYb5AopbSyGVS6gFoV6+'  # Secret key for session management

# Initialize the database
db = SQLAlchemy(app)

# User model for login/signup
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Contact model to store inquiries
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)

# ==============================================================================================

# ------------------------- LOADING THE TRAINED MODELS ---------------------------------------
# Loading plant disease classification model
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                   'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                   'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                   'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                   'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                   'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                   'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# =========================================================================================

# SOIL RECOMMENDATION DICTIONARY
soil_recommendations = {
    "Black Soil": "Cotton, Soybean, Groundnut, Sunflower",
    "Cinder Soil": "Not ideal for cultivation, but can grow hardy crops like Millets",
    "Laterite Soil": "Cashew, Coffee, Tea, Coconut",
    "Peat Soil": "Paddy, Sugarcane, Jute",
    "Yellow Soil": "Pulses, Oilseeds, Maize"
}

# =========================================================================================

# Custom functions for calculations
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity or None if error
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    try:
        response = requests.get(complete_url)
        x = response.json()
        if x["cod"] != "404":
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)  # Convert Kelvin to Celsius
            humidity = y["humidity"]
            return temperature, humidity
        else:
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        image = Image.open(io.BytesIO(img))  # Ensure the image is read correctly
        img_t = transform(image)
        img_u = torch.unsqueeze(img_t, 0)

        # Get predictions from model
        yb = model(img_u)
        _, preds = torch.max(yb, dim=1)
        prediction = disease_classes[preds[0].item()]
        return prediction
    except Exception as e:
        print(f"Error in disease prediction: {e}")
        return None

# ===============================================================================================

# -------------------- LOGIN REQUIRED DECORATOR ----------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("You must be logged in to access this page.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------- FLASK ROUTES -------------------------------------------------

@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)


# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('app/Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@app.route('/soil')
@login_required
def soil_form():
    title = 'Harvestify - Soil Type Detection'
    return render_template('soil.html', title=title)

@app.route('/soil-predict', methods=['POST'])
@login_required
def soil_prediction():
    title = 'Harvestify - Soil Type Result'
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return redirect(request.url)
        img_bytes = file.read()
        soil_type = predict_soil(img_bytes)
        crops = soil_recommendations.get(soil_type, "Data not available")
        return render_template('soil-result.html', title=title, soil_type=soil_type, crops=crops)

@app.route('/login', methods=['GET', 'POST'])
def login():
    title = 'Harvestify - Login'
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate user credentials
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user_id in session
            flash("Login successful!", "success")
            return redirect(url_for('home'))  # Redirect to home page after successful login
        else:
            flash("Invalid username or password.", "error")
            return render_template('login.html', title=title)
    
    return render_template('login.html', title=title)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    title = 'Harvestify - Signup'
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Check if all fields are filled
        if not username or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return render_template('signup.html', title=title)

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('signup.html', title=title)

        # Check if username or email already exists
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Username or email already exists.", "danger")
            return render_template('signup.html', title=title)

        # Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))  # 🔁 Redirects to login page

    return render_template('signup.html', title=title)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    title = 'Harvestify - Contact Us'
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if name and email and message:
            new_contact = Contact(name=name, email=email, message=message)
            db.session.add(new_contact)
            db.session.commit()
            flash("Your message has been sent!", "success")

            # ✅ Render the result page directly with submitted data
            return render_template('contact-result.html', name=name, email=email, message=message, title='Message Sent')

        else:
            flash("All fields are required.", "error")

    return render_template('contact.html', title=title)

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user_id from session
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

# ===============================================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # ✅ Creates the 'contact' and 'user' tables
    app.run(debug=True)
