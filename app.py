import pickle
from flask import Flask, request, app, render_template,jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gspread

from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

from flask_pymongo import PyMongo
from datetime import timedelta
from bson.objectid import ObjectId
import os
import datetime

app = Flask(__name__)


app.config["MONGO_URI"] = "mongodb://localhost:27017/college_predictor"
app.config["JWT_SECRET_KEY"] = "your_secret_key"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

users_collection = mongo.db.users

# ✅ Ensure admin user exists in MongoDB
admin_user = {
    "email": "mayankrathore9897@gmail.com",
    "password": bcrypt.generate_password_hash("1101009109").decode("utf-8"),
    "role": "admin"
}

if not users_collection.find_one({"email": admin_user["email"]}):
    users_collection.insert_one(admin_user)
    print("✅ Admin user added successfully!")
else:
    print("⚠️ Admin already exists!")

# ✅ User Registration Route
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    users_collection.insert_one({"email": email, "password": hashed_password, "role": "user"})
    
    return jsonify({"message": "Registration successful"}), 201

# ✅ User Login Route
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})

    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    role = user.get("role", "user")  # Ensuring default role if missing
    print("role",role)

    # Generate JWT Token
    access_token = create_access_token(identity={"email": user["email"], "role": role})

    print("✅ Generated JWT Token Payload:", {"email": user["email"], "role": role})  # Debugging

    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "redirect": "/dashboard" if role == "admin" else "/"
    }), 200




# ✅ Admin Dashboard Route (Only for Admins)
@app.route("/dashboard", methods=["GET"])

def admin_dashboard():
    return render_template("dashboard/dashboard.html")




# Load dataset
df = pd.read_csv(r'C:\Users\Mayank Rathore\Desktop\College-Predictor-System\College_data_updated.csv')
df = df.replace('--', np.nan)

numeric_columns = ['Avg_Package', 'Rating', 'Academic', 'Faculty', 'Infrastructure', 'Placement']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=['Placement'], inplace=True)

college_names = df['College_Name']
features = ['Rating', 'Academic', 'Faculty', 'Infrastructure', 'Avg_Package']
X = df[features].fillna(df[features].mean())
y = df['Placement']

# Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Function to predict and recommend colleges
def recommend_colleges_flask(rating, academic, faculty, infrastructure, avg_package, top_n=5):
    input_data = pd.DataFrame([[rating, academic, faculty, infrastructure, avg_package]], columns=features)
    input_scaled = scaler.transform(input_data)
    
    predicted_placement = round(rf_model.predict(input_scaled)[0], 2)

    # College Recommendation Logic (Predicted Placement ke close colleges)
    df['Placement_Diff'] = abs(df['Placement'] - predicted_placement)

    recommended_colleges = df.sort_values(by='Placement_Diff').head(top_n)[['College_Name', 'Placement', 'Rating', 'Avg_Package']]

    print("Predicted Placement Score:", predicted_placement)
    print("Recommended Colleges:\n", recommended_colleges)  # Debugging ke liye

    return predicted_placement, recommended_colleges



# Load the model
model = pickle.load(open("model1.pkl", "rb"))

@app.route('/')
def home():

    return render_template('home.html')
@app.route("/get_colleges", methods=["GET"])
def get_colleges():
    colleges = df["College_Name"].unique().tolist()
    return jsonify(colleges)

@app.route("/compare_colleges", methods=["GET"])

def compare_colleges():
    college1 = request.args.get("college1", "").strip()
    college2 = request.args.get("college2", "").strip()

    if not college1 or not college2:
        return jsonify({"error": "Both college names must be provided"}), 400

    # Ensure proper matching
    df["College_Name"] = df["College_Name"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Case-insensitive search
    data1 = df[df["College_Name"].str.lower() == college1.lower()]
    data2 = df[df["College_Name"].str.lower() == college2.lower()]

    if data1.empty or data2.empty:
        return jsonify({
            "error": "One or both colleges not found in the dataset",
            "available_colleges": df["College_Name"].unique().tolist()
        }), 404

    data1 = data1.iloc[0].to_dict()
    data2 = data2.iloc[0].to_dict()

    features = ["Rating", "Placement", "Avg_Package", "Faculty", "Infrastructure"]
    comparison_data = {
        college1: {feature: data1[feature] for feature in features},
        college2: {feature: data2[feature] for feature in features}
    }

    return jsonify(comparison_data)



@app.route("/ranking")
def ranking():
   
    return render_template("ranking.html")
def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

@app.route('/get_ranking', methods=['GET'])
def get_ranking():
    """API to fetch ranked college data based on filters with Pagination"""

    # Get user selections from request
    stream = request.args.get("Stream")
    state = request.args.get("State")
    page = int(request.args.get("page", 1))  # Default page = 1
    limit = int(request.args.get("limit", 10))  # Default limit = 10

    # Apply filtering
    filtered_df = df[
        (df["Stream"] == stream) & 
        (df["State"] == state)
    ]

    if filtered_df.empty:
        return jsonify({"message": "No colleges found for the selected criteria"}), 404

    # Normalize relevant columns for ranking
    filtered_df["Placement_Norm"] = normalize(filtered_df["Placement"])
    filtered_df["Rating_Norm"] = normalize(filtered_df["Rating"])
    filtered_df["Faculty_Norm"] = normalize(filtered_df["Faculty"])

    # Compute ranking score
    filtered_df["Ranking_Score"] = (
        (0.4 * filtered_df["Placement_Norm"]) +
        (0.3 * filtered_df["Rating_Norm"]) +
        (0.3 * filtered_df["Faculty_Norm"])
    )

    # Sort and assign rank
    filtered_df = filtered_df.sort_values(by="Ranking_Score", ascending=False)
    filtered_df["Rank"] = range(1, len(filtered_df) + 1)

    # Convert ratings to star format
    filtered_df["Star_Review"] = filtered_df["Rating"].apply(lambda x: f"⭐ {round((x / 10) * 5, 1)} / 5")

    # Pagination Logic (Make sure pagination works correctly)
    total_colleges = len(filtered_df)
    total_pages = (total_colleges + limit - 1) // limit  # Ceiling division

    if total_colleges > 10:  # Only paginate if more than 10 records
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_df = filtered_df.iloc[start_idx:end_idx]
    else:
        paginated_df = filtered_df  # Show all data if <= 10

    # Select only required columns for frontend
    final_df = paginated_df[["Rank", "College_Name", "Avg_Package", "Placement", "Star_Review"]]

    return jsonify({
        "total_pages": total_pages,
        "current_page": page,
        "total_records": total_colleges,
        "data": final_df.to_dict(orient="records")
    })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')  
 
@app.route('/login')
def lg():
    return render_template('login.html')
@app.route('/register')
def registr():
    return render_template('register.html')    
@app.route('/recommend-college')
def  rc():
    return render_template('recommend.html')  
@app.route('/recommend', methods=['POST'])
def recommend():
    rating = float(request.form['rating'])
    academic = float(request.form['academic'])
    faculty = float(request.form['faculty'])
    infrastructure = float(request.form['infrastructure'])
    avg_package = float(request.form['Average_Package'])

    predicted_score, recommended_colleges = recommend_colleges_flask(rating, academic, faculty, infrastructure, avg_package)
    
    recommendations_list = recommended_colleges.to_dict(orient='records')

    return render_template('recommend.html', predicted_score=predicted_score, recommendations=recommendations_list)

@app.route('/learn')
def learn():
    return render_template('Coding.html')     
   

Category = {
    '0': 'General', '1': 'OBC-NCL', '6': 'SC', '8': 'ST',
    '3': 'General-PwD', '5': 'OBC-PwD', '7': 'SC-PwD',
    '9': 'ST-PwD', '10': 'EWS', '11': 'EWS-PwD'
}

Quota = {
    '0': 'All-India', '3': 'Home-State', '1': 'Andhra Pradesh',
    '2': 'Goa', '4': 'Jammu & Kashmir', '5': 'Ladakh'
}

Pool = {'0': 'Neutral', '1': 'Female Only'}
Institute = {'0': 'IIT', '1': 'NIT'}

@app.route('/predict', methods = ['POST'])
def predict():

    Category = {'0':'General', '1':'Other Backward Classes-Non Creamy Layer', '6':'Scheduled Castes', '8':'Scheduled Tribes',
                '3':'General & Persons with Disabilities', '5':'Other Backward Classes & Persons with Disabilities', 
                '7':'Scheduled Castes & Persons with Disabilities', '9':'Scheduled Tribes & Persons with Disabilities',
                '1':'General & Economically Weaker Section', '2':'General & Economically Weaker Section & Persons with Disability'}
    
    Quota = {'0':'All-India', '3':'Home-State', '1':'Andhra Pradesh', '2':'Goa', '4':'Jammu & Kashmir', '5':'Ladakh'}

    Pool = {'0':'Neutral', '1':'Female Only'}

    Institute = {'0':'IIT', '1':'NIT'}

    sa = gspread.service_account(filename="collegeranking-b56d0233ac8b.json")
    sh = sa.open("collegeranking")
    wks = sh.worksheet("collegeranking")

    data = [x for x in request.form.values()]
    
    list1 = data.copy()

    list1[2] = Category.get(list1[2])
    list1[3] = Quota.get(list1[3])
    list1[4] = Pool.get(list1[4])
    list1[5] = Institute.get(list1[5])

    data.pop(0)
    data.pop(0)
    data.pop(7)
    data1 = [float(x) for x in data]

    final_output = np.array(data1).reshape(1, -1)
    output = model.predict(final_output)[0]

    list1.append(output[0])
    list1.append(output[1])
    list1.append(output[2])
    wks.append_row(list1, table_range="A2:M2")

    return render_template("home.html", prediction_text = "College : {} ,  Degree : {} , Course : {}".format(output[0], output[1], output[2]), prediction = "Thank you, Hope this will match your requirement !!!")



if __name__ == '__main__':
    app.run(debug = True)
