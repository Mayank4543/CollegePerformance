import pickle
from flask import Flask, request, app, render_template,jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gspread
import os

app = Flask(__name__)
df = pd.read_csv(r'C:\Users\Mayank Rathore\Desktop\College-Predictor-System\College_data_updated.csv')
df = df.replace('--', np.nan)
numeric_columns = ['UG_fee', 'PG_fee', 'Rating', 'Academic', 'Faculty', 'Infrastructure', 'Placement']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Placement'])
college_names = df['College_Name']
features = ['Rating', 'Academic', 'Faculty', 'Infrastructure', 'UG_fee', 'PG_fee']
X = df[features].copy()
y = df['Placement']
for column in features:
    X.loc[:, column] = X[column].fillna(X[column].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
def recommend_colleges_flask(rating, academic, faculty, infrastructure, ug_fee, pg_fee, top_n=5):
    input_data = pd.DataFrame([[rating, academic, faculty, infrastructure, ug_fee, pg_fee]], 
                               columns=features)
    input_scaled = scaler.transform(input_data)
    predicted_placement = rf_model.predict(input_scaled)[0]

    all_colleges_df = pd.DataFrame({
        'College_Name': college_names,
        'Rating': df['Rating'],
        'Academic': df['Academic'],
        'Faculty': df['Faculty'],
        'Infrastructure': df['Infrastructure'],
        'UG_fee': df['UG_fee'],
        'PG_fee': df['PG_fee'],
        'Actual_Placement': df['Placement']
    })
    recommendations = all_colleges_df.sort_values('Actual_Placement', ascending=False).head(top_n)
    return predicted_placement, recommendations

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

def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# Normalize relevant columns
df["Placement Score"] = normalize(df["Placement"])
df["Rating Score"] = normalize(df["Rating"])
df["Faculty Score"] = normalize(df["Faculty"])
df["Infrastructure Score"] = normalize(df["Infrastructure"])

df["UG Fees Score"] = 1 - normalize(df["UG_fee"])  # Lower fees better
df["PG Fees Score"] = 1 - normalize(df["PG_fee"])  # Lower fees better

# Calculate final ranking score
df["Final Score"] = (
    (0.3 * df["Placement Score"]) +
    (0.2 * df["Rating Score"]) +
    (0.15 * df["Faculty Score"]) +
    (0.1 * df["Infrastructure Score"]) +
    (0.1 * df["UG Fees Score"]) +
    (0.05 * df["PG Fees Score"])
)

# Sort colleges by ranking score
df = df.sort_values(by="Final Score", ascending=False)
df["Rank"] = range(1, len(df) + 1)

# Convert to list of dicts for HTML rendering
colleges = df.to_dict(orient="records")

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
def login():
    return render_template('login.html')    

@app.route('/colleges')
def colleges():
    return render_template('Top Colleges.html') 
@app.route('/recommend-college')
def rc():
    return render_template('recommend.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    rating = float(request.form['rating'])
    academic = float(request.form['academic'])
    faculty = float(request.form['faculty'])
    infrastructure = float(request.form['infrastructure'])
    ug_fee = float(request.form['ug_fee'])
    pg_fee = float(request.form['pg_fee'])

    predicted_score, recommended_colleges = recommend_colleges_flask(rating, academic, faculty, infrastructure, ug_fee, pg_fee)
    
    # Convert recommendations to a list of dictionaries for easy rendering
    recommendations_list = recommended_colleges.to_dict(orient='records')

    return render_template('results.html', predicted_score=predicted_score, recommendations=recommendations_list)


@app.route('/feature-importance')
def feature_importance():
    importance = rf_model.feature_importances_
    feat_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)

    # Save the chart as an image
    plt.figure(figsize=(10, 6))
    plt.bar(feat_importance['Feature'], feat_importance['Importance'])
    plt.xticks(rotation=45)
    plt.title('Feature Importance in Predicting Placement')
    plt.tight_layout()
    chart_path = 'static/feature_importance.png'
    plt.savefig(chart_path)
    plt.close()

    return render_template('feature_importance.html', chart_path=chart_path)
@app.route('/learn')
def learn():
    return render_template('Coding.html')     

@app.route('/support')
def support():
    return render_template('support.html')   

@app.route('/faq')
def faq():
    return render_template('faq.html')              


@app.route('/predict', methods = ['POST'])
def predict():
   
    Category = {'0':'General', '1':'Other Backward Classes-Non Creamy Layer', '6':'Scheduled Castes', '8':'Scheduled Tribes',
                '3':'General & Persons with Disabilities', '5':'Other Backward Classes & Persons with Disabilities', 
                '7':'Scheduled Castes & Persons with Disabilities', '9':'Scheduled Tribes & Persons with Disabilities',
                '1':'General & Economically Weaker Section', '2':'General & Economically Weaker Section & Persons with Disability'}
    
    Quota = {'0':'All-India', '3':'Home-State', '1':'Andhra Pradesh', '2':'Goa', '4':'Jammu & Kashmir', '5':'Ladakh'}

    Pool = {'0':'Neutral', '1':'Female Only'}

    Institute = {'0':'IIT', '1':'NIT'}

    sa = gspread.service_account(filename="collegeranking-8af6481697ee.json")
    sh = sa.open("collegeranking")
    wks = sh.worksheet("collegeranking")   
    data = wks.get_all_values()
    

    data = [x for x in request.form.values()]
    print("Received Data:", data)
    print("Data Length:", len(data))
    list1 = data.copy()

   

    list1[4] = Category.get(list1[4])
    list1[5] = Quota.get(list1[5])
    list1[6] = Pool.get(list1[6])
    list1[7] = Institute.get(list1[7])

    print("Printing list1[4]:", list1[4])
    print("Printing list1[5]:", list1[5])
    print("Printing list1[6]:", list1[6])
    print("Printing list1[7]:", list1[7])

# Ensure data has enough elements before popping
    if len(data) > 1:
       data.pop(0)  # Remove first element
    if len(data) > 1:
       data.pop(0)  # Remove second element
    if len(data) > 8:  
       data.pop(8)  # Remove 9th element only if it exists

# Convert remaining elements to float safely
    try:
       data1 = [float(x) for x in data]
    except ValueError as e:
       print("Error converting to float:", e)
       data1 = []  # Handle error by setting empty list

    print("Final Processed Data:", data1)


    final_output = np.array(data1).reshape(1, -1)
    output = model.predict(final_output)[0]

    list1.append(output[0])
    list1.append(output[1])
    list1.append(output[2])
    wks.append_row(list1, table_range="A2:M2")

    return render_template("home.html", prediction_text = "College : {} ,  Degree : {} , Course : {}".format(output[0], output[1], output[2]), prediction = "Thank you, Hope this will match your requirement !!!")

if __name__ == '__main__':
    app.run(debug = True)
