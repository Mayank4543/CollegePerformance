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

    sa = gspread.service_account(filename="collegeranking.json")
    sh = sa.open("collegeranking")
    wks = sh.worksheet("collegeranking")   
    data = wks.get_all_values()
    

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
