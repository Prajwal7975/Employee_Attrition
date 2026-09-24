from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Example department encoder classes (use same as training)
departments = ['sales', 'technical', 'support', 'IT', 'hr', 'accounting', 'marketing', 'product_mng', 'RandD', 'management']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        satisfaction = float(request.form['satisfaction'])
        evaluation = float(request.form['evaluation'])
        num_projects = int(request.form['num_projects'])
        monthly_hours = int(request.form['monthly_hours'])
        years_at_company = int(request.form['years_at_company'])
        accident = int(request.form['accident'])
        promotion = int(request.form['promotion'])
        salary_level = request.form['salary']
        department = request.form['department'].lower()

        # Salary encoding
        salary_map = {'low': 0, 'medium': 1, 'high': 2}
        salary_encoded = salary_map.get(salary_level, 0)

        # Department encoding
        if department not in departments:
            department = departments[0]
        dept_encoded = departments.index(department)

        # Prepare input
        features = np.array([[satisfaction, evaluation, num_projects, monthly_hours,
                              years_at_company, accident, promotion,
                              salary_encoded, dept_encoded]])

        prediction = model.predict(features)[0]

        result = "Employee is likely to LEAVE ❌" if prediction == 1 else "Employee is likely to STAY ✅"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)