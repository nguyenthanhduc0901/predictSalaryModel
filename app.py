# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import numpy as np
import json
import os
from datetime import datetime, timezone
from threading import Lock

app = Flask(__name__)

# Cấu hình Secret Key cho Flask (cần thiết cho Flash messages)
app.config['SECRET_KEY'] = 'your_secure_secret_key_here'  # Thay thế bằng khóa bí mật của bạn

# Đường dẫn tới tệp JSON lưu trữ lịch sử dự đoán
HISTORY_FILE = 'prediction_history.json'

# Khóa để đồng bộ hóa việc ghi vào tệp JSON
lock = Lock()

# Tải mô hình đã huấn luyện
model_path = os.path.join(os.path.dirname(__file__), 'predict_salary_random_forest.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")
model = joblib.load(model_path)

# Danh sách các cột trong dữ liệu
columns = [
    'Experience', 'Job Level', 'Follower', 'Education Levels',
    'Employment Type: Internship', 'Employment Type: Part-time',
    'Employment Type: Official', 'Employment Type: Freelance',
    'Industry: accounting_auditing', 'Industry: administrative_secretarial',
    'Industry: advertising_public_relations_media', 'Industry: agriculture',
    'Industry: aquaculture_seafood', 'Industry: architecture', 'Industry: aviation',
    'Industry: banking', 'Industry: biotechnology', 'Industry: chemistry',
    'Industry: construction', 'Industry: consulting', 'Industry: customer_service',
    'Industry: education_training', 'Industry: electricity_electronics_refrigeration',
    'Industry: entertainment', 'Industry: environment', 'Industry: event_organization',
    'Industry: executive_management', 'Industry: finance_investment',
    'Industry: fine_arts_art_design', 'Industry: food_beverages',
    'Industry: food_technology_nutrition', 'Industry: forestry',
    'Industry: healthcare_medical', 'Industry: household_goods_personal_care',
    'Industry: human_resources', 'Industry: import_export', 'Industry: insurance',
    'Industry: interior_exterior_design', 'Industry: irrigation',
    'Industry: it_hardware_networking', 'Industry: it_software', 'Industry: law_legal',
    'Industry: library', 'Industry: livestock_veterinary', 'Industry: maintenance_repair',
    'Industry: manual_labor', 'Industry: manufacturing_operations', 'Industry: maritime',
    'Industry: marketing', 'Industry: mechanical_automotive_automation',
    'Industry: minerals', 'Industry: new_graduates_internship', 'Industry: ngo_non_profit',
    'Industry: not_found', 'Industry: occupational_safety', 'Industry: oil_gas',
    'Industry: online_marketing', 'Industry: other_industries',
    'Industry: pharmaceuticals', 'Industry: postal_telecommunications',
    'Industry: printing_publishing', 'Industry: procurement_supplies',
    'Industry: quality_management_qaqc', 'Industry: real_estate',
    'Industry: restaurant_hotel', 'Industry: retail_wholesale',
    'Industry: sales_business', 'Industry: securities', 'Industry: security_protection',
    'Industry: statistics', 'Industry: surveying_geology',
    'Industry: television_journalism_editing', 'Industry: textile_leather_fashion',
    'Industry: tourism', 'Industry: translation_interpretation',
    'Industry: transportation_logistics_warehouse', 'Industry: wooden_goods',
    'Welfare: allowance', 'Welfare: allowance thâm niên', 'Welfare: annual_leave',
    'Welfare: bonus', 'Welfare: business_trip_expenses', 'Welfare: healthcare',
    'Welfare: insurance', 'Welfare: laptop', 'Welfare: not_specified',
    'Welfare: overseas_travel', 'Welfare: salary_increase', 'Welfare: shuttle_service',
    'Welfare: sports_club', 'Welfare: training', 'Welfare: travel', 'Welfare: uniform',
    'Location_An Giang', 'Location_Ba Ria - Vung Tau', 'Location_Bac Can',
    'Location_Bac Giang', 'Location_Bac Lieu', 'Location_Bac Ninh', 'Location_Bangkok',
    'Location_Ben Tre', 'Location_Binh Duong', 'Location_Binh Phuoc', 'Location_Binh Thuan',
    'Location_Binh Đinh', 'Location_Ca Mau', 'Location_Can Tho', 'Location_Cao Bang',
    'Location_Champasak', 'Location_Dak Lak', 'Location_Dak Nong', 'Location_Gia Lai',
    'Location_Ha Giang', 'Location_Ha Nam', 'Location_Ha Noi', 'Location_Ha Tinh',
    'Location_Hai Duong', 'Location_Hai Phong', 'Location_Hau Giang', 'Location_Ho Chi Minh',
    'Location_Hoa Binh', 'Location_Hokkaido', 'Location_Hung Yen', 'Location_Khanh Hoa',
    'Location_Kien Giang', 'Location_Kon Tum', 'Location_Kratie', 'Location_Kv Bac Trung Bo',
    'Location_Kv Nam Trung Bo', 'Location_Kv Tay Nguyen', 'Location_Kv Đong Nam Bo',
    'Location_Lai Chau', 'Location_Lam Đong', 'Location_Lang Son', 'Location_Lao Cai',
    'Location_Long An', 'Location_Malaysia', 'Location_Nghe An', 'Location_Ninh Binh',
    'Location_Ninh Thuan', 'Location_Not Specified', 'Location_Phnompenh',
    'Location_Phu Tho', 'Location_Phu Yen', 'Location_Quang Binh', 'Location_Quang Nam',
    'Location_Quang Ngai', 'Location_Quang Ninh', 'Location_Quang Tri',
    'Location_Soc Trang', 'Location_Son La', 'Location_Svay Rieng', 'Location_Tay Ninh',
    'Location_Thai Binh', 'Location_Thai Nguyen', 'Location_Thanh Hoa',
    'Location_Thua Thien- Hue', 'Location_Tien Giang', 'Location_Toan Quoc',
    'Location_Tokyo', 'Location_Tra Vinh', 'Location_Tuyen Quang', 'Location_Vinh Long',
    'Location_Vinh Phuc', 'Location_Xiangkhouang', 'Location_Yen Bai',
    'Location_Yokohama', 'Location_Đa Nang', 'Location_Đien Bien',
    'Location_Đong Bang Song Cuu Long', 'Location_Đong Nai', 'Location_Đong Thap',
    'Language requirement', 'Gender_requirement_both', 'Gender_requirement_female',
    'Gender_requirement_male'
]

def load_history():
    """Load prediction history from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    with lock:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

def save_history(history):
    """Save prediction history to JSON file."""
    with lock:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

@app.route('/')
def home():
    # Lấy tất cả lịch sử dự đoán từ tệp JSON, sắp xếp theo ngày mới nhất
    history = load_history()
    return render_template('index.html', result=None, history=history)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        experience = float(request.form.get('experience', 0))
        job_level = int(request.form.get('job_level', 0))
        follower = int(request.form.get('follower', 0))
        education_levels = int(request.form.get('education', 0)) 
        employment_type = request.form.get('employment_type', 'None')
        selected_industries = request.form.getlist('industries')
        selected_welfares = request.form.getlist('welfares')
        selected_locations = request.form.getlist('locations')
        language_requirement = request.form.get('language_requirement', '0')
        gender_requirement = request.form.get('gender_requirement', 'None')

        # Chuẩn bị các cột phân loại
        employment_data = {
            'Employment Type: Internship': 1 if employment_type == 'Internship' else 0,
            'Employment Type: Part-time': 1 if employment_type == 'Part-time' else 0,
            'Employment Type: Official': 1 if employment_type == 'Official' else 0,
            'Employment Type: Freelance': 1 if employment_type == 'Freelance' else 0,
        }

        # Chuẩn bị các cột phân loại ngành nghề
        industry_data = {col: 1 if col in selected_industries else 0 for col in columns if 'Industry:' in col}

        # Chuẩn bị các cột phân loại phúc lợi
        welfare_data = {col: 1 if col in selected_welfares else 0 for col in columns if 'Welfare:' in col}

        # Chuẩn bị các cột phân loại địa điểm
        location_data = {col: 1 if col in selected_locations else 0 for col in columns if 'Location_' in col}

        # Chuẩn bị các cột phân loại giới tính
        gender_data = {
            'Gender_requirement_both': 1 if gender_requirement == 'Both' else 0,
            'Gender_requirement_female': 1 if gender_requirement == 'Female' else 0,
            'Gender_requirement_male': 1 if gender_requirement == 'Male' else 0,
        }

        # Chuẩn bị dữ liệu đầu vào cho mô hình
        input_data = {
            'Experience': experience,
            'Job Level': job_level,
            'Follower': follower,
            'Education Levels': education_levels,
            'Language requirement': int(language_requirement)
        }

        # Thêm dữ liệu phân loại vào input_data
        input_data.update(employment_data)
        input_data.update(industry_data)
        input_data.update(welfare_data)
        input_data.update(location_data)
        input_data.update(gender_data)

        # Chuyển đổi input_data thành DataFrame
        input_df = pd.DataFrame([input_data])

        # Đảm bảo rằng các cột trong input_df theo đúng thứ tự như mô hình đã huấn luyện
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Dự đoán
        prediction = model.predict(input_df)[0]

        # Chuyển đổi prediction sang float để JSON có thể serialize
        prediction = float(prediction)

        # Tạo bản ghi dự đoán mới
        new_prediction = {
            "id": len(load_history()) + 1,  # Tạo ID tự động
            "experience": experience,
            "job_level": job_level,
            "follower": follower,
            "education_levels": education_levels,
            "employment_type_internship": employment_data['Employment Type: Internship'],
            "employment_type_part_time": employment_data['Employment Type: Part-time'],
            "employment_type_official": employment_data['Employment Type: Official'],
            "employment_type_freelance": employment_data['Employment Type: Freelance'],
            "industry": ', '.join(selected_industries),
            "welfare": ', '.join(selected_welfares),
            "location": ', '.join(selected_locations),
            "language_requirement": 'Có' if language_requirement == '1' else 'Không',
            "gender_requirement": gender_requirement,
            "result": prediction,
            "date": datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M:%S')  # Sử dụng datetime có múi giờ
        }

        # Lấy lịch sử hiện tại, thêm dự đoán mới và lưu lại
        history = load_history()
        history.insert(0, new_prediction)  # Thêm vào đầu danh sách để hiển thị mới nhất lên trên
        save_history(history)

        # Tạo kết quả hiển thị
        result = f"Predicted Salary: {prediction:,.2f} VND"

    except ValueError as e:
        result = f"Invalid input: {e}"
    except Exception as e:
        result = f"An error occurred: {e}"

    # Lấy lại lịch sử dự đoán sau khi thêm mới
    history = load_history()

    return render_template('index.html', result=result, history=history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Xóa toàn bộ lịch sử dự đoán bằng cách ghi đè tệp JSON với danh sách rỗng
        save_history([])
        flash("Đã xóa toàn bộ lịch sử dự đoán.", 'success')
    except Exception as e:
        flash(f"Đã xảy ra lỗi khi xóa lịch sử: {e}", 'danger')
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
