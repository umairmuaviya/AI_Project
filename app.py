import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from fpdf import FPDF
import os
import zipfile
import io

# --- Page Configuration ---
st.set_page_config(page_title="AI Student Dossier Generator", page_icon="🎓", layout="centered")

# --- HELPER: Radar Chart ---
def create_radar_chart(student_stats, class_avgs, student_id):
    labels = ['Attendance', 'Mid Exams', 'Quizzes', 'Assignments', 'Labs']
    student_values = [
        student_stats['Attendance_%'], student_stats['Mid_Exam_Marks'] * 2, 
        student_stats['Quiz_Avg'], student_stats['Assignment_Avg'], student_stats['Lab_Scores']
    ]
    class_values = [
        class_avgs['Attendance_%'], class_avgs['Mid_Exam_Marks'] * 2, 
        class_avgs['Quiz_Avg'], class_avgs['Assignment_Avg'], class_avgs['Lab_Scores']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    student_values += student_values[:1]
    class_values += class_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, class_values, color='gray', linewidth=1, linestyle='solid', label='Class Average')
    ax.fill(angles, class_values, color='gray', alpha=0.1)
    ax.plot(angles, student_values, color='blue', linewidth=2, linestyle='solid', label=f'Student {student_id}')
    ax.fill(angles, student_values, color='blue', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    chart_path = f"radar_{student_id}.png"
    plt.savefig(chart_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig) 
    return chart_path

# --- HELPER: Pie Chart ---
def create_class_summary_chart(risk_counts):
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(risk_counts.keys())
    sizes = list(risk_counts.values())
    colors = []
    for label in labels:
        if 'HIGH' in label: colors.append('#ff9999')
        elif 'MEDIUM' in label: colors.append('#ffcc99')
        else: colors.append('#99ff99')
        
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal') 
    
    chart_path = "class_summary_pie.png"
    plt.savefig(chart_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    return chart_path

# --- MAIN DASHBOARD UI ---
st.title("🎓 AI Student Performance Dashboard")
st.markdown("Upload a class dataset to instantly generate predictive AI dossiers and intervention reports.")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload Student Data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Button to trigger generation
    if st.button("🚀 Generate AI Reports"):
        
        with st.spinner("Loading AI Models..."):
            try:
                gpa_model = joblib.load('gpa_model.pkl')
                risk_model = joblib.load('risk_model.pkl')
                scaler = joblib.load('scaler.pkl')
                expected_cols = joblib.load('model_columns.pkl')
            except Exception as e:
                st.error("❌ Missing AI Model files! Please ensure your .pkl files are in the same folder as this script.")
                st.stop()

        # Read the file
        if uploaded_file.name.endswith('.csv'):
            new_df = pd.read_csv(uploaded_file, header=1)
        else:
            new_df = pd.read_excel(uploaded_file, sheet_name='Student_Dataset', header=1)
            
        class_avgs = new_df.mean(numeric_only=True)
        
        # Preprocess
        X_new = new_df.drop(columns=['Student_ID', 'Final_GPA', 'Pass_Fail', 'Risk_Level'], errors='ignore')
        X_new_encoded = pd.get_dummies(X_new, columns=['Gender', 'Department', 'Parents_Education', 'Family_Income', 'Internet_Available'], drop_first=True)
        X_new_aligned = X_new_encoded.reindex(columns=expected_cols, fill_value=0)
        X_new_scaled = pd.DataFrame(scaler.transform(X_new_aligned), columns=X_new_aligned.columns)
        
        # Predict
        predicted_gpas = gpa_model.predict(X_new_scaled)
        predicted_risks = risk_model.predict(X_new_scaled)
        
        pdf_filenames = []
        risk_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        total_gpa = 0
        total_students = len(new_df)
        
        # UI Progress Bar
        progress_text = "Generating Dossiers..."
        my_bar = st.progress(0, text=progress_text)
        
        for index, row in new_df.iterrows():
            gpa = predicted_gpas[index]
            raw_risk = predicted_risks[index]
            total_gpa += gpa
            
            if gpa < 2.0:
                risk = "HIGH (System Override: Critical GPA)"
                risk_distribution["HIGH"] += 1
            else:
                risk = raw_risk.upper()
                if "HIGH" in risk: risk_distribution["HIGH"] += 1
                elif "MEDIUM" in risk: risk_distribution["MEDIUM"] += 1
                else: risk_distribution["LOW"] += 1
                
            feedback = []
            if row['Attendance_%'] < 75: feedback.append("- Critical: Attendance is below 75%. Prioritize live lectures.")
            if row['Study_Hours_Per_Day'] < 2.5: feedback.append("- Study Habits: Daily study hours are low. Block out 3 hours daily.")
            if row['Quiz_Avg'] < 50: feedback.append("- Quizzes: Low scores indicate poor short-term retention. Review notes quickly.")
            if gpa < 2.0: feedback.append("- Academic Warning: Projected GPA is failing. See an advisor immediately.")
            if not feedback: feedback.append("- Excellent trajectory! Keep up your current habits.")

            chart_filename = create_radar_chart(row, class_avgs, row['Student_ID'])
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="AI ACADEMIC DOSSIER & ANALYSIS", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 10, txt=f"Student ID: {row['Student_ID']} | Department: {row['Department']}", ln=True, align='C')
            pdf.line(10, 30, 200, 30)
            
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 8, txt="Complete Data Profile:", ln=True)
            pdf.set_font("Arial", '', 10)
            
            cols_to_print = [c for c in new_df.columns if c not in ['Student_ID', 'Final_GPA', 'Pass_Fail', 'Risk_Level']]
            for i in range(0, len(cols_to_print), 2):
                col1 = cols_to_print[i]
                val1 = row[col1]
                text1 = f"{col1}: {val1}"
                if i + 1 < len(cols_to_print):
                    col2 = cols_to_print[i+1]
                    val2 = row[col2]
                    text2 = f"{col2}: {val2}"
                    pdf.cell(95, 6, txt=text1)
                    pdf.cell(95, 6, txt=text2, ln=True)
                else:
                    pdf.cell(95, 6, txt=text1, ln=True)
                    
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="AI Predictive Analysis:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 8, txt=f"Projected Final GPA: {gpa:.2f}", ln=True)
            
            if "HIGH" in risk: pdf.set_text_color(200, 0, 0)
            elif "MEDIUM" in risk: pdf.set_text_color(200, 150, 0)
            else: pdf.set_text_color(0, 150, 0)
            pdf.cell(200, 8, txt=f"Academic Risk Level: {risk}", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            pdf.image(chart_filename, x=110, y=100, w=90)
            
            pdf.ln(70)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Actionable Recommendations:", ln=True)
            pdf.set_font("Arial", '', 12)
            for item in feedback:
                pdf.multi_cell(0, 8, txt=item)
                
            pdf_filename = f"Report_{row['Student_ID']}.pdf"
            pdf.output(pdf_filename)
            pdf_filenames.append(pdf_filename)
            os.remove(chart_filename)
            
            # Update Progress Bar
            progress_percent = int(((index + 1) / total_students) * 100)
            my_bar.progress(progress_percent, text=f"Processing student {index + 1} of {total_students}...")

        # Master Summary
        my_bar.progress(99, text="Generating Master Class Summary...")
        summary_pdf = FPDF()
        summary_pdf.add_page()
        summary_pdf.set_font("Arial", 'B', 20)
        summary_pdf.cell(200, 15, txt="MASTER CLASS SUMMARY REPORT", ln=True, align='C')
        summary_pdf.line(10, 25, 200, 25)
        
        summary_pdf.ln(10)
        summary_pdf.set_font("Arial", 'B', 14)
        summary_pdf.cell(200, 10, txt="Class-Wide Metrics:", ln=True)
        summary_pdf.set_font("Arial", '', 12)
        summary_pdf.cell(200, 8, txt=f"Total Students Evaluated: {total_students}", ln=True)
        summary_pdf.cell(200, 8, txt=f"Average Projected Class GPA: {total_gpa/total_students:.2f}", ln=True)
        
        summary_pdf.ln(10)
        summary_pdf.set_font("Arial", 'B', 14)
        summary_pdf.cell(200, 10, txt="Risk Level Distribution:", ln=True)
        summary_pdf.set_font("Arial", '', 12)
        summary_pdf.set_text_color(200, 0, 0); summary_pdf.cell(200, 8, txt=f"High Risk Students: {risk_distribution['HIGH']}", ln=True)
        summary_pdf.set_text_color(200, 150, 0); summary_pdf.cell(200, 8, txt=f"Medium Risk Students: {risk_distribution['MEDIUM']}", ln=True)
        summary_pdf.set_text_color(0, 150, 0); summary_pdf.cell(200, 8, txt=f"Low Risk Students: {risk_distribution['LOW']}", ln=True)
        summary_pdf.set_text_color(0, 0, 0)
        
        pie_chart = create_class_summary_chart(risk_distribution)
        summary_pdf.image(pie_chart, x=55, y=100, w=100)
        
        summary_filename = "00_MASTER_CLASS_SUMMARY.pdf"
        summary_pdf.output(summary_filename)
        pdf_filenames.append(summary_filename)
        os.remove(pie_chart)
            
        my_bar.progress(100, text="Zipping files...")
        
    zip_name = "Class_Performance_Dossiers.zip"
        with zipfile.ZipFile(zip_name, "w") as zip_file:
            for pdf_file in pdf_filenames:
                zip_file.write(pdf_file)
                os.remove(pdf_file) # Clean up the individual PDFs
                
        st.success("✅ All reports generated successfully!")
        st.balloons() # Fun animation for your presentation!
        
        # --- NEW: Tell Streamlit to remember the file exists! ---
        st.session_state['file_ready'] = True

    # --- NEW: Move Download Button OUTSIDE the Generate block ---
    if st.session_state.get('file_ready', False):
        with open("Class_Performance_Dossiers.zip", "rb") as fp:
            st.download_button(
                label="📥 Download Class Dossiers (ZIP)",
                data=fp,
                file_name="Class_Performance_Dossiers.zip",
                mime="application/zip"
            )
