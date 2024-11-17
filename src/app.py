import math
import streamlit as st
import ast
import pickle

from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit
from radon.raw import analyze

from radon.metrics import h_visit


import pandas as pd
import matplotlib.pyplot as plt

  
# Load the pre-trained model

software_defect_prediction_model = None

#software_defect_prediction_model = joblib.load('../models/ml_model.pkl')

with open('src/ml_model.pkl', 'rb') as file:
    software_defect_prediction_model = pickle.load(file)


# Title and description
st.title("Python Code Quality and Metric Analyzer")
st.write("Upload a Python file or paste code to analyze its quality, complexity, and other important metrics.")

# Code input options
uploaded_file = st.file_uploader("Upload Python file", type="py")
code_input = st.text_area("Or paste your Python code here:", height=200, placeholder="print(\"Hello world\"!)")

# Button layout
col1, col2, col3, col4 = st.columns([2, 3.5, 2.5, 2])
analyze_btn = col1.button("Analyze", type='primary')
halstead_btn = col2.button("Get Halstead Metrics", type='primary')
cocomo_btn = col3.button("Run COCOMO", type='primary')
predict_btn = col4.button("Predict Defects", type='primary')
clear_btn = st.button("Clear Results", use_container_width=True) if analyze_btn or halstead_btn or cocomo_btn or predict_btn else None

# Initialize code variable
code = None

# Clear results and rerun the app
if clear_btn:
    st.rerun()


def is_python(code):
    """
    Check if the provided text is Python code by parsing and examining it.
    """
    # Remove any non-code sections (e.g., lines that are plain text)
    lines = code.splitlines()
    filtered_lines = [
        line for line in lines if line.strip() and not line.strip().startswith("#")
    ]
    code_to_check = "\n".join(filtered_lines)

    # AST parsing: Try parsing the filtered code
    try:
        ast.parse(code_to_check)
        return True
    except (SyntaxError, ValueError):
        # SyntaxError or ValueError in AST parsing means it's not valid Python
        return False 

def plot_code_composition(df):
    """
    Function to plot the composition of the code: LLOC, SLOC, LOC, Comments, Blank Lines, etc.
    """
    try:
        # Extract relevant values
        lloc = df[df['Metric'] == 'Logical lines of Code']['Value'].values[0]
        sloc = df[df['Metric'] == 'Source lines of Code']['Value'].values[0]
        loc = df[df['Metric'] == 'Lines of Code']['Value'].values[0]
        comments = df[df['Metric'] == 'Comments']['Value'].values[0]
        blank = df[df['Metric'] == 'Blank Lines']['Value'].values[0]
        multi_comments = df[df['Metric'] == 'Multi-line Comments']['Value'].values[0]

        # Prepare data for the pie chart
        labels = ['Logical Lines of Code (LLOC)', 'Source Lines of Code (SLOC)', 
                  'Lines of Code (LOC)', 'Comments', 'Blank Lines', 'Multi-line Comments']
        values = [lloc, sloc, loc, comments, blank, multi_comments]


        # Filter out labels and values where the value is 0
        filtered_labels = [label for label, value in zip(labels, values) if value > 0]
        filtered_values = [value for value in values if value > 0]
        
        explode = [0.03 if value != 0 else 0 for value in values]
        
         # Explode non-zero values
        explode = [0.03 if value > 0 else 0 for value in filtered_values]

        # Plotting the pie chart
        fig, ax = plt.subplots()
        ax.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', startangle=90, 
               colors=plt.cm.Paired.colors, explode=explode)
        ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
        ax.set_title('Composition of Code')

        # Display the plot in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error plotting code composition: {str(e)}")

def plot_halstead_metrics(df):
    
    try:
        halstead_metrics = [
            'h1 (Unique Operators)', 'h2 (Unique Operands)', 'N1 (Total Operators)', 'N2 (Total Operands)',
            'Vocabulary (V)', 'Length (L)', 'Calculated Length', 'Volume (V)', 'Difficulty (D)', 
            'Effort (E)', 'Time (T)', 'Bugs (B)'
        ]
        values = df['Value'].values

        # Plotting the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(halstead_metrics, values, color='skyblue')
        ax.set_xlabel('Value')
        ax.set_title('Halstead Metrics')

        # Display the plot
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting Halstead Metrics: {str(e)}")


def generate_analysis_report(code):
    """
    Generate a report with various code metrics and return it as a DataFrame.
    """
    # Initialize a list to store the metrics
    report = []
    
    # Cyclomatic Complexity
    try:
        complexity_results = cc_visit(code)
        cyclomatic_complexity = sum(block.complexity for block in complexity_results)
        report.append({'Metric': 'Cyclomatic Complexity', 'Value': cyclomatic_complexity})
        complexity_rank = cc_rank(cyclomatic_complexity)
        report.append({'Metric': 'Complexity Rank', 'Value': complexity_rank})
    except Exception as e:
        report.append({'Metric': 'Cyclomatic Complexity', 'Value': f"Error calculating complexity: {str(e)}"})
        report.append({'Metric': 'Complexity Rank', 'Value': f"Error calculating complexity rank: {str(e)}"})
    
    # Maintainability Index
    try:
        maintainability_index = mi_visit(code, multi=True)
        report.append({'Metric': 'Maintainability Index', 'Value': maintainability_index})
    except Exception as e:
        report.append({'Metric': 'Maintainability Index', 'Value': f"Error calculating maintainability index: {str(e)}"})
    
    # Lines of Code (LOC)
    try:
        loc_data = analyze(code)
        report.append({'Metric': 'Lines of Code', 'Value': loc_data.loc})
        report.append({'Metric': 'Source lines of Code', 'Value': loc_data.sloc})
        report.append({'Metric': 'Logical lines of Code', 'Value': loc_data.lloc})
        report.append({'Metric': 'Comments', 'Value': loc_data.comments})
        report.append({'Metric': 'Multi-line Comments', 'Value': loc_data.multi})
        report.append({'Metric': 'Blank Lines', 'Value': loc_data.blank})
    except Exception as e:
        report.append({'Metric': 'Lines of Code', 'Value': f"Error calculating LOC: {str(e)}"})
        report.append({'Metric': 'Source lines of Code', 'Value': f"Error calculating Source LOC: {str(e)}"})
        report.append({'Metric': 'Logical lines of Code', 'Value': f"Error calculating Logical LOC: {str(e)}"})
        report.append({'Metric': 'Comments', 'Value': f"Error calculating Comments: {str(e)}"})
        report.append({'Metric': 'Multi-line Comments', 'Value': f"Error calculating Multi-line Comments: {str(e)}"})
        report.append({'Metric': 'Blank Lines', 'Value': f"Error calculating Blank Lines: {str(e)}"})
    
    # Convert the report list into a DataFrame
    df_report = pd.DataFrame(report)
    
    return df_report

def generate_halstead_report(code):
    halstead_report, _ = h_visit(code) 
    
    # Create a dictionary with metrics
    metrics = {
        "Metric": ['h1 (Unique Operators)', 'h2 (Unique Operands)', 'N1 (Total Operators)', 'N2 (Total Operands)', 
                   'Vocabulary (V)', 'Length (L)', 'Calculated Length', 'Volume (V)', 'Difficulty (D)', 
                   'Effort (E)', 'Time (T)', 'Bugs (B)'],
        "Value": [
            halstead_report.h1, halstead_report.h2, halstead_report.N1, halstead_report.N2, halstead_report.vocabulary,
            halstead_report.length, halstead_report.calculated_length, halstead_report.volume,
            halstead_report.difficulty, halstead_report.effort, halstead_report.time, halstead_report.bugs
        ]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    
    return df

# COCOMO Functionality
def cocomo_effort(kloc, project_type='Organic'):
    if project_type == 'Organic':
        a, b, c, d = 2.4, 1.05, 2.5, 0.38
    elif project_type == 'Semi-Detached':
        a, b, c, d = 3.0, 1.12, 2.5, 0.35
    elif project_type == 'Embedded':
        a, b, c, d = 3.6, 1.20, 2.5, 0.32
    else:
        raise ValueError("Invalid project type. Choose from 'Organic', 'Semi-Detached', or 'Embedded'.")

    # Calculate Effort (Person-Months)
    effort = a * (math.pow(kloc,b))

    # Estimate Development Time (months) and Number of People
    time = c * (math.pow(effort, d))
    
    people = effort / time
    
    productivity = kloc / effort

    return effort, time, people, productivity

# Prediction logic
def get_defect_features(combined_report):
    """
    Extracts defect-related features from the combined analysis report.
    
    Parameters:
        combined_report (pd.DataFrame): A dataframe containing combined code metrics and their values.

    Returns:
        dict: A dictionary of features relevant to software defect prediction.
    """
    defect_features = []

    try:
        # Extract relevant metrics for defect prediction
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Blank Lines', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Comments', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Cyclomatic Complexity', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Logical lines of Code', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Difficulty (D)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Effort (E)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Bugs (B)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Length (L)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Time (T)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Volume (V)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'N2 (Total Operands)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'N1 (Total Operators)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'h2 (Unique Operands)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'h1 (Unique Operators)', 'Value'].values[0])
        defect_features.append(combined_report.loc[combined_report['Metric'] == 'Lines of Code', 'Value'].values[0])
        
        
    except Exception as e:
        print(f"Error extracting defect features: {e}")

    return defect_features

# Main logic for analysis
if analyze_btn:
    # Determine source of code input
    if uploaded_file:
        code = uploaded_file.read().decode("utf-8")
    elif code_input.strip():
        code = code_input
    else:
        st.warning("Please upload a file or paste code to analyze.")

    # Display code if available
    if code:
        
        if is_python(code):
            
             # Display the Python code with syntax highlighting
            st.subheader("Code to Analyze:")
            st.code(code, language="python")

            analysis_report = generate_analysis_report(code)
            
            st.subheader("Analysis Report")
            st.dataframe(analysis_report, width=700, height=350)     
            
            
            plot_code_composition(analysis_report)       
        else:
            st.info("The provided code is not valid Python code. Please check and try again.")

    else:
        st.info("No code to analyze. Please upload a Python file or paste your code.")
        
        
        
if halstead_btn:
    # Determine source of code input
    if uploaded_file:
        code = uploaded_file.read().decode("utf-8")
    elif code_input.strip():
        code = code_input
    else:
        st.warning("Please upload a file or paste code to analyze.")

    # Display code if available
    if code:
        
        if is_python(code):
            
             # Display the Python code with syntax highlighting
            st.subheader("Code to Analyze:")
            st.code(code, language="python", wrap_lines=True)

            halstead_report = generate_halstead_report(code)
            
            st.subheader("Halstead Metrics: ")
            st.dataframe(halstead_report, width=700, height=450)
            
            plot_halstead_metrics(halstead_report)
            
            
        else:
            st.info("The provided code is not valid Python code. Please check and try again.")

    else:
        st.info("No code to analyze for Halstead metrics. Please upload a Python file or paste your code.")


if cocomo_btn:
    if uploaded_file:
        code = uploaded_file.read().decode("utf-8")
    elif code_input.strip():
        code = code_input
    else:
        st.warning("Please upload a file or paste code to analyze.")

    if code:
        if is_python(code):
            loc_data = analyze(code)
            kloc = loc_data.loc / 1000  # Convert LOC to KLOC
            st.subheader("COCOMO Effort Estimation:")
            effort, time, people, productivity = cocomo_effort(kloc, project_type='Organic')
            
            st.write(f"Estimated Effort: {effort:.2f} Person-Months")
            st.write(f"Estimated Development Time: {time:.2f} Person-months")
            st.write(f"Estimated Number of People: {people:.2f} people")
            st.write(f"Estimated Productivity: {productivity:.2f} KLOC/PM")
        else:
            st.info("The provided code is not valid Python code. Please check and try again.")


if predict_btn:
    if uploaded_file:
        code = uploaded_file.read().decode("utf-8")
    elif code_input.strip():
        code = code_input
    else:
        st.warning("Please upload a file or paste code to analyze.")

    if code:
        if is_python(code):
            analysis_report = generate_analysis_report(code)
            halstead_report = generate_halstead_report(code)
            combined_report = pd.concat([analysis_report, halstead_report], ignore_index=True)
            print("combined report:\n", combined_report)
            
            feature_names = ['LOC_BLANK', 'LOC_COMMENTS', 'CYCLOMATIC_COMPLEXITY', 'LOC_EXECUTABLE',
                'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST',
                'HALSTEAD_LENGTH', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
                'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS',
                'NUM_UNIQUE_OPERATORS', 'LOC_TOTAL'
            ]
            
            features = get_defect_features(combined_report)
            print("Feature array: ",features)
            
            # Wrap the input in a DataFrame with the correct feature names
            input_df = pd.DataFrame([features], columns=feature_names)
            
            # Make prediction using the loaded stacked model
            prediction = software_defect_prediction_model.predict(input_df)
            
            defect_status = "Defective" if prediction[0] == 1 else "Non-Defective"
            
            confidence_score=software_defect_prediction_model.predict_proba([features])[0][prediction[0]]
            
            st.subheader("Defect Prediction Result:")
            st.write(f"The code is predicted to be **{defect_status}**, with a confidence of **{confidence_score}** %")
            