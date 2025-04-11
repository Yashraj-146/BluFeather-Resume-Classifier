import pandas as pd

# Load the dataset
file_path = '/Users/yashraj146/Documents/resume_classifier/resume_data.csv'
df = pd.read_csv(file_path)

# Select required columns
required_columns = [
    'career_objective', 'skills', 'educational_institution_name', 'degree_names', 
    'passing_years', 'major_field_of_studies', 'professional_company_names', 'start_dates', 'end_dates', 
    'positions', 'responsibilities', 'languages', 'certification_providers', 
    'certification_skills', 'job_position_name', 'educationaL_requirements', 
    'experiencere_requirement', 'age_requirement', 'skills_required', 'matched_score'
]
df = df[required_columns]

# Add a new column 'label' based on 'matched_score'
df['label'] = df['matched_score'].apply(lambda x: 1 if x > 0.75 else 0)

# Save the modified dataset
output_file_path = '/Users/yashraj146/Documents/resume_classifier/ideal_resume_data.csv'
df.to_csv(output_file_path, index=False)