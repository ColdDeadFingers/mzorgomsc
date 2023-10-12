import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load the employee performance dataset
performance_data = pd.read_csv('employeeraw.csv')

# Load the data
@st.cache_resource
def load_data():
    df = pd.read_csv('employeeraw.csv')  # Replace with your CSV file path
    return df

df = load_data()

# Title and Subtitle
st.title("Employee Dashboard")
st.subheader("Years of Experience vs. Technical Skills")

# Create a scatter plot
fig = px.scatter(df, x='YearsExperience', y='TechnicalSkills', color='PerformanceRating')
st.plotly_chart(fig)

# Show the data
st.subheader("Employee Data")
st.write(df)

# Load the prediction model
loaded_model = joblib.load("Employee_Perfomance_Prediction.joblib")

# Input fields for employee data
years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=30.0, step=0.1, value=0.0)
technical_skills = st.number_input("Technical Skills", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
communication_skills = st.number_input("Communication Skills", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
problem_solving = st.number_input("Problem Solving", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
leadership = st.number_input("Leadership", min_value=0.0, max_value=10.0, step=0.1, value=0.0)


# Make predictions
user_input = [years_experience, technical_skills, communication_skills, problem_solving, leadership]
predicted_performance = loaded_model.predict([user_input])

# Display predicted performance
st.write(f"Predicted Performance Rating: {predicted_performance[0]}")

# Load the course ratings dataset
ratings_data = pd.read_csv('employee_course_ratings.csv')

# Extract relevant columns
course_ratings = ratings_data[['CourseID', 'CourseName', 'Rating']]

def create_course_average_ratings(course_ratings):
    # Calculate the average rating for each course
    average_ratings = course_ratings.groupby('CourseName')['Rating'].mean().reset_index()
    # Remove duplicate course names to get unique courses
    unique_courses = average_ratings.drop_duplicates(subset='CourseName')
    # Rename the columns to 'Courses' and 'AverageRating'
    unique_courses = unique_courses.rename(columns={'CourseName': 'Courses', 'Rating': 'AverageRating'})
    return unique_courses

# Create the DataFrame
course_average_ratings = create_course_average_ratings(course_ratings)


print(course_average_ratings)

def recommend_courses_based_on_score(predicted_score, course_average_ratings):
    # Define a mapping of predicted scores to rating groups
    rating_group_mapping = {
        10: 5,
        9: 5,
        8: 4,
        7: 4,
        6: 3,
        5: 3,
        4: 2,
        3: 2,
        2: 1,
        1: 1,
    }

    # Ensure the predicted score is within the range of available rating groups
    predicted_score = int(round(predicted_score))
    if predicted_score < 1:
        predicted_score = 1
    elif predicted_score > len(rating_group_mapping):
        predicted_score = len(rating_group_mapping)

    # Map the predicted score to the corresponding rating group
    predicted_rating_group = rating_group_mapping.get(predicted_score, 0)  # Default to 0 if score not found

    
    # Filter courses based on the specified rating group
    courses_in_rating_group = course_average_ratings[course_average_ratings['AverageRating'].astype(int) == predicted_rating_group]

    # Get the unique course names in the rating group
    recommended_courses = list(set(courses_in_rating_group['Courses']))
    
    return recommended_courses

# Test the Model:

# Load the trained model
trained_model = joblib.load('Employee_Perfomance_Prediction.joblib')

# Define the feature names
feature_names = ['YearsExperience', 'TechnicalSkills', 'CommunicationSkills', 'ProblemSolving', 'Leadership']

# Input the new data with feature names
new_data = pd.DataFrame([[1.0, 2.5, 1.8, 2.2, 1.3]], columns=feature_names)

# Use the trained model to make predictions
predicted_scores = trained_model.predict(new_data)[0]

predicted_employee_score = predicted_scores  # Extract the first element of the predicted_scores array
recommended_courses = recommend_courses_based_on_score(predicted_scores, course_average_ratings)
print(f"Recommended courses for predicted score {predicted_scores}:")
for i, course in enumerate(recommended_courses):
    print(f"{i+1}. {course}")


# Input fields for employee data
years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=30.0, step=0.1, value=0.0, key="years_experience")
technical_skills = st.number_input("Technical Skills", min_value=0.0, max_value=10.0, step=0.1, value=0.0, key="technical_skills")
communication_skills = st.number_input("Communication Skills", min_value=0.0, max_value=10.0, step=0.1, value=0.0, key="communication_skills")
problem_solving = st.number_input("Problem Solving", min_value=0.0, max_value=10.0, step=0.1, value=0.0, key="problem_solving")
leadership = st.number_input("Leadership", min_value=0.0, max_value=10.0, step=0.1, value=0.0, key="leadership")


# Button to trigger recommendation
if st.button("Get Course Recommendations"):
    user_input = [years_experience, technical_skills, communication_skills, problem_solving, leadership]
    predicted_performance = loaded_model.predict([user_input])[0]
    recommended_courses = recommend_courses_based_on_score(predicted_performance, course_average_ratings)
    st.write(f"Recommended courses for predicted score {predicted_performance}:")
    for i, course in enumerate(recommended_courses):
        st.write(f"{i+1}. {course}")





