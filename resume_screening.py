from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read resume from PDF
reader = PdfReader("resume.pdf")

resume = ""

for page in reader.pages:
    resume += page.extract_text()

# Read job description
with open("job_description.txt", "r") as file:
    job_description = file.read()

# Convert text into numerical features
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform([resume, job_description])

# Calculate similarity
similarity = cosine_similarity(vectors[0], vectors[1])

# Convert similarity into percentage
match_score = similarity[0][0] * 100

# Skills list
skills = [
    "python",
    "java",
    "sql",
    "machine learning",
    "natural language processing",
    "scikit-learn",
    "tensorflow",
    "pandas",
    "numpy",
    "nltk",
    "git",
    "docker",
    "fastapi"
]

# Convert text to lowercase
resume_lower = resume.lower()
job_lower = job_description.lower()

# Find matching and missing skills
matching_skills = []
missing_skills = []

for skill in skills:
    if skill in resume_lower and skill in job_lower:
        matching_skills.append(skill)
    elif skill in job_lower and skill not in resume_lower:
        missing_skills.append(skill)

# Final output
print("\n======================================")
print("       AI RESUME SCREENING SYSTEM")
print("======================================")

print("\nResume Match Score:", round(match_score, 2), "%")

print("\nMatching Skills:")
for skill in matching_skills:
    print("✓", skill.title())

print("\nMissing Skills:")
for skill in missing_skills:
    print("✗", skill.title())

print("\n======================================")
print("          SCREENING COMPLETED")
print("======================================")