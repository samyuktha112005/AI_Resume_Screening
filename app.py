import streamlit as st
from io import BytesIO
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# -------------------- TITLE --------------------

st.title("📄 AI Resume Screening System")
st.write(
    "Analyze a resume against a job description using "
    "NLP, TF-IDF and skill-based matching."
)


# -------------------- HELPER FUNCTION --------------------

def contains_skill(text, skill):
    """
    Check whether a skill exists as a complete word/phrase.
    This prevents incorrect matches such as Java inside JavaScript.
    """
    pattern = r"\b" + re.escape(skill) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def contains_any_skill(text, aliases):
    """
    Check whether any alias of a skill exists in the text.
    """
    for alias in aliases:
        if contains_skill(text, alias):
            return True
    return False


# -------------------- INPUT SECTION --------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")

    resume_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )

with col2:
    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste the complete job description..."
    )


# -------------------- SCREEN BUTTON --------------------

if st.button("🔍 Screen Resume", use_container_width=True):

    # Check resume
    if resume_file is None:
        st.error("Please upload a resume PDF.")
        st.stop()

    # Check job description
    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()


    # -------------------- PDF EXTRACTION --------------------

    try:
        pdf_bytes = resume_file.read()

        reader = PdfReader(BytesIO(pdf_bytes))

        resume = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                resume += text + " "

    except Exception as e:
        st.error(
            "Unable to read the uploaded PDF. "
            "Please upload a valid, non-password-protected PDF."
        )
        st.stop()


    # Check extracted text
    if not resume.strip():
        st.error(
            "No readable text was found in the PDF. "
            "Please upload a text-based resume PDF."
        )
        st.stop()


    # -------------------- TEXT NORMALIZATION --------------------

    resume_clean = resume.lower()
    job_clean = job_description.lower()


    # -------------------- TF-IDF SIMILARITY --------------------

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            [resume_clean, job_clean]
        )

        similarity = cosine_similarity(
            vectors[0],
            vectors[1]
        )

        text_match_score = similarity[0][0] * 100

    except Exception:
        st.error(
            "Unable to calculate text similarity. "
            "Please provide a more detailed resume and job description."
        )
        st.stop()


    # -------------------- SKILLS --------------------

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


    # Skill aliases
    skill_aliases = {
        "python": ["python"],
        "java": ["java"],
        "sql": ["sql"],
        "machine learning": [
            "machine learning",
            "ml"
        ],
        "natural language processing": [
            "natural language processing",
            "nlp"
        ],
        "scikit-learn": [
            "scikit-learn",
            "sklearn"
        ],
        "tensorflow": [
            "tensorflow"
        ],
        "pandas": [
            "pandas"
        ],
        "numpy": [
            "numpy"
        ],
        "nltk": [
            "nltk"
        ],
        "git": [
            "git"
        ],
        "docker": [
            "docker"
        ],
        "fastapi": [
            "fastapi"
        ]
    }


    # -------------------- REQUIRED SKILLS --------------------

    required_skills = []

    for skill in skills:

        aliases = skill_aliases.get(
            skill,
            [skill]
        )

        if contains_any_skill(
            job_clean,
            aliases
        ):
            required_skills.append(skill)


    # -------------------- MATCHING SKILLS --------------------

    matching_skills = []

    missing_skills = []

    for skill in required_skills:

        aliases = skill_aliases.get(
            skill,
            [skill]
        )

        if contains_any_skill(
            resume_clean,
            aliases
        ):
            matching_skills.append(skill)
        else:
            missing_skills.append(skill)


    # -------------------- SKILL SCORE --------------------

    if required_skills:

        skill_match_score = (
            len(matching_skills)
            / len(required_skills)
        ) * 100

    else:

        skill_match_score = 0


    # -------------------- OVERALL SCORE --------------------

    # Application-defined weighted score:
    # 60% text similarity + 40% skill matching

    if required_skills:

        overall_score = (
            (text_match_score * 0.60)
            + (skill_match_score * 0.40)
        )

    else:

        overall_score = text_match_score


    # -------------------- SCREENING RESULT --------------------

    if overall_score >= 80:

        screening_result = "Strong Match"
        result_message = (
            "The resume shows a strong alignment "
            "with the job description."
        )

    elif overall_score >= 60:

        screening_result = "Moderate Match"
        result_message = (
            "The resume shows a moderate alignment "
            "with the job description."
        )

    else:

        screening_result = "Low Match"
        result_message = (
            "The resume has limited alignment "
            "with the job description."
        )


    # -------------------- RESULTS --------------------

    st.success("✅ Resume screened successfully!")

    st.divider()

    st.subheader("📊 Screening Results")


    # Metrics

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Text Match Score",
            f"{text_match_score:.2f}%"
        )

    with metric2:
        st.metric(
            "Skill Match Score",
            f"{skill_match_score:.2f}%"
        )

    with metric3:
        st.metric(
            "Overall Match Score",
            f"{overall_score:.2f}%"
        )


    # -------------------- RESULT MESSAGE --------------------

    st.subheader("🎯 Screening Result")

    if screening_result == "Strong Match":

        st.success(
            f"🟢 {screening_result}\n\n"
            f"{result_message}"
        )

    elif screening_result == "Moderate Match":

        st.warning(
            f"🟡 {screening_result}\n\n"
            f"{result_message}"
        )

    else:

        st.error(
            f"🔴 {screening_result}\n\n"
            f"{result_message}"
        )


    # -------------------- REQUIRED SKILLS --------------------

    st.subheader("🧠 Required Skills Detected")

    if required_skills:

        st.write(
            f"The job description contains "
            f"**{len(required_skills)}** predefined skills."
        )

        st.write(
            ", ".join(
                skill.title()
                for skill in required_skills
            )
        )

    else:

        st.info(
            "No predefined skills were detected "
            "in the job description."
        )


    # -------------------- MATCHING / MISSING --------------------

    col3, col4 = st.columns(2)


    with col3:

        st.subheader("✅ Matching Skills")

        if matching_skills:

            for skill in matching_skills:
                st.success(skill.title())

        else:

            st.write("No matching skills found.")


    with col4:

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:
                st.error(skill.title())

        else:

            st.success(
                "No missing skills found."
            )


    # -------------------- EXPLANATION --------------------

    st.divider()

    st.subheader("ℹ️ How the Score Works")

    st.write(
        "The **Text Match Score** is calculated using "
        "TF-IDF vectorization and cosine similarity "
        "between the resume and job description."
    )

    st.write(
        "The **Skill Match Score** compares predefined "
        "technical skills detected in the job description "
        "with skills found in the resume."
    )

    st.write(
        "The **Overall Match Score** uses an "
        "application-defined weighting of 60% text similarity "
        "and 40% skill matching."
    )

    st.divider()

    st.success(
        "✅ Resume Screening Completed!"
    )