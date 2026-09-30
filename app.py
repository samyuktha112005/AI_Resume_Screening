import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ---------------- TITLE ----------------

st.title("📄 AI Resume Screening System")

st.write(
    "Analyze a resume against a job description using "
    "NLP, TF-IDF and skill-based matching."
)

st.divider()


# ---------------- INPUT SECTION ----------------

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
        "Enter the job description",
        height=180,
        placeholder="Paste the job description here..."
    )


st.divider()


# ---------------- SCREEN BUTTON ----------------

if st.button(
    "🔍 Screen Resume",
    use_container_width=True
):

    if resume_file is None:

        st.warning(
            "Please upload your resume."
        )

    elif job_description.strip() == "":

        st.warning(
            "Please enter a job description."
        )

    else:

        # ---------------- PDF EXTRACTION ----------------

        pdf_bytes = resume_file.read()

        reader = PdfReader(
            BytesIO(pdf_bytes)
        )

        resume = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                resume += text + " "


        # ---------------- CHECK PDF ----------------

        if resume.strip() == "":

            st.error(
                "Could not extract text from the PDF."
            )

        else:

            # ---------------- TF-IDF ----------------

            vectorizer = TfidfVectorizer()

            vectors = vectorizer.fit_transform(
                [resume, job_description]
            )


            # ---------------- COSINE SIMILARITY ----------------

            similarity = cosine_similarity(
                vectors[0],
                vectors[1]
            )

            text_match_score = (
                similarity[0][0] * 100
            )


            # ---------------- SKILLS ----------------

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


            # ---------------- NORMALIZE TEXT ----------------

            resume_lower = " ".join(
                resume.lower().split()
            )

            job_lower = " ".join(
                job_description.lower().split()
            )


            # ---------------- SKILL ALIASES ----------------

            skill_aliases = {

                "natural language processing":
                    [
                        "natural language processing",
                        "nlp"
                    ],

                "scikit-learn":
                    [
                        "scikit-learn",
                        "sklearn"
                    ],

                "machine learning":
                    [
                        "machine learning",
                        "ml"
                    ],

                "python":
                    ["python"],

                "java":
                    ["java"],

                "sql":
                    ["sql"],

                "tensorflow":
                    ["tensorflow"],

                "pandas":
                    ["pandas"],

                "numpy":
                    ["numpy"],

                "nltk":
                    ["nltk"],

                "git":
                    ["git"],

                "docker":
                    ["docker"],

                "fastapi":
                    ["fastapi"]

            }


            # ---------------- MATCHING SKILLS ----------------

            matching_skills = []

            missing_skills = []


            for skill in skills:

                aliases = skill_aliases.get(
                    skill,
                    [skill]
                )


                required = False

                for alias in aliases:

                    if alias in job_lower:

                        required = True

                        break


                if required:

                    found = False

                    for alias in aliases:

                        if alias in resume_lower:

                            found = True

                            break


                    if found:

                        matching_skills.append(
                            skill
                        )

                    else:

                        missing_skills.append(
                            skill
                        )


            # ---------------- REQUIRED SKILLS ----------------

            required_skills = []


            for skill in skills:

                aliases = skill_aliases.get(
                    skill,
                    [skill]
                )


                for alias in aliases:

                    if alias in job_lower:

                        required_skills.append(
                            skill
                        )

                        break


            # ---------------- SKILL SCORE ----------------

            if len(required_skills) > 0:

                skill_match_score = (

                    len(matching_skills)
                    / len(required_skills)

                ) * 100

            else:

                skill_match_score = 0


            # ---------------- SCREENING RESULT ----------------

            if skill_match_score >= 80:

                result = "Strong Match"

            elif skill_match_score >= 60:

                result = "Moderate Match"

            else:

                result = "Low Match"


            # ---------------- RESULTS ----------------

            st.success(
                "Resume screened successfully!"
            )


            st.subheader(
                "📊 Screening Results"
            )


            score1, score2 = st.columns(2)


            with score1:

                st.metric(
                    "Text Match Score",
                    f"{text_match_score:.2f}%"
                )


            with score2:

                st.metric(
                    "Skill Match Score",
                    f"{skill_match_score:.2f}%"
                )


            st.divider()


            # ---------------- RESULT ----------------

            st.subheader(
                "🎯 Screening Result"
            )


            if result == "Strong Match":

                st.success(
                    "🟢 Strong Match - Resume skills "
                    "align well with the job requirements."
                )

            elif result == "Moderate Match":

                st.warning(
                    "🟡 Moderate Match - Resume has "
                    "several relevant skills but some "
                    "requirements are missing."
                )

            else:

                st.error(
                    "🔴 Low Match - Resume has limited "
                    "alignment with the job requirements."
                )


            st.divider()


            # ---------------- SKILLS ----------------

            skill1, skill2 = st.columns(2)


            with skill1:

                st.subheader(
                    "✅ Matching Skills"
                )


                if matching_skills:

                    for skill in matching_skills:

                        st.write(
                            f"✅ {skill.title()}"
                        )

                else:

                    st.write(
                        "No matching skills found."
                    )


            with skill2:

                st.subheader(
                    "❌ Missing Skills"
                )


                if missing_skills:

                    for skill in missing_skills:

                        st.write(
                            f"❌ {skill.title()}"
                        )

                else:

                    st.write(
                        "No missing skills found."
                    )


            st.divider()


            st.success(
                "✅ Resume Screening Completed!"
            )