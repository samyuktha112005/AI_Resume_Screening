# AI Resume Screening System

An AI-based resume screening application that analyzes a candidate's resume against a job description using Natural Language Processing and Machine Learning techniques.

## Features

- Upload resume in PDF format
- Extract text from resume
- Compare resume with job description
- TF-IDF based text similarity
- Cosine similarity scoring
- Automatic skill matching
- Missing skill detection
- Strong, Moderate and Low match classification
- Interactive Streamlit interface

## Technologies Used

- Python
- Streamlit
- Natural Language Processing
- Scikit-learn
- TF-IDF
- Cosine Similarity
- PyPDF

## How It Works

1. Upload a resume in PDF format.
2. Enter the job description.
3. The system extracts text from the resume.
4. Resume and job description are converted into TF-IDF vectors.
5. Cosine similarity calculates the text match score.
6. Required skills are compared with the resume.
7. Matching and missing skills are displayed.
8. The system generates an overall screening result.

## Installation

```bash
pip install -r requirements.txt