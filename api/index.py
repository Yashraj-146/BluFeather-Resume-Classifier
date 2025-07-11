from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import torch
import pdfplumber
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shutil
import os

app = FastAPI()

# Load trained model and tokenizer
MODEL_PATH = os.environ.get("MODEL_PATH", "/Users/yashraj146/Documents/resume_classifier/resume_classifier_colab")
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("mps")
try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
except Exception as e:
    print(f"Model load failed: {e}")

# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Function to predict resume match score
def predict_resume_match(resume_text, job_description):
    """Ensures that the model actually uses the uploaded resume and new job description dynamically."""
    text = resume_text + " [SEP] " + job_description  # Dynamically adding new job description
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    
    score = torch.softmax(output.logits, dim=1)[0][1].item()
    label = "Good Match" if score > 0.75 else "Not a Match"
    
    return score, label  # Ensures the score is based on the new job description

# FastAPI Endpoint for Uploading Resume & Entering Job Description
@app.post("/upload_resume/", response_class=HTMLResponse)
async def upload_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF resume
    resume_text = extract_text_from_pdf(file_location)
    if not resume_text:
        return HTMLResponse(content="<h3>Error: Could not extract text from the resume.</h3>", status_code=400)

    # 🔥 Ensuring the newly entered job description is actually used
    score, label = predict_resume_match(resume_text, job_description)

    # Return result as an HTML page
    return f"""
    <html>
    <head>
        <title>Match Result</title>
    </head>
    <body>
        <h2>Resume Matching Result</h2>
        <p><strong>Resume Filename:</strong> {file.filename}</p>
        <p><strong>Entered Job Description:</strong> {job_description}</p>
        <p><strong>Match Score:</strong> {score:.2f}</p>
        <p><strong>Result:</strong> {label}</p>
        <br>
        <a href="/">Upload Another Resume</a>
    </body>
    </html>
    """

# Simple HTML Frontend for Uploading PDF Resume & Entering Job Description
@app.get("/", response_class=HTMLResponse)
def main():
    return """
    <html>
    <head>
        <title>Resume Match</title>
    </head>
    <body>
        <h2>Upload Resume & Enter Job Description</h2>
        <form action="/upload_resume/" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".pdf" required><br><br>
            <label for="job_description">Enter Job Description:</label><br>
            <textarea name="job_description" rows="4" cols="50" required></textarea><br><br>
            <input type="submit" value="Upload & Match">
        </form>
    </body>
    </html>
    """
