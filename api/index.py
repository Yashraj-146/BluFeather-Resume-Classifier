from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import torch
import pdfplumber
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shutil
import os

app = FastAPI()

# Load model and tokenizer from relative directory
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../resume_classifier_colab")  # relative path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
except Exception as e:
    print(f"Model load failed: {e}")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Predict resume match score
def predict_resume_match(resume_text, job_description):
    text = resume_text + " [SEP] " + job_description
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    
    score = torch.softmax(output.logits, dim=1)[0][1].item()
    label = "Good Match" if score > 0.75 else "Not a Match"
    return score, label

@app.post("/upload_resume/", response_class=HTMLResponse)
async def upload_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_location = os.path.join(temp_dir, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resume_text = extract_text_from_pdf(file_location)
    if not resume_text:
        return HTMLResponse(content="<h3>Error: Could not extract text from the resume.</h3>", status_code=400)

    score, label = predict_resume_match(resume_text, job_description)

    return f"""
    <html>
    <head><title>Match Result</title></head>
    <body>
        <h2>Resume Matching Result</h2>
        <p><strong>Resume Filename:</strong> {file.filename}</p>
        <p><strong>Entered Job Description:</strong> {job_description}</p>
        <p><strong>Match Score:</strong> {score:.2f}</p>
        <p><strong>Result:</strong> {label}</p>
        <br><a href="/">Upload Another Resume</a>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
def main():
    return """
    <html>
    <head><title>Resume Match</title></head>
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
