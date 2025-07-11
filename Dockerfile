# Use a minimal base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port that will be used
EXPOSE 7860

# Run FastAPI app via uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
