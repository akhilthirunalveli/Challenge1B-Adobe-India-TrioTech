# Use a slim, official Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This step requires internet, but only during the image build process.
# The final container will not need internet.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-downloaded model into the image
COPY ./model_cache /app/model_cache

# Copy the application files and data into the image
COPY main.py .
COPY challenge1b_input.json .
COPY pdfs/ ./pdfs/

RUN mkdir -p /app/output

# Command to run the application when the container starts
CMD ["python", "main.py"]
