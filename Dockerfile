# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables to prevent Python from generating .pyc files and to buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including python3-opencv
RUN apt-get update && apt-get install -y opencv-python-headless

# Create and set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
