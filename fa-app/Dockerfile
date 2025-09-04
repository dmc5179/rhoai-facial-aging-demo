# 1. Start with a lean, official Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
# This is done first to leverage Docker's layer caching.
# The dependencies layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code into the container
COPY app.py .

# 6. Expose the port that Streamlit runs on
EXPOSE 8501

# 7. Define the command to run your app when the container starts
# The server.address flag is crucial for making the app accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
