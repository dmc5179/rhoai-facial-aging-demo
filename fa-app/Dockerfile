# Start with a lean, official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# This environment variable tells the 'transformers' library to use /app/.cache
# instead of the default non-writable system directory.
ENV HF_HOME=/app/.cache

# Copy the requirements file into the container
# This is done first to leverage Docker's layer caching.
# The dependencies layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY app.py .

# Create the .cache directory and set correct permissions for OpenShift.
# This ensures the directory is owned by the root group (GID 0) and is writable
# by any user belonging to that group.
RUN mkdir -p /app/.cache && \
    chgrp -R 0 /app && \
    chmod -R g+w /app

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run your app when the container starts
# The server.address flag is crucial for making the app accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
