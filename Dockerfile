# Start from a Python base image
FROM python:3.13.4-slim

# Set the working directory
WORKDIR /app

# Copy only the relevant files first (for caching dependencies):
#  - pyproject.toml
#  - README.md (if needed for build)
COPY pyproject.toml README.md ./

# Upgrade pip and install build tooling
RUN pip install --no-cache-dir --upgrade pip build

# Copy the rest of your project
COPY . ./

# Build the wheel
RUN python -m build

# Install the newly built wheel from the dist directory
RUN pip install --no-cache-dir dist/*.whl

# Expose port 8000 (or whatever port your app runs on, if it does)
EXPOSE 8000

# The default command to run the server
CMD ["kolada-mcp"]
