# Salary-Predictor-App
A Salary prediction application built with python and scikit learn library with an automated workflow (CI/CD) using GitHub Actions.

## Files
1. salary_app.py which contains the model wrapped in an API (fastapi).
2. Dockerfile which contains commands that a user could call on the command line to build/assemble an image.
3. requirement.txt file which contains the necessary dependencies for running the application.
4. docker-build.yml file is stored in the .github/workflows directory and is used for running the application in the docker container.
5. salary_model.py which contains the codes for the business logic (OPTIONAL, since the model has been wrapped in an API already as presented in salary_app.py).

## Folders/Directories
1. Notebooks consists of Jupyter notebooks on which the development codes were written.
  -Model Development(salary-model): These codes include exploratory analyses and predictive modeling.
  -API Development(salary-model-api): This refer to the codes that were used to write the API for the Machine Learning model (fastapi)
  
2. .github/workflows is the workflow directory and it contains the docker-build.yml file. 
