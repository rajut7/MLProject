# Deploying Machine Learning Model with FASTAPI

## Overview

This project aims to develop a machine learning model for predicting an individual's annual income level based on a given set of attributes. The project includes data cleaning, model training, API creation using FastAPI, and deployment to a cloud application platform. 

## Project Structure

- `data` folder: Contains the dataset file `census.csv`.
- `src` folder: Includes the machine learning model code and unit tests, the API code implemented using FastAPI and unit tests,scripts for data cleaning, model training, and API testing.

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository: `git clone <repository-url>`
2. Create a virtual environment: `python -m venv env`
3. Activate the virtual environment:
   - For Windows: `env\Scripts\activate`
   - For Unix or Linux: `source env/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`
5. Download the dataset: [census.csv](data/census.csv)
6. Open the dataset in pandas to explore its contents.
7. Clean the dataset by removing all spaces using your favorite text editor.
8. Start working on the project!

## Model Training

- The machine learning model code found in the `src/train.py`
- Train the model using the clean dataset and save the trained model as a pickle file.
- Write unit tests for at least 3 functions in the model code to ensure its correctness.
- Evaluate the model's performance on slices of the data, focusing on categorical features.

## API Creation

- The API code is implemented using FastAPI found in `src/app.py`
- The API provides a GET endpoint that returns a welcome message.
- It also includes a POST endpoint for model inference.
- Type hinting is used throughout the API code.
- A Pydantic model is used to ingest the request body for the POST endpoint, and an example is provided.
- Ensure that the API handles column names with hyphens properly without modifying the original column names in the CSV.

## API Testing

- Write unit tests to verify the functionality of the API.
- Include tests for the GET and POST endpoints, with one test for each prediction.
- Use the `sanitycheck.py` script to run a sanity check on your test cases and ensure their correctness.

## API Deployment to a Cloud Application Platform

- Create an account on a cloud application platform like Heroku .
- Create a new app and connect it to your GitHub repository.
- Configure automatic deployments that only trigger if the continuous integration (CI) passes.
- Ensure that the paths in the deployment environment match those used in your local environment.
- Use the `requests` module to perform a POST request on your live API and verify its functionality.


## Conclusion

This project aims to provide a machine learning model for predicting an individual's annual income level, along with a robust API for accessing the model's predictions. By following the instructions and guidelines provided in this documentation, you can successfully run, test, and deploy the project components.

For any further assistance or clarification, please refer to the project's GitHub repository or reach out to the project maintainers. Happy coding!