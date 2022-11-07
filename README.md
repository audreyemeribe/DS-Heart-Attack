# DS-Heart-Attack
A machine learning app created using flask and deployed on Heroku

**About the repository Structure :**

- Project consist app.py script which is used to run the application. It contains a FLASK API that gets input from the user and computes a predicted value based on the model.
- xgbpredict.py contains code to build and train the Machine learning model.
- templates folder contains two files main.html and result.html which describes the structure of the app and the way the web application works. These files are connected with Python via Flask framework.
- static folder contains file style.css which adds some styling and enhances the look of the application.

**Installation**

The Code is written in Python 3.8

To install the required packages and libraries, run this command in the project directory after cloning the repository:

```bash
$ pip install -r requirements.txt 
```
- If you have some of the applications in requirements.txt already installed, you can create a virtual environment. Go to your project's directory and run:

```bash
python -m venv env
```
to create the virtual environment and activate the environment use:

```bash
.\env\Scripts\activate
```
then proceed with installing the applications in requirements.txt to your virtual environment


**To clone the repository:**

git clone https://github.com/audreyemeribe/Heart-Attack-Prediction-Deployment.git

**Run**

To Run the Application

```bash
python xgb_app.py
```
