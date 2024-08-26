# This is the base image im using for my project
# 3.9 because some libraries in project requires python > 3.8
FROM python:3.9-slim-buster

# Defining my working directory
WORKDIR /app

# copying model's pkl file to the working directory
COPY models/model_recall.pkl /app

# copying requirements file to the working directory
COPY ./requirements.txt /app

# Once copied, it will run the requirements file from the working directory
RUN pip install -r requirements.txt

# copying everything from current to the docker's current environment
COPY . .

# exposing port
EXPOSE 5000

# defining flask environment
# ENV FLASK_APP = app.py

ENTRYPOINT [ "python" ]

# passing commands which will be used by docker to run our application
CMD ["app.py"]