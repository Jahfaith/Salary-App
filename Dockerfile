FROM python:3.8
LABEL maintainer="Jahfaith Irokanulo"

COPY . /app
WORKDIR /app
RUN pip install -r requirement.txt

# command to run on container start
CMD [ "python", "salary_app.py" ]
