FROM michalzg/keplervis_web:latest
ADD . /code
WORKDIR /code
CMD ["python", "app.py"]
