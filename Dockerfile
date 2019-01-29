FROM michalzg/keplervis_web:latest
ADD . /code
WORKDIR /code
CMD ["python", "-u", "app.py"]
