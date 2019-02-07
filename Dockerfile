FROM michalzg/keplervis_web:latest

RUN adduser -D -u 1001 -g appuser appuser
USER appuser

ADD . /code
WORKDIR /code

CMD ["python", "-u", "app.py"]
