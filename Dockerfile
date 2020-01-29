FROM michalzg/keplervis_web:latest

#RUN adduser -D -u 1001 -g appuser appuser
#USER appuser



ADD . /code
WORKDIR /code

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


#CMD ["pip", "-u", "app.py"]
