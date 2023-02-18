FROM tensorflow/tensorflow:2.9.3-gpu

WORKDIR /JIDENN

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "train.py"]