# FROM tensorflow/tensorflow:nightly-gpu
FROM tensorflow/tensorflow:2.9.1-gpu


# COPY install-packages.sh .
# RUN ./install-packages.sh

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


COPY . .

RUN mkdir /JIDENN
RUN mkdir /JIDENN/data
RUN mkdir /JIDENN/logs

CMD ["python3", "runMe.py"]