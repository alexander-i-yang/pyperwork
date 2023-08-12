FROM jjanzic/docker-python3-opencv:latest
WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install poppler-utils tesseract-ocr -y
# RUN virtualenv -p python3.7 /env

RUN pip install -r requirements.txt
# CMD ["python", "main.py"]
# EXPOSE 3000
