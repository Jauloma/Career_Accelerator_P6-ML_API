FROM python:3.8
WORKDIR /app
COPY ./requirements.txt ./
RUN pip install --timeout=1200 -r requirements.txt
COPY . .


CMD ["python", "/app/src/Sepsis_App/main.py"]