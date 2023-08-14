FROM python:3.8
WORKDIR /src/Sepsis_App
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY ./src/Sepsis_App/main.py .

EXPOSE 7680
CMD ["python", "/src/Sepsis_App/main.py"]