FROM python:3.8
WORKDIR /src/Sepsis_App/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY ./main.py .

EXPOSE 7680
CMD ["python", "/src/Sepsis_App/app/main.py"]