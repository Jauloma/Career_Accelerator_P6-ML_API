docker build . -t main: latest
docker images
docker run -p 8080:7860 --name main image_id
docker ps
