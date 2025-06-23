sudo docker rm mesa_instance
sudo docker rmi mesa_image
sudo docker build . -t mesa_image
sudo docker run --name mesa_instance -p 8765:8765 -it mesa_image