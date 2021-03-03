```
sudo dd if=/dev/zero of=/swapfile bs=64M count=16
sudo mkswap /swapfile
sudo swapon /swapfile

sudo swapoff /swapfile
sudo rm /swapfile
docker build . -t allinone:1.0

```
