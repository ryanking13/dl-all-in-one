
```
cat ~/TOKEN.txt | docker login https://docker.pkg.github.com -u USERNAME --password-stdin
DOCKER_BUILDKIT=1 docker build . -t allinone
docker tag allinone docker.pkg.github.com/ryanking13/dl-all-in-one/allinone:VERSION
docker push docker.pkg.github.com/ryanking13/dl-all-in-one/allinone:VERSION
```