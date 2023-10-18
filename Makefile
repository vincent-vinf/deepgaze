IMG ?= registry.cn-hangzhou.aliyuncs.com/adpc/deep:1.0.0

build-push: docker-build docker-push

docker-build:
	docker build --platform=linux/amd64 -t ${IMG} .

docker-push:
	docker push ${IMG}

build-arm:
	docker build --platform=linux/arm64 -t ${IMG} .
