IMG ?= registry.cn-hangzhou.aliyuncs.com/adpc/deep:1.0.0

build-push:
	docker build --platform=linux/amd64 -t ${IMG} .
	docker push ${IMG}