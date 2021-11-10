# 使用官方 Python 轻量级镜像
# https://hub.docker.com/_/python
FROM python:3.9

# 将本地代码拷贝到容器内
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# 安装依赖
RUN pip install Flask gunicorn
RUN pip install torch==1.9.1 torchvision==0.10.1
RUN pip install numpy==1.21.3
RUN pip install git+https://github.com/rwightman/pytorch-image-models

# 启动 Web 服务
# 这里我们使用了 gunicorn 作为 Server，1 个 worker 和 8 个线程
# 如果您的容器实例拥有多个 CPU 核心，我们推荐您把线程数设置为与 CPU 核心数一致
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app
