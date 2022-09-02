
# 用Docker 运行 Podracer-ElegantRL

## 在Linux系统中安装docker

### 安装docker主体
安装docker需要的前置包
```
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release
```

下载docker
```
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

安装docker
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

启动docker服务，确认安装成功
```
sudo service docker stop  //停止
sudo service docker start  //启动
sudo service docker status  //检查状态显示running即可
```

将用户加入docker组别，使其更方便使用docker命令
```
sudo usermod -aG docker <用户名>
```

验证docker是否成功安装
```
sudo docker ps
```
终端会print出以下信息，记录最近docker启动记录：
```
CONTAINER ID   IMAGE          COMMAND        CREATED        STATUS       PORTS        NAMES
...
```

### 安装docker-compose

下载并安装docker-compose
```
sudo apt-get remove docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.6.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

验证以上docker-compose安装是否成功：
```
docker-compose --version
```
终端会print出以下信息，显示docker-compose的版本：
```
Docker Compose version v2.6.0
```

### 为docker环境连接计算服务器生成ssh key
**可跳过不做**，但配置ssh key 能免去每次登陆输入用户密码的麻烦，操作流程与网上能搜到的 ssh key 配置方案相同

```
ssh-keygen //一直enter
```

## 启动docker

### 创建 docker context

创建docker context 命令：
```
docker context create gpu_64 --docker host=tcp://192.168.52.164:2375
```
参数解释：
- `gpu_64` 是用户自己定义的 docker context 名称
- `tcp://192.168.52.164` 是作为计算节点使用的远程服务器IP地址
- `tcp://192.168.52.164:2375` 的2375 是远程服务器的端口号，用户可以自己随便指定一个

更多网络的配置，可以在文件 `./Podracer/compose/docker-compose.dev.yml` 里看到
https://github.com/AI4Finance-Foundation/ElegantRL_Jiahao/blob/2b2dd2d993780c5327a11340bb2d4e7d61b7bd4c/Podracer/compose/docker-compose.dev.yml#L19-L21

### 切换 docker context

在自己的个人电脑启动的终端上，可以切换指定名字的 docker context 
```
docker context use gpu_64
```

检查切换是否成功：
```
$ docker context ls
NAME       DESCRIPTION                               DOCKER ENDPOINT               KUBERNETES ENDPOINT   ORCHESTRATOR
default    Current DOCKER_HOST based configuration   unix:///var/run/docker.sock                         swarm
gpu_64 *                                             tcp://192.168.52.164:2375
```

## 特定项目的 docker context

### 开启特定项目的 docker context
在个人电脑上，打开一个bash终端：
- LinuxOS 可以直接打开
- IOS属于Linux系列，也可以直接打开
- WindowOS需要安装 Window的Linux子系统WSL

```
cd ./compose/
docker-compose -p <自定义项目名称> -f docker-compose.dev.yml up -d --build
```
参数解释：
- `./compose/` 就是项目里面的文件夹，在 [github的Podracer/compose](https://github.com/AI4Finance-Foundation/ElegantRL_Jiahao/tree/main/Podracer/compose)
- `docker-compose.dev.yml` 文件在`compose`文件夹里，在[github的Podracer/compose/docker-compose.dev.yml](https://github.com/AI4Finance-Foundation/ElegantRL_Jiahao/blob/main/Podracer/compose/docker-compose.dev.yml)
- `-build` 参数用于重新构建services，更新旧镜像的环境

更多端口的配置，可以在文件 `./Podracer/compose/docker-compose.dev.yml` 里看到
https://github.com/AI4Finance-Foundation/ElegantRL_Jiahao/blob/2b2dd2d993780c5327a11340bb2d4e7d61b7bd4c/Podracer/compose/docker-compose.dev.yml#L22-L24

### 停止特定项目的 docker context

```
cd compose/
docker-compose -p <自定义项目名称> -f docker-compose.dev.yml down
```
