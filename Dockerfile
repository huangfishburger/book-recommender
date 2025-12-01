# 使用官方 Python 基礎映像
FROM python:3.10-slim

# 設定容器內的工作目錄
WORKDIR /

# 複製 requirements.txt 進容器
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製整個專案到容器
COPY . .

# 預設進入 bash 或 python
CMD ["bash"]
