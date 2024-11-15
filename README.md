1.下載專案
git clone https://github.com/eric-yhlu/TM_Instructor.git
cd TM_Project

2.設置環境變數
請在 programe/.env 文件中添加您的 OpenAI API 金鑰，格式如下：
OPENAI_API_KEY=your_openai_api_key

3. 建立 Docker 映像檔
執行以下命令來建構 Docker 映像檔：
複製程式碼
docker build -t my-flask-app .

4.啟動Docker容器
docker run -p 5000:5000 my-flask-app

5.使用應用程式
在瀏覽器中輸入:
http://localhost:5000
