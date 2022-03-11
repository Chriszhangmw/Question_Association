# Question_Association
question association for chatbot

1. 先执行train.py,根据自己数据的格式修改load函数，保存向量，w,b

2. 执行train_faiss.py，将第一部得到的向量保存在faiss中，并写到一个index文件里

3. 执行model.py，先加载第2步得到的index文件，再解析输入的text，计算，最后从faiss中检索答案