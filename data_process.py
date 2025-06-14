from utils import *

import os
from glob import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def doc2vec():
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    # 读取并分割文件
    dir_path = os.path.join(os.path.dirname(__file__), './data/inputs/')

    documents = []
    for file_path in glob(dir_path + '*.*'):
        loader = None
        if file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        elif file_path.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')

        if loader:
            try:
                documents += loader.load_and_split(text_splitter)
            except Exception as e:
                print(f"载入文件失败: {file_path}，错误: {e}")

    # 向量化并存储
    if documents:
        vdb = Chroma.from_documents(
            documents=documents,
            embedding=get_embeddings_model(),  # 从 utils 获取 embedding
            persist_directory=os.path.join(os.path.dirname(__file__), './data/db/')
        )
        vdb.persist()
        print("数据向量化完成并已持久化！")
    else:
        print("没有加载到任何文档，检查路径和格式。")


if __name__ == '__main__':
    doc2vec()
