import os
from langchain_openai import ChatOpenAI
from py2neo import Graph
from dotenv import load_dotenv
load_dotenv()
import requests
from langchain.embeddings.base import Embeddings

class BaiLianEmbeddings(Embeddings):
    def __init__(self, model=None, bailian_api_key=None, endpoint=None):
        self.api_key = bailian_api_key or os.getenv("BAILIAN_API_KEY")
        self.model = model or os.getenv("BAILIAN_EMBEDDINGS_MODEL", "text-embedding-v1")
        self.api_base = endpoint or os.getenv("BAILIAN_API_BASE", "https://dashscope.aliyuncs.com")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _embed(self, texts: list):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "input": {"texts": texts}
        }

        url = f"{self.api_base}/api/v1/services/embeddings/text-embedding/text-embedding"

        # print("请求体:调试用", data)  # 调试用
        response = requests.post(url, headers=headers, json=data)
        # print("响应内容:调试用", response.text)  # 调试用
        response.raise_for_status()

        result = response.json()
        return [item["embedding"] for item in result["output"]["embeddings"]]




def get_embeddings_model():
    model_map = {
        "bailian": BaiLianEmbeddings()
    }
    return model_map.get(os.getenv("EMBEDDINGS_MODEL"))


def get_llm_model():
    llm_model = os.getenv('LLM_MODEL')
    model_name = os.getenv('OPENAI_LLM_MODEL')
    temperature = float(os.getenv('TEMPERATURE', 0))
    max_tokens = int(os.getenv('MAX_TOKENS', 2048))

    if llm_model == 'deepseek':
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base='https://api.deepseek.com/v1'
        )
    else:
        raise ValueError(f"不支持的 LLM_MODEL: {llm_model}")

def structured_output_parser(response_schemas):
    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'
    return text


def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string


def get_neo4j_conn():
    return Graph(
        os.getenv('NEO4J_URI'),
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

# if __name__ == '__main__':
#     embeddings_model = get_embeddings_model()
#     print(embeddings_model.embed_query("你要向量化的文本内容"))