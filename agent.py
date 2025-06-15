from utils import *
from config import *
from prompt import *
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma, FAISS

from langchain.schema import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import hub


load_dotenv()



class Agent():
    def __init__(self):
        self.vdb = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), './data/db'),
            embedding_function=get_embeddings_model()
        )

    def generic_func(self, x, query):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )
        return llm_chain.invoke(query)['text']

    def retrival_func(self, x, query):
        # 召回并过滤文档
        print("正在使用本地知识库进行检索回答...（调试使用）")
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=5)
        query_result = [doc[0].page_content for doc in documents if doc[1] > 0.7]

        # 填充提示词并总结答案
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        llm=get_llm_model()
        retrival_chain = prompt | llm
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return retrival_chain.invoke(inputs).content

    def graph_func(self, x, query):
        # 命名实体识别
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药品名称实体'),
        ]

        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = structured_output_parser(response_schemas)

        ner_prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables={'format_instructions': format_instructions},
            input_variables=['query']
        )

        ner_chain = LLMChain(
            llm=get_llm_model(),
            prompt=ner_prompt,
            verbose=os.getenv('VERBOSE')
        )

        result = ner_chain.invoke({
            'query': query
        })['text']

        ner_result = output_parser.parse(result)

        # 命名实体识别结果，填充模板
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = ner_result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
        if not graph_templates:
            return

            # 计算问题相似度，筛选最相关问题
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        # print(graph_documents_filter)

        # 执行CQL，拿到结果
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    query_result.append(f'问题：{question}\n答案：{answer_str}')
            except:
                pass
        # print(query_result)

        # 总结答案
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return graph_chain.invoke(inputs)['text']

    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm = get_llm_model()
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        llm_request_chain = LLMRequestsChain(
            llm_chain=llm_chain,
            requests_key='query_result'
        )
        inputs = {
            'query': query,
            'url': 'https://www.baidu.com/' + query.replace(' ', '+')
        }
        return llm_request_chain.invoke(inputs)['output']

    def query(self, query):
        tools = [
            Tool.from_function(
                name='generic_func',
                func=lambda x: self.generic_func(x, query),
                description='可以解答通用领域的知识，例如打招呼，问你是谁等问题',
            ),
            Tool.from_function(
                name='retrival_func',
                func=lambda x: self.retrival_func(x, query),
                description='用于回答寻医问药网相关问题',
            ),
            Tool(
                name='graph_func',
                func=lambda x: self.graph_func(x, query),
                description='用于回答疾病、症状、药物等医疗相关问题',
            ),
            Tool(
                name='search_func',
                func=self.search_func,
                description='其他工具没有正确答案时，通过搜索引擎，回答通用类问题',
            ),
        ]

        # prefix = """请用中文，尽你所能回答以下问题。您可以使用以下工具："""
        # suffix = """Begin!

        # History: {chat_history}
        # Question: {input}
        # Thought:{agent_scratchpad}"""

        # agent_prompt = ZeroShotAgent.create_prompt(
        #     tools=tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=['input', 'agent_scratchpad', 'chat_history']
        # )
        # llm_chain = LLMChain(llm=get_llm_model(), prompt=agent_prompt)
        # agent = ZeroShotAgent(llm_chain=llm_chain)

        # memory = ConversationBufferMemory(memory_key='chat_history')
        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent = agent,
        #     tools = tools,
        #     memory = memory,
        #     verbose = os.getenv('VERBOSE')
        # )
        # return agent_chain.run({'input': query})

        prompt = hub.pull('hwchase17/react-chat')
        prompt.template = '请用中文回答问题！Final Answer 必须尊重 Obversion 的结果，不能改变语义。\n\n' + prompt.template
        agent = create_react_agent(llm=get_llm_model(), tools=tools, prompt=prompt)
        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            handle_parsing_errors=True,
            verbose=os.getenv('VERBOSE')
        )
        return agent_executor.invoke({"input": query})['output']


# if __name__ == '__main__':
#     agent = Agent()
#
#     # print(agent.query('你好'))
#     print(agent.query('寻医问药网获得过哪些投资？'))
    # print(agent.query('鼻炎和感冒是并发症吗？'))
    # print(agent.query('鼻炎怎么治疗？'))
    # print(agent.query('烧橙子可以治感冒吗？'))

    # print(agent.generic_func('', '你好'))
    # print(agent.generic_func('你叫什么名字？'))

    # print(agent.retrival_func('介绍一下寻医问药网'))
    # print(agent.retrival_func('寻医问药网的客服电话是多少？'))

    # print(agent.graph_func('感冒和鼻炎是并发症吗？'))
    # print(agent.graph_func('感冒一般是由什么引起的？'))
    # print(agent.graph_func('感冒吃什么药好得快？可以吃阿莫西林吗？'))

