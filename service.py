from prompt import *
from utils import *
from agent import *

from langchain.chains import LLMChain
from langchain.prompts import Prompt

class Service():
    def __init__(self):
        self.agent = Agent()

    def get_summary_message(slef, message, history):
        llm = get_llm_model()
        prompt = Prompt.from_template(SUMMARY_PROMPT_TPL)
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=os.getenv('VERBOSE'))
        chat_history = ''
        # for q, a in history[-2:]:
        #     chat_history += f'问题:{q}, 答案:{a}\n'=====
        for i in range(len(history) - 2, len(history), 2):
            if i + 1 < len(history):
                if isinstance(history[i], dict) and isinstance(history[i + 1], dict):
                    # Gradio 的格式
                    q = history[i].get("content", "")
                    a = history[i + 1].get("content", "")
                elif isinstance(history[i], (list, tuple)) and len(history[i]) == 2:
                    # 你在 service.py 中自己传的格式
                    q, a = history[i]
                else:
                    continue  # 非法格式跳过
                chat_history += f"问题:{q}, 答案:{a}\n"

        return llm_chain.invoke({'query':message, 'chat_history':chat_history})['text']

    def answer(self, message, history):
        if history:
            message = self.get_summary_message(message, history)
        return self.agent.query(message)


# if __name__ == '__main__':
#     service = Service()
#     # print(service.answer('你好', []))
#     # print(service.answer('得了鼻炎怎么办？', [
#     #     ['你好', '你好，有什么可以帮到您的吗？']
#     # ]))
#     print(service.answer('大概多长时间能治好？', [
#         ['你好', '你好，有什么可以帮到您的吗？'],
#         ['得了鼻炎怎么办？', '以考虑使用丙酸氟替卡松鼻喷雾剂、头孢克洛颗粒等药物进行治疗。'],
#     ]))
