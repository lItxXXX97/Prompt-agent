import os
import asyncio
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai_tools import PGSearchTool

# 加载环境变量，确保API密钥等敏感信息不直接暴露在代码中
load_dotenv()

# 获取通义千问 API密钥
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("在环境变量中未找到 DASHSCOPE_API_KEY ")

# 获取百川 API密钥
baichuan_api_key = os.environ.get("baichuan_key")
if not baichuan_api_key:
    raise ValueError("在环境变量中未找到 baichuan_key ")

# 初始化PGSearchTool，用于从PostgreSQL数据库中搜索相关内容
search_tool = PGSearchTool(db_uri='postgresql://ai:ai@localhost:5532/ai', table_name='pdf_documents')

def create_agents(task_type):
    """
    创建四个不同角色的AI代理
    :param task_type: 任务类型
    :return: 包含四个Agent对象的列表
    """
    # 初始化ChatOpenAI模型
    # llm = ChatOpenAI(api_key=api_key,
    #                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #                    model="qwen-plus",
    #                    temperature=creativity)

    llm = ChatOpenAI(api_key=baichuan_api_key,
                     model="Baichuan3-Turbo-128k",
                     base_url="https://api.baichuan-ai.com/v1",
                     temperature=0.7)

    # 创建Prompt分析师代理
    prompt_analyst = Agent(
        role='Prompt Analyst',
        goal=f'分析 {task_type} 任务的现有prompts并提出改进建议',
        backstory='你是prompt工程方面的专家，在优化各种 AI 模型的提示方面拥有多年经验。',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool]
    )

    # 创建Prompt优化师代理
    prompt_optimizer = Agent(
        role='Prompt Optimizer',
        goal=f'根据分析和最佳实践优化 {task_type} 任务的prompts',
        backstory='你是一位熟练的prompt工程师，可以优化和增强prompts以提高 AI 模型性能。',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool]
    )

    # 创建Prompt测试员代理
    prompt_tester = Agent(
        role='Prompt Tester',
        goal=f'测试 {task_type} 任务的优化prompts并提供性能反馈',
        backstory='你是一位经验丰富的 AI QA 专家，专注于评估prompt的有效性。',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 创建最终优化专家代理
    ultimate_optimizer = Agent(
        role='Ultimate Optimizer',
        goal=f'为 {task_type} 任务提供最终优化的prompts',
        backstory='你是一位经验丰富的 AI prompt 专家，对细节有着敏锐的洞察力，并且深刻理解用户的需求。',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return [prompt_analyst, prompt_optimizer, prompt_tester, ultimate_optimizer]

def create_tasks(agents, task_type, original_prompt, user_requirements):
    """
    为每个代理创建相应的任务
    :param agents: 代理列表
    :param task_type: 任务类型
    :param original_prompt: 原始提示词
    :param user_requirements: 用户特定要求
    :return: 任务列表
    """
    tasks = [
        # Prompt分析任务
        Task(
            description=f'''分析给定的提示和数据库中的类似示例。并考虑以下方面：
            1. 清晰度和特异性
            2. 与预期任务 ({task_type}) 一致
            3. 潜在的歧义或误解
            4. 结构和格式
            原始提示: "{original_prompt}"
            使用搜索工具查找与 {task_type} 任务相关的prompts示例。
            以 Markdown 格式呈现你的分析。''',
            agent=agents[0],
            expected_output="给出Markdown 格式的示及相关示例详细解析"
        ),
        # Prompt优化任务
        Task(
            description=f'''根据分析结果，优化提示。您的优化应该：
            1. 提高清晰度并减少歧义
            2. 增强与 {task_type} 任务的一致性
            3. 结合示例提示中的有效元素
            4. 考虑以下具体要求：{user_requirements}
            原始提示：“{original_prompt}”
            使用搜索工具查找 {task_type} 提示的最佳实践。
            提供优化的提示，并说明您的更改。''',
            agent=agents[1],
            expected_output="优化prompt，解释所做的更改"
        ),
        # Prompt测试任务
        Task(
            description=f'''将优化后的提示与原始提示进行测试。您的测试应该：
            1. 比较 {task_type} 任务的潜在性能
            2. 评估清晰度和易理解性
            3. 评估误用或意外输出的可能性
            4. 验证它是否满足以下要求：{user_requirements}
            原始提示：“{original_prompt}”
            优化后的提示：[插入来自上一个任务的优化提示]
            在清晰的 Markdown 格式的报告中展示你的发现。''',
            agent=agents[2],
            expected_output="Markdown 格式的原始prompts与优化提示的综合测试报告对比"
        ),
        # 最终优化任务
        Task(
            description=f'''审查所有先前的输出并提供最终优化的提示。您的任务包括：
            1. 分析初始提示、优化提示和测试结果
            2. 确保满足所有用户要求：{user_requirements}
            3. 根据所有可用信息进一步优化提示
            4. 为 {task_type} 任务提供最终、高度优化的提示
            展示您的最终优化提示以及对改进的简要说明。''',
            agent=agents[3],
            expected_output="给出最终优化的提示，并附有改进说明"
        )
    ]
    return tasks

async def main():
    """
    主函数，处理用户输入并启动优化流程
    """
    print("欢迎使用prompts优化助手！")

    # 获取用户输入
    original_prompt = input("请输入您想要优化的prompts：")
    task_type = input("这个prompts是什么类型的任务? ")
    user_requirements = input("请输入对优化后prompts的任何具体要求: ")

    print(f"\n谢谢！开始为一个{task_type}prompts进行优化过程，具体要求如下: {user_requirements}\n")

    # 创建代理和任务
    agents = create_agents(task_type)
    tasks = create_tasks(agents, task_type, original_prompt, user_requirements)

    # 创建并启动Crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=2
    )

    # 异步执行Crew的kickoff方法
    result = await asyncio.to_thread(crew.kickoff)

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())