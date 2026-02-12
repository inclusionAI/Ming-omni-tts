'''AntGLM Chat-model data format.

格式化 AntGLM 以及各种开源模型的符号系统:
  - 确定 Chat 模型依赖的文件数据结构协议
  - 确定单轮/多轮的统一结构
  - 确定 Chat 符号系统的协议, 包括角色定义、分隔符等
  - 方便做开源模型依赖的 prompt 转换
  - 支持工具、代码、推理等支持

参考 FastChat Conversation 对象的设计思路.
Reference: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
'''

import copy
import dataclasses
import logging
import re
import uuid
from copy import deepcopy
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptStyle(IntEnum):
    '''Prompt styles.'''

    # 原始 antglm format 格式, 单轮指令没有结构, 多轮 `第1轮\n用户: xx\n机器人: xx\n`
    ANTGLM_RAW = auto()
    # Chat format 格式, 单轮多轮统一为 chat format 格式
    ANTGLM_CHAT = auto()
    # 单轮指令没有结构, 只有多轮为 chat format 格式
    ANTGLM_ONLY_MULTITURN_CHAT = auto()
    # OpenAI ChatML 格式, 包括千问
    CHATML = auto()
    # LLAMA2 格式
    LLAMA2 = auto()
    # ChatGLM 1/2 格式
    CHATGLM = auto()
    # ChatGLM3 格式
    CHATGLM3 = auto()
    # 百川格式
    BAICHUAN2 = auto()


@dataclasses.dataclass
class Chat:
    '''Chat 数据符号结构, 格式化 AntGLM 以及各种开源模型的符号系统.

    Examples:

    ```python
    >>> from antllm.data.chat_format import Chat

    >>> ### 从 json 数据结构创建 chat 对象, 并且 format 结构使用 AntGLM 原始结构
    >>> input_json = {
    ...     "messages": [
    ...         {"role": "HUMAN", "content": "讲一个笑话"},
    ...         {"role": "ASSISTANT", "content": "为什么猪不能上网？因为它们会被网上的“猪”骗！哈哈哈！"},
    ...         {"role": "HUMAN", "content": "不好笑，换个程序员的笑话"}
    ...      ],
    ... }
    >>> chat = Chat.from_json(input_json, name='antglm_raw')

    >>> ### 根据 chat 对象创建大模型训练所需 pack 数据
    >>> pack_data = chat.prompt_pack
    >>> print(pack_data)

    >>> ### 根据 chat 对象创建大模型训练所需 input, output 数据
    >>> data = chat.prompt_inout
    >>> print(data)

    >>> ### 根据 chat 对象创建大模型预测用的 prompt
    >>> prompt = chat.prompt_str
    >>> print(prompt)

    >>> ### 从大模型训练数据 {"input": "xx", "output": "xx"} 中创建 chat 对象
    >>> data = {
    ...     'input': (
    ...         '第1轮\n用户: 讲一个笑话\n机器人: 为什么猪不能上网？因为它们会被网上的“猪”骗！哈哈哈！\n'
    ...         '第2轮\n用户: 不好笑，换个程序员的笑话\n机器人:'
    ...     ),
    ...     'output': ''
    ... }
    >>> chat = Chat.from_inout(data, name='antglm_raw')

    >>> ### 从大模型 pack 训练数据创建 chat 对象列表
    >>> pack_data = {
    ...     'inputs': ['第1轮\n用户: 讲一个笑话\n机器人:', '第2轮\n用户: 不好笑，换个程序员的笑话\n机器人:', '第1轮\n用户: 写首诗\n机器人:'],
    ...     'outputs': [
    ...         '为什么猪不能上网？因为它们会被网上的“猪”骗！哈哈哈！\n',
    ...         '为什么程序员总是喜欢使用黑色主题？因为他们喜欢“黑暗模式”（Dark Mode),这样他们就可以在晚上加班时更好地隐藏自己的错误！',
    ...         '']
    ... }
    >>> chats = Chat.from_pack(pack_data, name='antglm_raw')
    >>> assert len(chats) == 2
    >>> print(chats[0])
    >>> print(chats[1])

    >>> ### 显示总交互轮数 (以用户输出多少次为轮数个数)
    >>> print(chat.turns_num)

    >>> ### 根据 chat 对象创建 json 格式化输出
    >>> data_json = chat.to_json()
    >>> print(data_json)

    >>> ### 增加轮次信息
    >>> content = (
    ...     '为什么程序员总是喜欢使用黑色主题？'
    ...     '因为他们喜欢“黑暗模式”（Dark Mode),这样他们就可以在晚上加班时更好地隐藏自己的错误！'
    ... )
    >>> chat.append_message(chat.role_assistant, content)

    >>> ### 将 chat 对象转成 OpenAI ChatCompletion 接口的入参
    >>> openai_messages = chat.to_openai_api_messages()
    >>> print(openai_messages)

    >>> ### 复制一个 chat 对象
    >>> chat_new = chat.copy()
    ```
    '''

    # 数据结构名称
    id: str = None

    # format 支持: antglm_raw, antglm_chat, chatglm1, chatglm2, llama2, qwen, baichuan2
    name: Optional[str] = None

    # Prompt 风格
    prompt_style: Optional[PromptStyle] = None

    # System Template 和 message
    system_template: str = '<role>SYSTEM</role>{}'
    system_message: str = ''

    # 角色定义
    role_human: str = 'HUMAN'
    role_assistant: str = 'ASSISTANT'
    role_observation: str = 'OBSERVATION'
    role_template: str = '<role>{}</role>'

    # 每轮符号定义
    turn_start: str = ''
    human_end: str = ''
    assistant_start: str = ''
    assistant_end: str = ''
    assistant_end_ids: Optional[List[int]] = None
    general_role_end: str = ''

    # agent 符号定义
    tool_template = '<tool>{}</tool>'
    code_template = '<code>{}</code>'
    arithemetic_templte = '<arithemetic>{}</arithemetic>'
    image_template = '<image>{}</image>'

    # All messages. Each item is (role, message).
    messages: List[Tuple[str, str]] = ()

    # messages 中用于 few-shot messages
    offset: int = 0

    # 其他 meta data
    source: Optional[str] = None
    lang: Optional[str] = None
    topic: Optional[str] = None

    # 原始 json 数据
    origin_json: Optional[dict] = None

    @property
    def support_names(self) -> Dict[str, str]:
        '''支持的数据对象名称.'''
        return {
            'antglm_raw': '原始 antglm format 格式, 单轮指令没有结构, 多轮 `第1轮\\n用户:xx\\n机器人xx\\n`',
            'antglm_chat': 'Chat format 格式, 单轮多轮统一为 chat format 格式',
            'chatglm1': 'chatglm1 format',
            'chatglm2': 'chatglm2 format',
            'llama2': 'llama2 format',
            'qwen': '千问 format',
            'baichuan2': '百川 2 format',
        }

    @classmethod
    def from_json(
        cls,
        input: dict,
        name: Optional[str] = None,
        prompt_style: Optional[PromptStyle] = None,
    ):
        '''从文件数据结构到数据对象的转换.

        Params:
            name: `Optional[str]`, 符号系统名称
                - format 支持: antglm_raw, antglm_chat, chatglm1, chatglm2, llama2, qwen, baichuan2
                - 如果指定了 format name, 使用该 name 符号系统, 否则使用 input 中 `name` 字段

            prompt_style: `Optional[PromptStyle]`, 指定 prompt 风格, 默认使用和 name 一致的风格

            input: `dict`, 文件中的 json dict 对象, 协议为:
                - 既支持 `messages` 字段, 也支持 `turns` 字段
            {
                "id": "xxx",
                "name": "antglm",
                "source": "xxx",
                "lang": "xx",
                "topic": "xx",
                "system_template": "",
                "system_message": "xx",
                "messages": [
                    {
                        "role": "HUMAN",
                        "content": "Hi"
                    },
                    {
                        "role": "ASSISTANT",
                        "content": "Hello"
                    },
                    {
                        "role": "OBSERVATION",
                        "content": "xxx"
                    },
                    {
                        "role": "ASSISTANT",
                        "content": "xxx"
                    }
                ],
                "turns": [
                    {"HUMAN": "xxx", "OBSERVATION": "xx", "ASSISTANT": "xx"}
                ]
            }

        Returns:
            `Chat` 对象
        '''
        _id = input.get('id')
        if name:
            _name = name
        else:
            _name = input.get('name')
        source = input.get('source')
        lang = input.get('lang')
        topic = input.get('topic')
        kwargs = {}
        if 'system_template' in input:
            kwargs['system_template'] = input['system_template']
        if 'system_message' in input:
            kwargs['system_message'] = input['system_message']

        # 转换成 Chat 对象
        chat = cls(
            id=_id,
            name=_name,
            prompt_style=prompt_style,
            source=source,
            lang=lang,
            topic=topic,
            origin_json=deepcopy(input),
            **kwargs,
        )
        if 'messages' in input:
            for msg in input['messages']:
                if msg['role'] == 'HUMAN':
                    role = chat.role_human
                elif msg['role'] == 'OBSERVATION':
                    role = chat.role_observation
                elif msg['role'] == 'ASSISTANT':
                    role = chat.role_assistant
                else:
                    raise ValueError(f'不支持数据集中的 role: {msg["role"]}')

                chat.append_message(role, msg['content'])

        elif 'turns' in input:
            for turn in input['turns']:
                if 'HUMAN' in turn:
                    content = turn['HUMAN']
                    chat.append_message(chat.role_human, content)
                if 'OBSERVATION' in turn:
                    content = turn['OBSERVATION']
                    chat.append_message(chat.role_observation, content)
                if 'ASSISTANT' in turn:
                    content = turn['ASSISTANT']
                    chat.append_message(chat.role_assistant, content)

        return chat

    @classmethod
    def from_pack(
        cls,
        packs: Dict[str, List[str]],
        name: str,
        prompt_style: Optional[PromptStyle] = None,
    ) -> list:
        '''根据 pack 数据创建 Chat 对象.

        Params:
            packs: `dict`, pack 样本数据
                {
                    'inputs': ['xx', 'xx'],
                    'outputs': ['xx', 'xx'],
                }

            name: `str`, 符号系统名称
            prompt_style: `Optional[PromptStyle]`, 指定 prompt 风格, 默认使用和 name 一致的风格
        '''
        chat = cls(name=name, prompt_style=prompt_style)
        packs = cls._format_packs(packs)

        sys_pattern = re.compile(chat.system_template.format(r'(.*?)'), re.DOTALL)
        turn_pattern = re.compile(chat.turn_start.format(r'(\d+)'), re.DOTALL)
        human_pattern = re.compile(chat.role_template.format(chat.role_human).strip(), re.DOTALL)
        observe_pattern = re.compile(chat.role_template.format(chat.role_observation).strip(), re.DOTALL)
        assistant_pattern = re.compile(chat.role_template.format(chat.role_assistant).strip(), re.DOTALL)

        chats = []
        for input, output in zip(packs['input'], packs['output']):
            # system message
            sys_match = sys_pattern.search(input)
            if sys_match and sys_match.group(0):
                # system 指令只在首轮, 新增 chat 对象
                if len(chat.messages) > 0:
                    chats.append(chat)
                    chat = cls(name=name, prompt_style=prompt_style)

                input = input[sys_match.end() :]
                chat.system_message = sys_match.group(1)

            # turn start
            turn_match = turn_pattern.search(input)
            if turn_match and turn_match.group(0):
                # 当出现下一个轮次开始信息, 新增 chat 对象
                if name in ['antglm', 'antglm_raw', 'chatglm2']:
                    round_start = 1
                else:
                    round_start = 0

                if all(
                    [
                        len(turn_match.groups()) > 0,
                        int(turn_match.group(1)) == round_start,
                        len(chat.messages) > 0,
                    ]
                ):
                    chats.append(chat)
                    chat = cls(name=name, prompt_style=prompt_style)

                input = input[turn_match.end() :]

            human_iter = human_pattern.finditer(input)
            observe_iter = observe_pattern.finditer(input)
            assistant_iter = assistant_pattern.finditer(input)
            human_match = next(human_iter, None)
            observe_match = next(observe_iter, None)
            assistant_match = next(assistant_iter, None)

            if not human_match and not observe_match:
                # 无 role format
                chat.append_message(chat.role_human, input)

            while human_match or observe_match:
                next_human_match = next(human_iter, None)
                next_observe_match = next(observe_iter, None)
                input = cls._append_human_observation(
                    chat,
                    input,
                    human_match=human_match,
                    next_human_match=next_human_match,
                    observe_match=observe_match,
                    next_observe_match=next_observe_match,
                    assistant_match=assistant_match,
                )

                human_match = next_human_match
                observe_match = next_observe_match
                next_human_match = next(human_iter, None)
                next_observe_match = next(observe_iter, None)

            if output:
                chat.append_message(chat.role_assistant, output)

        if chat.messages:
            chats.append(chat)

        return chats

    @classmethod
    def _append_human_observation(
        cls,
        chat,
        input: str,
        human_match: Optional[re.Match] = None,
        next_human_match: Optional[re.Match] = None,
        observe_match: Optional[re.Match] = None,
        next_observe_match: Optional[re.Match] = None,
        assistant_match: Optional[re.Match] = None,
    ) -> str:
        '''给 chat 对象增加 human/observation message.'''
        if observe_match:
            # observation 在 human 之后
            if observe_match.span()[0] > observe_match.span()[0]:
                human_str = input[observe_match.span()[1] : observe_match.span()[0]]
                observe_str = input[observe_match.span()[1] : assistant_match.span()[0]]
                chat.append_message(chat.role_human, human_str.strip())
                input_end = observe_match.span()[1]
                if observe_match.span()[0] < next_human_match.span()[0]:
                    chat.append_message(chat.role_observation, observe_str.strip())
                    input_end = assistant_match.span()[1]
            else:
                # observation 在 human 之前
                human_str = input[observe_match.span()[1] : assistant_match.span()[0]]
                observe_str = input[observe_match.span()[1] : observe_match.span()[0]]
                chat.append_message(chat.role_observation, observe_str.strip())
                input_end = observe_match.span()[1]
                if observe_match.span()[0] < next_observe_match.span()[0]:
                    chat.append_message(chat.role_human, human_str.strip())
                    input_end = assistant_match.span()[1]
        else:
            if assistant_match:
                human_str = input[human_match.span()[1] : assistant_match.span()[0]]
                input_end = assistant_match.span()[1]
            else:
                human_str = input[human_match.span()[1] :]
                input_end = len(input)
            chat.append_message(chat.role_human, human_str.strip())

        return input[input_end:]

    @classmethod
    def from_inout(
        cls,
        sample: Dict[str, str],
        name: str,
        prompt_style: Optional[PromptStyle] = None,
    ):
        '''根据单样本创建一个 Chat 对象.

        Params:
            sample: `Dict[str, str]`, input/output 数据样本
                {
                    "input": "xxx",
                    "output": "xxx",
                }

            name: `str`, 符号系统名称
            prompt_style: `Optional[PromptStyle]`, 指定 prompt 风格, 默认使用和 name 一致的风格
        '''
        chat = cls(name=name, prompt_style=prompt_style)
        input = sample['input']
        output = sample['output']

        sys_pattern = re.compile(chat.system_template.format(r'(.*?)'), re.DOTALL)
        turn_pattern = re.compile(chat.turn_start.format(r'(\d+)'), re.DOTALL)
        human_pattern = re.compile(chat.role_template.format(chat.role_human).strip(), re.DOTALL)
        observe_pattern = re.compile(chat.role_template.format(chat.role_observation).strip(), re.DOTALL)
        assistant_pattern = re.compile(chat.role_template.format(chat.role_assistant).strip(), re.DOTALL)

        # 去除轮次信息
        input = turn_pattern.sub('', input)

        # system message search
        sys_match = sys_pattern.search(input)
        if sys_match and sys_match.group(0):
            input = input[sys_match.end() :]
            chat.system_message = sys_match.group(1)

        human_iter = human_pattern.finditer(input)
        observe_iter = observe_pattern.finditer(input)
        assistant_iter = assistant_pattern.finditer(input)
        human_match = next(human_iter, None)
        observe_match = next(observe_iter, None)
        assistant_match = next(assistant_iter, None)
        next_human_match = next(human_iter, None)
        next_observe_match = next(observe_iter, None)

        while any(
            [
                human_match,
                observe_match,
                assistant_match,
            ]
        ):

            # human/observation 先后顺序可能不一样, 并且有可能有多个
            # 判断 assitant 之前是否还有 human/observation
            while any(
                [
                    human_match and human_match.span()[0] < assistant_match.span()[0],
                    observe_match and observe_match.span()[0] < assistant_match.span()[0],
                    next_human_match and next_human_match.span()[0] < assistant_match.span()[0],
                    next_observe_match and next_observe_match.span()[0] < assistant_match.span()[0],
                ]
            ):
                if not input:
                    break

                cls._append_human_observation(
                    chat,
                    input,
                    human_match=human_match,
                    next_human_match=next_human_match,
                    observe_match=observe_match,
                    next_observe_match=next_observe_match,
                    assistant_match=assistant_match,
                )

                human_match = next_human_match
                observe_match = next_observe_match
                next_human_match = next(human_iter, None)
                next_observe_match = next(observe_iter, None)

            # assistant message
            if assistant_match and assistant_match.span():
                if observe_match:
                    if observe_match.span() and observe_match.span()[0] < human_match.span()[0]:
                        assistant_str = input[assistant_match.span()[1] : observe_match.span()[0]]
                elif human_match:
                    if human_match.span():
                        assistant_str = input[assistant_match.span()[1] : human_match.span()[0]]
                else:
                    assistant_str = input[assistant_match.span()[1] :]

                if assistant_str:
                    chat.append_message(chat.role_assistant, assistant_str)

            assistant_match = next(assistant_iter, None)

        if output:
            chat.append_message(chat.role_assistant, output)

        return chat

    def __hash__(self):
        '''数据对象的 hash 函数.'''
        return hash(self.id)

    def __post_init__(self):
        '''对象初始化后的处理, 处理包括:
        - 根据数据对象名称, 支持转成其他开源数据对象的基本信息
        '''
        self.id = str(uuid.uuid4())
        if not self.messages:
            self.messages = []

        if not self.name and not self.prompt_style:
            logger.error('构造 Chat 对象至少包含以下一个入参: `name/prompt_style`.\n\n' '`name` 支持以下 format 名称:')
            logger.error('\n'.join([f'{k}: {v}' for k, v in self.support_names.items()]))
            logger.error('\n`prompt_style` 参考 antllm.data.chat_format.PromptStyle')
            raise ValueError

        if self.name == 'antglm':
            # 默认 antglm 使用原始 antglm_raw - 第1轮\n用户: xx\n机器人: xx\n
            self.name = 'antglm_raw'

        if not self.name and self.prompt_style == PromptStyle.ANTGLM_CHAT:
            logger.info(
                'Chat 对象入参没有 `name`, 默认使用 `ANTGLM_CHAT`, format:\n'
                f'role_human: {self.role_human}\n'
                f'role_assistant: {self.role_assistant}\n'
                f'role_observation: {self.role_observation}\n'
                f'role_template: {self.role_template}\n'
                f'turn_start: {self.turn_start}\n'
                f'human_end: {self.human_end}\n'
                f'assistant_start: {self.assistant_start}\n'
                f'assistant_end: {self.assistant_end}\n'
                f'assistant_end_ids: {self.assistant_end_ids}\n'
                f'general_role_end: {self.general_role_end}\n'
                f'tool_template: {self.tool_template}\n'
                f'code_template: {self.code_template}\n'
                f'arithemetic_templte: {self.arithemetic_templte}\n'
                f'image_template: {self.image_template}\n'
                f'\n入参 `name` 支持: ``'
            )
            return

        if self.name == 'antglm_raw' or self.prompt_style == PromptStyle.ANTGLM_RAW:
            self.prompt_style = PromptStyle.ANTGLM_RAW
            self.role_template = '{}'
            self.role_human = '用户: '
            self.role_assistant = '机器人: '
            self.turn_start = '第{}轮\n'
            self.general_role_end = '\n'

        if self.name in ['chatglm1', 'chatglm2'] or self.prompt_style == PromptStyle.CHATGLM:
            self.prompt_style = PromptStyle.CHATGLM
            self.role_template = '{}'
            self.role_human = '问：'
            self.role_assistant = '答：'
            self.turn_start = '[Round {}]\n'
            if self.name == 'chatglm1':
                self.general_role_end = '\n'
            else:
                self.general_role_end = '\n\n'

        elif self.name == 'chatglm3' or self.prompt_style == PromptStyle.CHATGLM3:
            self.prompt_style = PromptStyle.CHATGLM3
            self.system_template = '<|system|>\n {}'
            self.role_human = '<|user|>\n '
            self.role_assistant = '<|assistant|>\n '
            self.role_template = '{}'

        elif self.name == 'llama2' or self.prompt_style == PromptStyle.LLAMA2:
            self.prompt_style = PromptStyle.LLAMA2
            self.role_template = '{}'
            self.system_template = '[INST] <<SYS>>\n{}\n<</SYS>>\n\n'
            self.role_human = '[INST] '
            self.role_assistant = '[/INST] '
            self.human_end = ' '
            self.assistant_end = ' </s><s>'

        elif self.name == 'qwen':
            self.prompt_style = PromptStyle.CHATML
            self.role_template = '{}'
            self.system_template = '<|im_start|>system\n{}'
            if not self.system_message:
                self.system_message = 'You are a helpful assistant.'
            self.role_human = '<|im_start|>user\n'
            self.role_assistant = '<|im_start|>assistant\n'
            self.general_role_end = '<|im_end|>\n'

        elif self.name == 'baichuan':
            self.prompt_style = PromptStyle.BAICHUAN2
            self.role_template = '{}'
            self.system_template = '{}'
            self.role_human = '<token_id-195>'
            self.role_assistant = '<token_id-196>'

        if not self.system_template:
            self.system_template = '{}'

    def readable_messages(self) -> str:
        '''将 messages 输出为人类可读的字符串, 方便分析数据.'''
        pass

    @property
    def prompt_str(self) -> str:
        '''将 Chat 对象转成 prompt str, 合并 human/assitant 输出为 format 字符串.'''
        return f'{self.prompt_inout["input"]}{self.prompt_inout["output"]}'

    @classmethod
    def _format_packs(cls, packs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        '''格式化 pack 样本, 输出相同 pack inputs, outputs 个数.'''
        _packs = copy.deepcopy(packs)
        if len(_packs['input']) - 1 == len(_packs['output']):
            _packs['output'].append('')

        if len(_packs['input']) != len(_packs['output']):
            print(packs)
            raise ValueError(
                '输入 input 和 output 数量不匹配, '
                f'input num: {len(packs["input"])}, '
                f'output num: {len(packs["output"])}'
            )

        return _packs

    @property
    def prompt_inout(self) -> Dict[str, str]:
        '''将 Chat 对象转成 input prompt, output prompt 字符串.

        Returns:
            `Dict[str, str]`, 示例:
            {
                "input": "<role>SYSTEM</role>xxxx<role>HUMAN</role>你好<role>ASSISTANT</role>你好，有什么可以帮您？<role>ASSISTANT</role>",  # noqa
                "output": "你好，有什么可以帮您？"
            }
        '''
        packs = self._format_packs(self.prompt_pack)

        # 兼容逻辑
        if self.prompt_style == PromptStyle.ANTGLM_RAW:
            packs['input'] = [f'{item} ' for item in packs['input']]

        prompt_input = ''.join([f'{x}{y}' for x, y in zip(packs['input'][:-1], packs['output'][:-1])])
        prompt_input += packs['input'][-1]
        prompt_output = packs['output'][-1]

        # 兼容逻辑
        if self.prompt_style == PromptStyle.ANTGLM_RAW:
            prompt_input = prompt_input.strip()

        return {
            'input': prompt_input,
            'output': prompt_output,
        }

    @property
    def prompt_pack(self) -> Dict[str, List[str]]:
        '''将数据对象转成 pack input prompt, output prompt 字符串列表.:

        Returns:
            `Dict[str, List[str]]`, 示例:

            {
                "input": [
                    "<role>SYSTEM</role>xxxx<role>HUMAN</role>你好<role>ASSISTANT</role>",
                    "<role>HUMAN</role>讲个笑话<role>ASSISTANT</role>",
                    "<role>OBSERVATION</role>{\"weather\": \"晴\"}<role>ASSISTANT</role>"
                ],
                "output": [
                    "你好，有什么可以帮您？",
                    "笑话 1",
                    "今天天气 xxx"
                ]
            }

        '''
        inputs = []
        outputs = []

        # 最开始 system 构造
        system_prompt = ''
        if self.system_message:
            system_prompt = self.system_template.format(self.system_message)

        if system_prompt:
            ret = system_prompt + self.general_role_end
        else:
            ret = ''

        # 有些 prompt style 单轮指令没有 format
        if self.prompt_style in [
            PromptStyle.ANTGLM_RAW,
            PromptStyle.ANTGLM_ONLY_MULTITURN_CHAT,
        ]:
            if len(self.messages) <= 2:
                output = ''
                for role, message in self.messages:
                    if role == self.role_assistant:
                        output = message
                    else:
                        input = ret + message
                return {
                    'input': [input],
                    'output': [output],
                }

        # 多轮对话
        if self.name in ['antglm_raw', 'chatglm2']:
            round_start = 1
        else:
            round_start = 0

        for i, (role, message) in enumerate(self.messages):
            # 轮次信息
            if self.name in ['antglm_raw', 'chatglm1', 'chatglm2']:
                if i % 2 == 0:
                    ret += self.turn_start.format(i // 2 + round_start)

            # 角色 + 内容
            role_end = self.general_role_end
            if role == self.role_assistant and self.assistant_end:
                role_end = self.assistant_end
            elif self.human_end:
                role_end = self.human_end

            ret += self.role_template.format(role) + message + role_end

            if role == self.role_assistant:
                # output 只保留实际 assistant 内容
                if not message:
                    outputs.append('')
                else:
                    outputs.append(message + role_end)
                # input 需要连接 assistant role
                inputs[-1] += ret[: -len(message + role_end)]
            elif all(
                [
                    role == self.role_observation,
                    len(self.messages) > 1,
                    self.messages[i - 1][0] != self.role_assistant,
                ]
            ):
                # observation 之前不是 assistant, 需要将 observation 和上一个 input 连接一起
                continue
            else:
                inputs.append(ret)
            ret = ''

            # 最后一轮不是机器人回复, 需要拼接机器人 role, 用于模型生成
            if i == len(self.messages) - 1 and role != self.role_assistant:
                inputs[-1] += self.role_template.format(self.role_assistant).strip()

        # 兼容逻辑, 去除 inputs 最后空格符号
        if self.prompt_style == PromptStyle.ANTGLM_RAW:
            inputs = [item.strip() for item in inputs]

        return {
            'input': inputs,
            'output': outputs,
        }

    @property
    def turns_num(self) -> int:
        '''和机器人的交互轮数, 以用户输出多少次为轮数个数.'''
        return sum([1 if msg[0] == self.role_human else 0 for msg in self.messages])

    def to_json(self) -> dict:
        '''输出 chat json dict 格式, 包含不同角色和机器人交互的每轮信息.

        Returns
            `List[dict]`, {
                "id": "xx",
                "messages": [
                    {"role": "HUMAN", "content": "xxx"}
                ]
                "turns": [
                    {"HUMAN": "xx", "OBSERVATION": "xx", "ASSISTANT": "xx"}
                ]
            }
        '''
        turns = []
        messages = []
        turn = {}
        for msg in self.messages:
            if msg[0] == self.role_assistant:
                messages.append({'role': 'ASSISTANT', 'content': msg[1]})
                turn['ASSISTANT'] = msg[1]
                turns.append(turn)
                turn = {}

            if msg[0] == self.role_human:
                messages.append({'role': 'HUMAN', 'content': msg[1]})
                turn['HUMAN'] = msg[1]

            if msg[0] == self.role_observation:
                messages.append({'role': 'OBSERVATION', 'content': msg[1]})
                turn['OBSERVATION'] = msg[1]

        if self.messages[-1][0] == self.role_human:
            messages.append({'role': 'ASSISTANT', 'content': ''})
            turn['ASSISTANT'] = ''
            turns.append(turn)

        result = self.origin_json or {}
        result.update(
            {
                'id': self.id,
                'name': self.name,
                'source': self.source,
                'lang': self.lang,
                'topic': self.topic,
                'system_template': self.system_template,
                'system_message': self.system_message,
                'turns': turns,
                'messages': messages,
            }
        )

        return result

    def set_system_message(self, system_message: str):
        '''Set the system message.'''
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        '''Append a new message.'''
        if not message:
            message = ''
        self.messages.append([role, message])

    def to_openai_api_messages(self) -> List[dict]:
        '''Convert the conversation to OpenAI chat completion format.'''
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return copy.deepcopy(self)
