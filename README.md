# Learning GenAI

## Basics
- [Catching up on the weird world of LLMs](https://simonwillison.net/2023/Aug/3/weird-world-of-llms/) - great overview of LLMs
- [Prompt injection attacks against GPT-3](https://simonwillison.net/2022/Sep/12/prompt-injection/) - prompt injection against LLMs is like SQL injection against Databases
- [Llama 2: Full Breakdown](https://www.youtube.com/watch?v=zJBpRn2zTco)
- [https://github.com/hollobit/Awesome-GenAITech](https://github.com/hollobit/Awesome-GenAITech) - list of GenAI applications/techniques

## Papers
- [Llama 1 paper](https://arxiv.org/pdf/2302.13971.pdf)
- [Llama 2 paper](https://arxiv.org/pdf/2307.09288.pdf)
- [GPT 4](https://openai.com/research/gpt-4)
   - [GPT 4 paper](https://arxiv.org/abs/2303.08774) 
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169) - the best AI paper of 2023?
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - “the Transformer paper” ([deep dive + implementation](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634))
- [https://github.com/Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) - curated list of papers about LLMs

## Articles/videos
- Q&A over docs: [ChatGPT for your data](https://jj09.net/chatgpt-for-your-data/)
   - [Using ChatGPT with YOUR OWN Data. This is magical. (LangChain OpenAI API)](https://www.youtube.com/watch?v=9AXP7tCI9PI)
   - [QA over Documents (LangChain docs)](https://python.langchain.com/docs/use_cases/question_answering/)
   - [How to implement Q&A against your documentation with GPT3, embeddings and Datasette](https://simonwillison.net/2023/Jan/13/semantic-search-answers/) (more complicated than LangChain)
   - [How To Install PrivateGPT - Chat With PDF, TXT, and CSV Files Privately! (Quick Setup Guide)](https://www.youtube.com/watch?v=jxSPx1bfl2M)
       - [https://github.com/imartinez/privateGPT](https://github.com/imartinez/privateGPT) 
           - pip3 install sentence_transformers
           - [privateGPT in Google collab](https://colab.research.google.com/drive/1zTG7gyTfqB19pGlJk0bk65_4N3fTgsqy)
- ["okay, but I want GPT to perform 10x for my specific use case" - Here is how](https://www.youtube.com/watch?v=Q9zv369Ggfk) - fine tuning LLM for your data
- [\[1hr Talk\] Intro to Large Language Models by Andrej Karpathy](https://youtu.be/zjkBMFhNj_g?si=X7Rz_2nHvyXBIfpS)
- [LLMs for Everyone: Running the LLaMA-13B model and LangChain in Google Colab](https://towardsdatascience.com/llms-for-everyone-running-the-llama-13b-model-and-langchain-in-google-colab-68d88021cf0b)
- [$0 Embeddings (OpenAI vs. free & open source)](https://www.youtube.com/watch?v=QdDoFfkVkcw)
- [Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps](https://www.youtube.com/watch?v=_j7JEDWuqLE)
- [AutoGPT tutorial: Build your personal assistant WITHOUT code (Via Relevance AI)](https://www.youtube.com/watch?v=iHzMg7gjJeY)
- [LangChain Crash Course: Build a AutoGPT app in 25 minutes!](https://www.youtube.com/watch?v=MlK6SIjcjE8)
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
   - [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&pp=iAQB)
   - [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&pp=iAQB)
   - [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB)
   - [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&pp=iAQB)
   - [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5&pp=iAQB)
   - [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=6&pp=iAQB)
   - **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&pp=iAQB)**
   - **[State of GPT | BRK216HFS](https://www.youtube.com/watch?v=bZQun8Y4L2A&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8&pp=iAQB)**
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c) - can fine tune and use Llama 2 on device
- [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [https://github.com/Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) 
- [LLMs on personal devices](https://simonwillison.net/series/llms-on-personal-devices/) 
- [Introducing Vector Search in Azure Cognitive Search | Azure Friday](https://www.youtube.com/watch?v=Bd9LWW4cxEU&t=114s&pp=ygUadmVjdG9yIHNlYXJjaCBhenVyZSBmcmlkYXk%3D)
- [What is a Vector database](https://www.pinecone.io/learn/vector-database/)
- [Vector Search Isn’t Enough | BRKFP301H](https://www.youtube.com/watch?v=5Qaxz2e2dVg)
- [Google "We Have No Moat, And Neither Does OpenAI"](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither) 
- [ExLlamaV2: The Fastest Library to Run LLMs | Towards Data Science](https://towardsdatascience.com/exllamav2-the-fastest-library-to-run-llms-32aeda294d26) 
- [Vector search and state of the art retrieval for Generative AI apps](https://ignite.microsoft.com/en-US/sessions/18618ca9-0e4d-4f9d-9a28-0bc3ef5cf54e?source=sessions) 
- [Stuff we figured out about AI in 2023](https://simonwillison.net/2023/Dec/31/ai-in-2023/) 

## DeepLearningAI
- [LangChain for LLM Application Development](https://learn.deeplearning.ai/langchain)
- [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system)
- [LangChain Chat with Your Data](https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/1/introduction)
- [Fine tuning LLMs](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)
- [How Diffusion Models Work](https://learn.deeplearning.ai/diffusion-models)
- [GenAI with LLMs (course)](https://www.deeplearning.ai/courses/generative-ai-with-llms)
- [Generative AI for Everyone](https://www.coursera.org/learn/generative-ai-for-everyone)

## Blogs/YouTube Channels
- [@AndrejKarpathy](https://www.youtube.com/@AndrejKarpathy) 
- [@AIJasonZ](https://www.youtube.com/@AIJasonZ)
- [@1littlecoder](https://www.youtube.com/@1littlecoder)
- [@aiexplained-official](https://www.youtube.com/@aiexplained-official)
- [@YannicKilcher](https://www.youtube.com/@YannicKilcher) 

## Other stuff
- OpenAI API cost: [https://gptforwork.com/tools/openai-chatgpt-api-pricing-calculator](https://gptforwork.com/tools/openai-chatgpt-api-pricing-calculator)
- [AI in Education](https://belitsoft.com/custom-elearning-development/ai-in-education/ai-in-edtech)
   - [How Can AI Be Used in Education](https://belitsoft.com/custom-elearning-development/ai-in-education) 
- [Applying Generative AI to Enterprise Use Cases: A Step-by-Step Guide](https://foundationcapital.com/applying-generative-ai-to-enterprise-use-cases-a-step-by-step-guide/)~~ ~~
   - Automating Customer Service is most popular now
   - Adjusting LLM to your use case: Prompt engineering, context retrieval, fine tuning
- [History, Waves and Winters in AI](https://medium.com/hackernoon/history-waves-and-winters-in-ai-dd5feb558e45)
- [top 20 AI tools for productivity](https://twitter.com/TheRundownAI/status/1686157794901217280?s=20) 
- [Practical Deep Learning](https://course.fast.ai/)
- [Bringing Generative AI to Life with NVIDIA Jetson](https://developer.nvidia.com/blog/bringing-generative-ai-to-life-with-jetson/)

# Things to Understand
- [Fine-tunning LLMs](https://teetracker.medium.com/fine-tuning-llms-9fe553a514d0)
- [Neural Networks Embeddings Explained](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)
- [Vector databases](https://www.pinecone.io/learn/vector-database/)
- [Few shot learning](https://blog.paperspace.com/few-shot-learning/)
- [Zero-shot and few-shot prompting](https://machinelearningmastery.com/what-are-zero-shot-prompting-and-few-shot-prompting/)
