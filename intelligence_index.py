# -*- coding: utf-8 -*-
# Yapay zeka basarimi tek sayiyla: Artificial Analysis Zeka Endeksi (Intelligence Index)
# Veri kaynagi: artificialanalysis.ai  (472 model, 2022-11-30 - 2026-09-03)
# Bu dosya scratchpad/make_ii_scripts.py ile uretildi; veri asagida GOMULU.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = [
[
"2022-11-30",
3.2,
"GPT-3.5 Turbo",
"OpenAI",
0
],
[
"2023-02-24",
1.7,
"Llama 65B",
"Meta",
1
],
[
"2023-03-14",
1.7,
"Claude Instant",
"Anthropic",
0
],
[
"2023-03-14",
6.8,
"GPT-4",
"OpenAI",
0
],
[
"2023-05-10",
2.8,
"PALM-2",
"Google",
0
],
[
"2023-07-11",
3.3,
"Claude 2.0",
"Anthropic",
0
],
[
"2023-07-18",
2.6,
"Llama 2 Chat 13B",
"Meta",
1
],
[
"2023-07-18",
2.6,
"Llama 2 Chat 70B",
"Meta",
1
],
[
"2023-07-18",
3.9,
"Llama 2 Chat 7B",
"Meta",
1
],
[
"2023-09-25",
1.7,
"Qwen Chat 14B",
"Alibaba",
1
],
[
"2023-09-27",
1.7,
"Mistral 7B Instruct",
"Mistral",
1
],
[
"2023-11-06",
7.7,
"GPT-4 Turbo",
"OpenAI",
0
],
[
"2023-11-21",
3.5,
"Claude 2.1",
"Anthropic",
0
],
[
"2023-11-29",
2.6,
"DeepSeek LLM 67B Chat",
"DeepSeek",
1
],
[
"2023-11-30",
3.0,
"Qwen Chat 72B",
"Alibaba",
1
],
[
"2023-12-06",
2.7,
"Gemini 1.0 Pro",
"Google",
0
],
[
"2023-12-06",
4.3,
"Gemini 1.0 Ultra",
"Google",
0
],
[
"2023-12-11",
2.0,
"Mixtral 8x7B Instruct",
"Mistral",
1
],
[
"2023-12-11",
3.2,
"Mistral Medium",
"Mistral",
0
],
[
"2023-12-18",
2.6,
"OpenChat 3.5",
"OpenChat",
1
],
[
"2024-01-25",
6.0,
"Solar Mini",
"Upstage",
1
],
[
"2024-02-26",
3.2,
"Mistral Small",
"Mistral",
0
],
[
"2024-02-26",
4.1,
"Mistral Large",
"Mistral",
0
],
[
"2024-02-26",
5.7,
"Phi-4 Mini Instruct",
"Microsoft",
1
],
[
"2024-03-04",
3.5,
"Claude 3 Haiku",
"Anthropic",
0
],
[
"2024-03-04",
4.4,
"Claude 3 Sonnet",
"Anthropic",
0
],
[
"2024-03-04",
11.8,
"Claude 3 Opus",
"Anthropic",
0
],
[
"2024-03-12",
1.7,
"Command-R",
"Cohere",
1
],
[
"2024-03-17",
5.8,
"Grok-1",
"xAI",
1
],
[
"2024-03-27",
2.6,
"DBRX Instruct",
"Databricks",
1
],
[
"2024-04-04",
2.6,
"Command-R+",
"Cohere",
1
],
[
"2024-04-17",
4.0,
"Mixtral 8x22B Instruct",
"Mistral",
1
],
[
"2024-04-18",
1.0,
"Llama 3 Instruct 8B",
"Meta",
1
],
[
"2024-04-18",
3.1,
"Llama 3 Instruct 70B",
"Meta",
1
],
[
"2024-04-23",
4.3,
"Phi-3 Mini Instruct 3.8B",
"Microsoft",
1
],
[
"2024-04-24",
3.0,
"Arctic Instruct",
"Snowflake",
1
],
[
"2024-04-25",
3.7,
"Qwen1.5 Chat 110B",
"Alibaba",
1
],
[
"2024-05-06",
3.3,
"DeepSeek-V2-Chat",
"DeepSeek",
1
],
[
"2024-05-13",
8.4,
"GPT-4o",
"OpenAI",
0
],
[
"2024-05-14",
4.6,
"Gemini 1.5 Flash",
"Google",
0
],
[
"2024-05-15",
6.1,
"Gemini 1.5 Pro",
"Google",
0
],
[
"2024-06-07",
5.7,
"Qwen2 Instruct 72B",
"Alibaba",
1
],
[
"2024-06-17",
2.7,
"DeepSeek Coder V2 Lite Instruct",
"DeepSeek",
1
],
[
"2024-06-17",
4.7,
"DeepSeek-Coder-V2",
"DeepSeek",
1
],
[
"2024-06-21",
8.1,
"Claude 3.5 Sonnet",
"Anthropic",
0
],
[
"2024-07-18",
6.7,
"GPT-4o mini",
"OpenAI",
0
],
[
"2024-07-23",
6.5,
"Llama 3.1 Instruct 70B",
"Meta",
1
],
[
"2024-07-23",
7.4,
"Llama 3.1 Instruct 8B",
"Meta",
1
],
[
"2024-07-23",
8.3,
"Llama 3.1 Instruct 405B",
"Meta",
1
],
[
"2024-07-24",
7.0,
"Mistral Large 2",
"Mistral",
1
],
[
"2024-08-06",
9.4,
"GPT-4o",
"OpenAI",
0
],
[
"2024-08-13",
7.3,
"Grok Beta",
"xAI",
0
],
[
"2024-08-15",
4.8,
"Hermes 3 - Llama-3.1 70B",
"Nous Research",
1
],
[
"2024-08-22",
2.3,
"Jamba 1.5 Mini",
"AI21 Labs",
1
],
[
"2024-08-22",
4.8,
"Jamba 1.5 Large",
"AI21 Labs",
1
],
[
"2024-09-06",
6.4,
"DeepSeek-V2.5",
"DeepSeek",
1
],
[
"2024-09-12",
14.0,
"o1-mini",
"OpenAI",
0
],
[
"2024-09-12",
17.2,
"o1-preview",
"OpenAI",
0
],
[
"2024-09-17",
4.3,
"Mistral Small",
"Mistral",
1
],
[
"2024-09-19",
4.1,
"Qwen2.5 Coder Instruct 7B",
"Alibaba",
1
],
[
"2024-09-19",
7.2,
"Qwen2.5 Instruct 32B",
"Alibaba",
1
],
[
"2024-09-19",
9.4,
"Qwen2.5 Instruct 72B",
"Alibaba",
1
],
[
"2024-09-24",
7.8,
"Gemini 1.5 Flash",
"Google",
0
],
[
"2024-09-24",
9.9,
"Gemini 1.5 Pro",
"Google",
0
],
[
"2024-09-25",
1.0,
"Llama 3.2 Instruct 1B",
"Meta",
1
],
[
"2024-09-25",
3.0,
"Llama 3.2 Instruct 11B",
"Meta",
1
],
[
"2024-09-25",
3.4,
"Molmo 7B-D",
"Allen Institute for AI",
1
],
[
"2024-09-25",
3.9,
"Llama 3.2 Instruct 3B",
"Meta",
1
],
[
"2024-09-25",
6.0,
"Llama 3.2 Instruct 90B",
"Meta",
1
],
[
"2024-09-30",
3.0,
"LFM 40B",
"Liquid AI",
0
],
[
"2024-10-03",
5.2,
"Gemini 1.5 Flash-8B",
"Google",
0
],
[
"2024-10-04",
6.0,
"Reka Flash",
"Reka AI",
0
],
[
"2024-10-15",
7.4,
"Llama 3.1 Nemotron Instruct 70B",
"NVIDIA",
1
],
[
"2024-10-22",
9.8,
"Claude 3.5 Sonnet",
"Anthropic",
0
],
[
"2024-10-22",
12.2,
"Claude 3.5 Haiku",
"Anthropic",
0
],
[
"2024-11-11",
6.9,
"Qwen2.5 Coder Instruct 32B",
"Alibaba",
1
],
[
"2024-11-18",
6.0,
"Qwen2.5 Turbo",
"Alibaba",
0
],
[
"2024-11-18",
8.0,
"Pixtral Large",
"Mistral",
1
],
[
"2024-11-18",
9.0,
"Mistral Large 2",
"Mistral",
1
],
[
"2024-11-20",
11.1,
"GPT-4o",
"OpenAI",
0
],
[
"2024-11-26",
3.5,
"OLMo 2 7B",
"Allen Institute for AI",
1
],
[
"2024-11-27",
9.1,
"QwQ 32B-Preview",
"Alibaba",
1
],
[
"2024-12-03",
4.4,
"Nova Micro",
"Amazon",
0
],
[
"2024-12-03",
6.7,
"Nova Lite",
"Amazon",
0
],
[
"2024-12-03",
7.5,
"Nova Pro",
"Amazon",
0
],
[
"2024-12-05",
23.9,
"o1",
"OpenAI",
0
],
[
"2024-12-06",
9.3,
"Llama 3.3 Instruct 70B",
"Meta",
1
],
[
"2024-12-10",
6.5,
"DeepSeek-V2.5",
"DeepSeek",
1
],
[
"2024-12-11",
10.6,
"Gemini 2.0 Flash",
"Google",
0
],
[
"2024-12-12",
4.6,
"Phi-4",
"Microsoft",
1
],
[
"2024-12-12",
7.8,
"Grok 2",
"xAI",
1
],
[
"2024-12-19",
6.4,
"Gemini 2.0 Flash Thinking Experimental",
"Google",
0
],
[
"2024-12-26",
14.2,
"DeepSeek V3",
"DeepSeek",
1
],
[
"2025-01-20",
3.3,
"DeepSeek R1 Distill Qwen 1.5B",
"DeepSeek",
1
],
[
"2025-01-20",
6.2,
"DeepSeek R1 Distill Llama 8B",
"DeepSeek",
1
],
[
"2025-01-20",
9.7,
"DeepSeek R1 Distill Qwen 14B",
"DeepSeek",
1
],
[
"2025-01-20",
9.8,
"DeepSeek R1 Distill Llama 70B",
"DeepSeek",
1
],
[
"2025-01-20",
11.0,
"DeepSeek R1 Distill Qwen 32B",
"DeepSeek",
1
],
[
"2025-01-20",
18.6,
"DeepSeek R1",
"DeepSeek",
1
],
[
"2025-01-21",
9.1,
"Sonar Pro",
"Perplexity",
0
],
[
"2025-01-21",
9.4,
"Sonar",
"Perplexity",
0
],
[
"2025-01-21",
13.3,
"Gemini 2.0 Flash Thinking Experimental",
"Google",
0
],
[
"2025-01-28",
10.1,
"Qwen2.5 Max",
"Alibaba",
0
],
[
"2025-01-28",
11.6,
"Sonar Reasoning",
"Perplexity",
0
],
[
"2025-01-28",
18.0,
"Sonar Reasoning Pro",
"Perplexity",
0
],
[
"2025-01-30",
6.7,
"Mistral Small 3",
"Mistral",
1
],
[
"2025-01-30",
8.1,
"Llama 3.1 Tulu3 405B",
"Allen Institute for AI",
1
],
[
"2025-01-31",
19.2,
"o3-mini",
"OpenAI",
0
],
[
"2025-02-05",
8.4,
"Gemini 2.0 Flash-Lite",
"Google",
0
],
[
"2025-02-05",
11.8,
"Gemini 2.0 Pro Experimental",
"Google",
0
],
[
"2025-02-05",
12.2,
"Gemini 2.0 Flash",
"Google",
0
],
[
"2025-02-13",
1.9,
"DeepHermes 3 - Llama-3.1 8B Preview",
"Nous Research",
1
],
[
"2025-02-15",
8.1,
"GPT-4o",
"OpenAI",
0
],
[
"2025-02-17",
6.2,
"Mistral Saba",
"Mistral",
0
],
[
"2025-02-18",
6.0,
"R1 1776",
"Perplexity",
1
],
[
"2025-02-19",
15.2,
"Grok 3 Reasoning Beta",
"xAI",
0
],
[
"2025-02-19",
18.6,
"Grok 3",
"xAI",
0
],
[
"2025-02-19",
22.9,
"Grok 3 mini Reasoning",
"xAI",
0
],
[
"2025-02-24",
27.6,
"Claude 3.7 Sonnet",
"Anthropic",
0
],
[
"2025-02-25",
8.6,
"Gemini 2.0 Flash-Lite",
"Google",
0
],
[
"2025-02-26",
4.2,
"Phi-4 Multimodal Instruct",
"Microsoft",
1
],
[
"2025-02-27",
13.6,
"GPT-4.5",
"OpenAI",
0
],
[
"2025-03-05",
13.4,
"QwQ 32B",
"Alibaba",
1
],
[
"2025-03-06",
2.1,
"Jamba 1.6 Mini",
"AI21 Labs",
1
],
[
"2025-03-06",
4.7,
"Jamba 1.6 Large",
"AI21 Labs",
1
],
[
"2025-03-10",
3.7,
"Reka Flash 3",
"Reka AI",
1
],
[
"2025-03-12",
1.0,
"Gemma 3 4B Instruct",
"Google",
1
],
[
"2025-03-12",
5.5,
"Gemma 3 12B Instruct",
"Google",
1
],
[
"2025-03-12",
7.4,
"Gemma 3 27B Instruct",
"Google",
1
],
[
"2025-03-13",
1.0,
"Gemma 3 1B Instruct",
"Google",
1
],
[
"2025-03-13",
4.7,
"OLMo 2 32B",
"Allen Institute for AI",
1
],
[
"2025-03-13",
5.0,
"DeepHermes 3 - Mistral 24B Preview",
"Nous Research",
1
],
[
"2025-03-13",
7.5,
"Command A",
"Cohere",
1
],
[
"2025-03-17",
14.9,
"Mistral Small 3.1",
"Mistral",
1
],
[
"2025-03-18",
12.2,
"Llama 3.3 Nemotron Super 49B v1",
"NVIDIA",
1
],
[
"2025-03-19",
19.1,
"o1-pro",
"OpenAI",
0
],
[
"2025-03-25",
15.2,
"DeepSeek V3 0324",
"DeepSeek",
1
],
[
"2025-03-25",
23.4,
"Gemini 2.5 Pro Preview",
"Google",
0
],
[
"2025-03-27",
12.3,
"GPT-4o",
"OpenAI",
0
],
[
"2025-04-05",
10.3,
"Llama 4 Scout",
"Meta",
1
],
[
"2025-04-05",
14.5,
"Llama 4 Maverick",
"Meta",
1
],
[
"2025-04-07",
8.9,
"Llama 3.1 Nemotron Ultra 253B v1",
"NVIDIA",
1
],
[
"2025-04-14",
9.6,
"GPT-4.1 nano",
"OpenAI",
0
],
[
"2025-04-14",
14.8,
"GPT-4.1 mini",
"OpenAI",
0
],
[
"2025-04-14",
19.6,
"GPT-4.1",
"OpenAI",
0
],
[
"2025-04-16",
1.3,
"Granite 3.3 8B",
"IBM",
1
],
[
"2025-04-16",
26.1,
"o4-mini",
"OpenAI",
0
],
[
"2025-04-16",
31.1,
"o3",
"OpenAI",
0
],
[
"2025-04-17",
17.7,
"Gemini 2.5 Flash Preview",
"Google",
0
],
[
"2025-04-28",
1.0,
"Qwen3 0.6B",
"Alibaba",
1
],
[
"2025-04-28",
2.2,
"Qwen3 1.7B",
"Alibaba",
1
],
[
"2025-04-28",
8.2,
"Qwen3 4B",
"Alibaba",
1
],
[
"2025-04-28",
8.3,
"Qwen3 8B",
"Alibaba",
1
],
[
"2025-04-28",
9.2,
"Qwen3 30B A3B",
"Alibaba",
1
],
[
"2025-04-28",
10.4,
"Qwen3 14B",
"Alibaba",
1
],
[
"2025-04-28",
11.4,
"Qwen3 32B",
"Alibaba",
1
],
[
"2025-04-28",
13.5,
"Qwen3 235B A22B",
"Alibaba",
1
],
[
"2025-04-30",
12.7,
"Nova Premier",
"Amazon",
0
],
[
"2025-05-06",
22.7,
"Gemini 2.5 Pro Preview",
"Google",
0
],
[
"2025-05-07",
12.5,
"Mistral Medium 3",
"Mistral",
0
],
[
"2025-05-20",
4.2,
"Gemma 3n E4B Instruct Preview",
"Google",
1
],
[
"2025-05-20",
8.4,
"Llama 3.1 Nemotron Nano 4B v1.1",
"NVIDIA",
1
],
[
"2025-05-20",
12.5,
"Solar Pro 2",
"Upstage",
0
],
[
"2025-05-20",
20.3,
"Gemini 2.5 Flash",
"Google",
0
],
[
"2025-05-21",
11.8,
"Devstral Small",
"Mistral",
1
],
[
"2025-05-22",
29.8,
"Claude 4 Sonnet",
"Anthropic",
0
],
[
"2025-05-22",
31.7,
"Claude 4 Opus",
"Anthropic",
0
],
[
"2025-05-23",
2.6,
"Sarvam M",
"Sarvam",
1
],
[
"2025-05-28",
20.4,
"DeepSeek R1 0528",
"DeepSeek",
1
],
[
"2025-05-29",
10.3,
"DeepSeek R1 0528 Qwen3 8B",
"DeepSeek",
1
],
[
"2025-06-05",
25.9,
"Gemini 2.5 Pro",
"Google",
0
],
[
"2025-06-10",
10.6,
"Magistral Small 1",
"Mistral",
1
],
[
"2025-06-10",
12.5,
"Magistral Medium 1",
"Mistral",
0
],
[
"2025-06-10",
33.3,
"o3-pro",
"OpenAI",
0
],
[
"2025-06-17",
11.4,
"Gemini 2.5 Flash-Lite",
"Google",
0
],
[
"2025-06-17",
14.5,
"MiniMax M1 40k",
"MiniMax",
1
],
[
"2025-06-17",
17.9,
"MiniMax M1 80k",
"MiniMax",
1
],
[
"2025-06-20",
10.7,
"Mistral Small 3.2",
"Mistral",
1
],
[
"2025-06-26",
1.0,
"Gemma 3n E2B Instruct",
"Google",
1
],
[
"2025-06-26",
1.0,
"Gemma 3n E4B Instruct",
"Google",
1
],
[
"2025-06-30",
8.9,
"ERNIE 4.5 300B A47B",
"Baidu",
1
],
[
"2025-07-07",
2.3,
"Jamba 1.7 Mini",
"AI21 Labs",
1
],
[
"2025-07-07",
5.0,
"Jamba 1.7 Large",
"AI21 Labs",
1
],
[
"2025-07-09",
8.8,
"Solar Pro 2",
"Upstage",
0
],
[
"2025-07-10",
1.0,
"LFM2 1.2B",
"Liquid AI",
1
],
[
"2025-07-10",
9.1,
"Devstral Small",
"Mistral",
1
],
[
"2025-07-10",
12.4,
"Devstral Medium",
"Mistral",
0
],
[
"2025-07-10",
34.1,
"Grok 4",
"xAI",
0
],
[
"2025-07-11",
19.7,
"Kimi K2",
"Kimi",
1
],
[
"2025-07-15",
2.5,
"Exaone 4.0 1.2B",
"LG AI Research",
1
],
[
"2025-07-15",
10.5,
"EXAONE 4.0 32B",
"LG AI Research",
1
],
[
"2025-07-21",
18.4,
"Qwen3 235B A22B 2507 Instruct",
"Alibaba",
1
],
[
"2025-07-22",
18.2,
"Qwen3 Coder 480B A35B Instruct",
"Alibaba",
1
],
[
"2025-07-25",
12.4,
"Llama Nemotron Super 49B v1.5",
"NVIDIA",
1
],
[
"2025-07-25",
19.9,
"Qwen3 235B A22B 2507",
"Alibaba",
1
],
[
"2025-07-28",
16.7,
"GLM-4.5-Air",
"Z.ai",
1
],
[
"2025-07-28",
19.7,
"GLM-4.5",
"Z.ai",
1
],
[
"2025-07-29",
8.9,
"Qwen3 30B A3B 2507 Instruct",
"Alibaba",
1
],
[
"2025-07-30",
14.6,
"Qwen3 30B A3B 2507",
"Alibaba",
1
],
[
"2025-07-31",
13.6,
"Qwen3 Coder 30B A3B Instruct",
"Alibaba",
1
],
[
"2025-08-05",
15.2,
"gpt-oss-20b",
"OpenAI",
1
],
[
"2025-08-05",
24.1,
"gpt-oss-120b",
"OpenAI",
1
],
[
"2025-08-05",
34.5,
"Claude 4.1 Opus",
"Anthropic",
0
],
[
"2025-08-06",
6.9,
"Qwen3 4B 2507 Instruct",
"Alibaba",
1
],
[
"2025-08-06",
11.9,
"Qwen3 4B 2507",
"Alibaba",
1
],
[
"2025-08-07",
20.1,
"GPT-5 nano",
"OpenAI",
0
],
[
"2025-08-07",
31.6,
"GPT-5 mini",
"OpenAI",
0
],
[
"2025-08-07",
35.3,
"GPT-5",
"OpenAI",
0
],
[
"2025-08-11",
9.0,
"GLM-4.5V",
"Z.ai",
1
],
[
"2025-08-12",
14.7,
"Mistral Medium 3.1",
"Mistral",
0
],
[
"2025-08-14",
2.0,
"Gemma 3 270M",
"Google",
1
],
[
"2025-08-18",
8.7,
"NVIDIA Nemotron Nano 9B V2",
"NVIDIA",
1
],
[
"2025-08-20",
18.5,
"Seed-OSS-36B-Instruct",
"ByteDance",
1
],
[
"2025-08-21",
21.4,
"DeepSeek V3.1",
"DeepSeek",
1
],
[
"2025-08-27",
8.8,
"Hermes 4 - Llama-3.1 405B",
"Nous Research",
1
],
[
"2025-08-27",
9.9,
"Hermes 4 - Llama-3.1 70B",
"Nous Research",
1
],
[
"2025-08-28",
22.0,
"Grok Code Fast 1",
"xAI",
0
],
[
"2025-09-02",
1.0,
"Apertus 8B Instruct",
"Swiss AI Initiative",
1
],
[
"2025-09-02",
2.0,
"Apertus 70B Instruct",
"Swiss AI Initiative",
1
],
[
"2025-09-05",
19.4,
"Qwen3 Max",
"Alibaba",
0
],
[
"2025-09-05",
24.0,
"Kimi K2 0905",
"Kimi",
1
],
[
"2025-09-08",
15.2,
"Gemini 2.5 Flash-Lite Preview",
"Google",
0
],
[
"2025-09-09",
3.4,
"Ling-mini-2.0",
"InclusionAI",
1
],
[
"2025-09-11",
13.8,
"Qwen3 Next 80B A3B Instruct",
"Alibaba",
1
],
[
"2025-09-11",
16.9,
"Qwen3 Next 80B A3B",
"Alibaba",
1
],
[
"2025-09-17",
9.6,
"Ling-flash-2.0",
"InclusionAI",
1
],
[
"2025-09-17",
11.5,
"Magistral Small 1.2",
"Mistral",
1
],
[
"2025-09-18",
18.0,
"Magistral Medium 1.2",
"Mistral",
0
],
[
"2025-09-19",
8.0,
"Ring-flash-2.0",
"InclusionAI",
1
],
[
"2025-09-19",
27.9,
"Grok 4 Fast",
"xAI",
0
],
[
"2025-09-22",
2.0,
"Granite 4.0 Micro",
"IBM",
1
],
[
"2025-09-22",
4.8,
"Qwen3 Omni 30B A3B Instruct",
"Alibaba",
1
],
[
"2025-09-22",
4.9,
"Granite 4.0 H Small",
"IBM",
1
],
[
"2025-09-22",
9.5,
"Qwen3 Omni 30B A3B",
"Alibaba",
1
],
[
"2025-09-22",
31.1,
"DeepSeek V3.1 Terminus",
"DeepSeek",
1
],
[
"2025-09-23",
2.3,
"LFM2 2.6B",
"Liquid AI",
1
],
[
"2025-09-23",
14.4,
"Qwen3 VL 235B A22B Instruct",
"Alibaba",
1
],
[
"2025-09-23",
20.9,
"Qwen3 VL 235B A22B",
"Alibaba",
1
],
[
"2025-09-23",
24.5,
"Qwen3 Max",
"Alibaba",
0
],
[
"2025-09-23",
37.0,
"GPT-5 Codex",
"OpenAI",
0
],
[
"2025-09-25",
13.1,
"Gemini 2.5 Flash-Lite Preview",
"Google",
0
],
[
"2025-09-25",
24.2,
"Gemini 2.5 Flash Preview",
"Google",
0
],
[
"2025-09-29",
25.9,
"DeepSeek V3.2 Exp",
"DeepSeek",
1
],
[
"2025-09-29",
37.4,
"Claude 4.5 Sonnet",
"Anthropic",
0
],
[
"2025-09-30",
21.6,
"Apriel-v1.5-15B-Thinker",
"ServiceNow",
1
],
[
"2025-09-30",
29.3,
"GLM-4.6",
"Z.ai",
1
],
[
"2025-10-03",
9.9,
"Qwen3 VL 30B A3B Instruct",
"Alibaba",
1
],
[
"2025-10-03",
13.4,
"Qwen3 VL 30B A3B",
"Alibaba",
1
],
[
"2025-10-07",
1.3,
"LFM2 8B A1B",
"Liquid AI",
1
],
[
"2025-10-08",
3.8,
"Jamba Reasoning 3B",
"AI21 Labs",
1
],
[
"2025-10-08",
12.7,
"Ling-1T",
"InclusionAI",
1
],
[
"2025-10-13",
16.3,
"Ring-1T",
"InclusionAI",
1
],
[
"2025-10-14",
3.7,
"Qwen3 VL 4B Instruct",
"Alibaba",
1
],
[
"2025-10-14",
7.7,
"Qwen3 VL 4B",
"Alibaba",
1
],
[
"2025-10-14",
8.2,
"Qwen3 VL 8B Instruct",
"Alibaba",
1
],
[
"2025-10-14",
10.5,
"Qwen3 VL 8B",
"Alibaba",
1
],
[
"2025-10-15",
29.9,
"Claude 4.5 Haiku",
"Anthropic",
0
],
[
"2025-10-21",
11.0,
"Qwen3 VL 32B Instruct",
"Alibaba",
1
],
[
"2025-10-21",
18.1,
"Qwen3 VL 32B",
"Alibaba",
1
],
[
"2025-10-26",
28.9,
"MiniMax-M2",
"MiniMax",
1
],
[
"2025-10-28",
1.0,
"Granite 4.0 350M",
"IBM",
1
],
[
"2025-10-28",
1.0,
"Granite 4.0 H 350M",
"IBM",
1
],
[
"2025-10-28",
1.6,
"Granite 4.0 1B",
"IBM",
1
],
[
"2025-10-28",
2.2,
"Granite 4.0 H 1B",
"IBM",
1
],
[
"2025-10-28",
8.8,
"NVIDIA Nemotron Nano 12B v2 VL",
"NVIDIA",
1
],
[
"2025-10-29",
20.8,
"Nova 2.0 Lite",
"Amazon",
0
],
[
"2025-10-30",
8.4,
"Kimi Linear 48B A3B Instruct",
"Kimi",
1
],
[
"2025-11-03",
25.5,
"Qwen3 Max Thinking",
"Alibaba",
0
],
[
"2025-11-06",
33.5,
"Kimi K2 Thinking",
"Kimi",
1
],
[
"2025-11-11",
26.5,
"Doubao Seed Code",
"ByteDance",
0
],
[
"2025-11-11",
28.9,
"KAT-Coder-Pro V1",
"KwaiKAT",
0
],
[
"2025-11-13",
22.3,
"ERNIE 5.0 Thinking Preview",
"Baidu",
0
],
[
"2025-11-13",
31.3,
"GPT-5.1 Codex mini",
"OpenAI",
0
],
[
"2025-11-13",
35.6,
"GPT-5.1 Codex",
"OpenAI",
0
],
[
"2025-11-13",
37.5,
"GPT-5.1",
"OpenAI",
0
],
[
"2025-11-18",
40.6,
"Gemini 3 Pro Preview",
"Google",
0
],
[
"2025-11-19",
31.3,
"Grok 4.1 Fast",
"xAI",
0
],
[
"2025-11-20",
2.4,
"Olmo 3 7B Instruct",
"Allen Institute for AI",
1
],
[
"2025-11-20",
3.6,
"Olmo 3 7B Think",
"Allen Institute for AI",
1
],
[
"2025-11-20",
6.1,
"Olmo 3 32B Think",
"Allen Institute for AI",
1
],
[
"2025-11-24",
41.9,
"Claude Opus 4.5",
"Anthropic",
0
],
[
"2025-11-25",
20.8,
"Apriel-v1.6-15B-Thinker",
"ServiceNow",
1
],
[
"2025-11-26",
21.3,
"Nova 2.0 Omni",
"Amazon",
0
],
[
"2025-11-27",
15.7,
"INTELLECT-3",
"Prime Intellect",
1
],
[
"2025-11-27",
22.1,
"Nova 2.0 Pro Preview",
"Amazon",
0
],
[
"2025-12-01",
22.6,
"DeepSeek V3.2 Speciale",
"DeepSeek",
1
],
[
"2025-12-01",
32.8,
"DeepSeek V3.2",
"DeepSeek",
1
],
[
"2025-12-02",
7.1,
"Ministral 3 3B",
"Mistral",
1
],
[
"2025-12-02",
9.0,
"Ministral 3 8B",
"Mistral",
1
],
[
"2025-12-02",
11.2,
"Ministral 3 14B",
"Mistral",
1
],
[
"2025-12-02",
15.9,
"Mistral Large 3",
"Mistral",
1
],
[
"2025-12-04",
12.8,
"Motif-2-12.7B-Reasoning",
"Motif Technologies",
0
],
[
"2025-12-05",
14.2,
"K2-V2",
"MBZUAI Institute of Foundation Models",
1
],
[
"2025-12-08",
16.9,
"GLM-4.6V",
"Z.ai",
1
],
[
"2025-12-09",
17.7,
"Devstral Small 2",
"Mistral",
1
],
[
"2025-12-09",
19.2,
"Devstral 2",
"Mistral",
1
],
[
"2025-12-11",
1.6,
"Molmo2-8B",
"Allen Institute for AI",
1
],
[
"2025-12-11",
16.6,
"Mi:dm K 2.5 Pro",
"Korea Telecom",
0
],
[
"2025-12-11",
41.2,
"GPT-5.2 Codex",
"OpenAI",
0
],
[
"2025-12-11",
43.3,
"GPT-5.2",
"OpenAI",
0
],
[
"2025-12-12",
7.9,
"Olmo 3.1 32B Think",
"Allen Institute for AI",
1
],
[
"2025-12-15",
14.5,
"NVIDIA Nemotron 3 Nano 30B A3B",
"NVIDIA",
1
],
[
"2025-12-15",
17.4,
"K2 Think V2",
"MBZUAI Institute of Foundation Models",
1
],
[
"2025-12-16",
34.0,
"MiMo-V2-Flash",
"Xiaomi",
1
],
[
"2025-12-17",
15.2,
"Solar Open 100B",
"Upstage",
1
],
[
"2025-12-17",
38.7,
"Gemini 3 Flash Preview",
"Google",
0
],
[
"2025-12-22",
34.5,
"GLM-4.7",
"Z.ai",
1
],
[
"2025-12-23",
32.1,
"MiniMax-M2.1",
"MiniMax",
1
],
[
"2025-12-26",
17.2,
"HyperCLOVA X SEED Think",
"Naver",
1
],
[
"2025-12-31",
22.5,
"K-EXAONE",
"LG AI Research",
1
],
[
"2026-01-04",
9.7,
"Falcon-H1R-7B",
"TII UAE",
1
],
[
"2026-01-05",
1.0,
"LFM2.5-VL-1.6B",
"Liquid AI",
1
],
[
"2026-01-05",
2.3,
"LFM2.5-1.2B-Instruct",
"Liquid AI",
1
],
[
"2026-01-13",
6.2,
"Olmo 3.1 32B Instruct",
"Allen Institute for AI",
1
],
[
"2026-01-19",
23.3,
"GLM-4.7-Flash",
"Z.ai",
1
],
[
"2026-01-20",
2.3,
"LFM2.5-1.2B-Thinking",
"Liquid AI",
1
],
[
"2026-01-20",
9.3,
"Step3 VL 10B",
"StepFun",
1
],
[
"2026-01-26",
32.5,
"Qwen3 Max Thinking",
"Alibaba",
0
],
[
"2026-01-27",
36.0,
"Kimi K2.5",
"Kimi",
1
],
[
"2026-01-28",
17.4,
"LongCat Flash Lite",
"LongCat",
1
],
[
"2026-02-02",
26.0,
"Step 3.5 Flash",
"StepFun",
1
],
[
"2026-02-03",
21.3,
"Qwen3 Coder Next",
"Alibaba",
1
],
[
"2026-02-05",
44.9,
"Claude Opus 4.6",
"Anthropic",
0
],
[
"2026-02-05",
45.5,
"GPT-5.3 Codex",
"OpenAI",
0
],
[
"2026-02-10",
12.3,
"Tri-21B-Think",
"Trillion Labs",
1
],
[
"2026-02-10",
13.6,
"Tri-21B-think Preview",
"Trillion Labs",
1
],
[
"2026-02-11",
11.0,
"Nanbeige4.1-3B",
"Nanbeige",
1
],
[
"2026-02-11",
40.6,
"GLM-5",
"Z.ai",
1
],
[
"2026-02-12",
34.5,
"MiniMax-M2.5",
"MiniMax",
1
],
[
"2026-02-16",
34.3,
"Qwen3.5 397B A17B",
"Alibaba",
1
],
[
"2026-02-17",
1.0,
"Tiny Aya Global",
"Cohere",
1
],
[
"2026-02-17",
48.4,
"Claude Sonnet 4.6",
"Anthropic",
0
],
[
"2026-02-19",
47.7,
"Gemini 3.1 Pro Preview",
"Google",
0
],
[
"2026-02-20",
21.9,
"Mercury 2",
"Inception",
0
],
[
"2026-02-24",
29.9,
"Qwen3.5 35B A3B",
"Alibaba",
1
],
[
"2026-02-24",
32.8,
"Qwen3.5 122B A10B",
"Alibaba",
1
],
[
"2026-02-24",
34.6,
"Qwen3.5 27B",
"Alibaba",
1
],
[
"2026-02-25",
4.6,
"LFM2 24B A2B",
"Liquid AI",
1
],
[
"2026-03-02",
5.2,
"Qwen3.5 0.8B",
"Alibaba",
1
],
[
"2026-03-02",
7.4,
"Qwen3.5 2B",
"Alibaba",
1
],
[
"2026-03-02",
20.4,
"Qwen3.5 4B",
"Alibaba",
1
],
[
"2026-03-02",
21.8,
"Qwen3.5 9B",
"Alibaba",
1
],
[
"2026-03-03",
25.6,
"Gemini 3.1 Flash-Lite",
"Google",
0
],
[
"2026-03-05",
53.1,
"GPT-5.4",
"OpenAI",
0
],
[
"2026-03-06",
6.4,
"Sarvam 30B",
"Sarvam",
1
],
[
"2026-03-06",
11.9,
"Sarvam 105B",
"Sarvam",
1
],
[
"2026-03-10",
37.4,
"Grok 4.20 0309",
"xAI",
0
],
[
"2026-03-11",
25.7,
"Nemotron 3 Super 120B A12B",
"NVIDIA",
1
],
[
"2026-03-15",
39.1,
"GLM-5-Turbo",
"Z.ai",
0
],
[
"2026-03-16",
8.6,
"NVIDIA Nemotron 3 Nano 4B",
"NVIDIA",
1
],
[
"2026-03-16",
19.7,
"Mistral Small 4",
"Mistral",
1
],
[
"2026-03-17",
39.7,
"GPT-5.4 nano",
"OpenAI",
0
],
[
"2026-03-17",
40.9,
"GPT-5.4 mini",
"OpenAI",
0
],
[
"2026-03-18",
38.9,
"MiniMax-M2.7",
"MiniMax",
1
],
[
"2026-03-18",
41.4,
"MiMo-V2-Pro",
"Xiaomi",
0
],
[
"2026-03-19",
17.8,
"Nemotron Cascade 2 30B A3B",
"NVIDIA",
1
],
[
"2026-03-19",
35.9,
"MiMo-V2-Omni",
"Xiaomi",
0
],
[
"2026-03-27",
34.5,
"KAT Coder Pro V2",
"KwaiKAT",
0
],
[
"2026-03-27",
37.3,
"MiMo-V2-Omni-0327",
"Xiaomi",
0
],
[
"2026-03-30",
19.2,
"Qwen3.5 Omni Flash",
"Alibaba",
0
],
[
"2026-03-30",
31.3,
"Qwen3.5 Omni Plus",
"Alibaba",
0
],
[
"2026-04-01",
18.4,
"Trinity Large Thinking",
"Arcee AI",
1
],
[
"2026-04-01",
35.3,
"GLM 5V Turbo",
"Z.ai",
0
],
[
"2026-04-02",
9.5,
"Gemma 4 E2B",
"Google",
1
],
[
"2026-04-02",
26.1,
"Gemma 4 26B A4B",
"Google",
1
],
[
"2026-04-02",
26.5,
"Step 3.5 Flash 2603",
"StepFun",
0
],
[
"2026-04-02",
29.7,
"Gemma 4 31B",
"Google",
1
],
[
"2026-04-02",
40.5,
"Qwen3.6 Plus",
"Alibaba",
0
],
[
"2026-04-03",
12.2,
"Gemma 4 E4B",
"Google",
1
],
[
"2026-04-06",
14.5,
"Solar Pro 3",
"Upstage",
0
],
[
"2026-04-07",
38.0,
"Grok 4.20 0309 v2",
"xAI",
0
],
[
"2026-04-07",
41.0,
"GLM-5.1",
"Z.ai",
1
],
[
"2026-04-08",
44.3,
"Muse Spark",
"Meta",
0
],
[
"2026-04-09",
20.5,
"EXAONE 4.5 33B",
"LG AI Research",
1
],
[
"2026-04-15",
18.8,
"JT-MINI",
"China Mobile",
0
],
[
"2026-04-16",
32.1,
"Qwen3.6 35B A3B",
"Alibaba",
1
],
[
"2026-04-16",
55.0,
"Claude Opus 4.7",
"Anthropic",
0
],
[
"2026-04-20",
41.1,
"Qwen3.6 Max Preview",
"Alibaba",
0
],
[
"2026-04-20",
45.1,
"Kimi K2.6",
"Kimi",
1
],
[
"2026-04-21",
14.2,
"Ling 2.6 Flash",
"InclusionAI",
1
],
[
"2026-04-22",
37.7,
"Qwen3.6 27B",
"Alibaba",
1
],
[
"2026-04-22",
38.0,
"MiMo-V2.5",
"Xiaomi",
1
],
[
"2026-04-22",
42.9,
"MiMo-V2.5-Pro",
"Xiaomi",
1
],
[
"2026-04-23",
26.6,
"Ling-2.6-1T",
"InclusionAI",
1
],
[
"2026-04-23",
34.4,
"Hy3-preview",
"Tencent",
1
],
[
"2026-04-23",
56.3,
"GPT-5.5",
"OpenAI",
0
],
[
"2026-04-24",
42.1,
"DeepSeek V4 Flash",
"DeepSeek",
1
],
[
"2026-04-24",
45.3,
"DeepSeek V4 Pro",
"DeepSeek",
1
],
[
"2026-04-29",
4.4,
"Granite 4.1 3B",
"IBM",
1
],
[
"2026-04-29",
6.4,
"Granite 4.1 8B",
"IBM",
1
],
[
"2026-04-29",
8.7,
"Granite 4.1 30B",
"IBM",
1
],
[
"2026-04-29",
15.0,
"Nemotron 3 Nano Omni 30B A3B Reasoning",
"NVIDIA",
1
],
[
"2026-04-29",
30.4,
"Mistral Medium 3.5",
"Mistral",
1
],
[
"2026-04-30",
37.9,
"Grok 4.3",
"xAI",
0
],
[
"2026-05-05",
34.3,
"GPT-5.5 Instant",
"OpenAI",
0
],
[
"2026-05-08",
31.3,
"Ring-2.6-1T",
"InclusionAI",
1
],
[
"2026-05-11",
3.8,
"MiniCPM-V 4.6 1.3B",
"OpenBMB",
1
],
[
"2026-05-14",
29.0,
"JT-35B-Flash",
"China Mobile",
0
],
[
"2026-05-19",
46.7,
"Qwen3.7 Max",
"Alibaba",
0
],
[
"2026-05-19",
52.0,
"Gemini 3.5 Flash",
"Google",
0
],
[
"2026-05-20",
22.8,
"Command A+",
"Cohere",
1
],
[
"2026-05-25",
11.9,
"MiniCPM5-1B",
"OpenBMB",
1
],
[
"2026-05-26",
18.3,
"HyperNova 60B 2605",
"Multiverse Computing",
1
],
[
"2026-05-28",
8.1,
"LFM2.5-8B-A1B",
"Liquid AI",
1
],
[
"2026-05-28",
57.3,
"Claude Opus 4.8",
"Anthropic",
0
],
[
"2026-05-29",
30.9,
"Step 3.7 Flash",
"StepFun",
1
],
[
"2026-06-01",
39.4,
"Qwen3.7 Plus",
"Alibaba",
0
],
[
"2026-06-01",
45.4,
"MiniMax-M3",
"MiniMax",
1
],
[
"2026-06-02",
42.1,
"Nex-N2-Pro",
"Nex AGI",
1
],
[
"2026-06-03",
22.2,
"Gemma 4 12B",
"Google",
1
],
[
"2026-06-04",
38.3,
"Nemotron 3 Ultra 550B A55B",
"NVIDIA",
1
],
[
"2026-06-09",
20.2,
"North Mini Code",
"Cohere",
1
],
[
"2026-06-09",
62.1,
"Claude Fable 5",
"Anthropic",
0
],
[
"2026-06-10",
13.5,
"DiffusionGemma 26B A4B",
"Google",
1
],
[
"2026-06-12",
43.0,
"Kimi K2.7 Code",
"Kimi",
1
],
[
"2026-06-16",
40.7,
"Grok Build 0.1 0616",
"xAI",
0
],
[
"2026-06-16",
52.6,
"GLM-5.2",
"Z.ai",
1
],
[
"2026-06-25",
29.2,
"GPT-5.5 Instant",
"OpenAI",
0
],
[
"2026-06-29",
34.3,
"LongCat 2.0",
"LongCat",
1
],
[
"2026-06-30",
55.3,
"Claude Sonnet 5",
"Anthropic",
0
],
[
"2026-07-06",
42.2,
"Hy3",
"Tencent",
1
],
[
"2026-07-08",
55.8,
"Grok 4.5",
"xAI",
0
],
[
"2026-07-09",
39.9,
"JT-4.1 Flash 236B A21B",
"China Mobile",
0
],
[
"2026-07-09",
52.3,
"GPT-5.6 Luna",
"OpenAI",
0
],
[
"2026-07-09",
53.2,
"Muse Spark 1.1",
"Meta",
0
],
[
"2026-07-09",
56.6,
"GPT-5.6 Terra",
"OpenAI",
0
],
[
"2026-07-09",
60.9,
"GPT-5.6 Sol",
"OpenAI",
0
],
[
"2026-07-14",
45.3,
"Motif 3",
"Motif Technologies",
0
],
[
"2026-07-15",
42.3,
"Inkling",
"Thinking Machines",
1
],
[
"2026-07-16",
59.7,
"Kimi K3",
"Kimi",
1
],
[
"2026-07-21",
37.4,
"Gemini 3.5 Flash-Lite",
"Google",
0
],
[
"2026-07-21",
51.6,
"Gemini 3.6 Flash",
"Google",
0
],
[
"2026-07-23",
16.2,
"G9v3-3B",
"AI9Stars",
1
],
[
"2026-07-24",
12.4,
"Celeris-1",
"Celeris",
0
],
[
"2026-07-24",
39.7,
"Agnes 2.5 Pro Alpha",
"Sapiens AI",
0
],
[
"2026-07-24",
63.1,
"Claude Opus 5",
"Anthropic",
0
],
[
"2026-07-30",
41.2,
"Inkling Small",
"Thinking Machines",
1
],
[
"2026-07-31",
51.8,
"DeepSeek V4 Flash 0731",
"DeepSeek",
1
],
[
"2026-08-03",
31.6,
"G9v3-39A5B",
"AI9Stars",
1
],
[
"2026-08-03",
58.1,
"Qwen3.8 Max",
"Alibaba",
0
],
[
"2026-08-04",
11.0,
"LFM2.5-2.6B",
"Liquid AI",
1
],
[
"2026-08-04",
37.8,
"Ling 3.0 Flash",
"InclusionAI",
1
],
[
"2026-08-05",
56.8,
"Muse Spark 1.2",
"Meta",
0
],
[
"2026-08-06",
24.5,
"Ling 3.0 Tiny",
"InclusionAI",
0
],
[
"2026-08-06",
41.6,
"Solar Pro 4",
"Upstage",
0
],
[
"2026-08-10",
35.1,
"Muse Glimmer",
"Meta",
1
],
[
"2026-08-10",
43.0,
"Quasar 438B",
"Multiverse Computing",
0
],
[
"2026-08-11",
23.6,
"Nemotron 3.5 Lightning",
"NVIDIA",
1
],
[
"2026-08-12",
31.0,
"K-EXAONE 2.0",
"LG AI Research",
1
],
[
"2026-08-12",
31.0,
"K-EXAONE 2.0 0803",
"LG AI Research",
1
],
[
"2026-08-12",
35.0,
"A.X-K2",
"SK Telecom",
1
],
[
"2026-08-12",
37.4,
"Solar Open2 250B",
"Upstage",
1
],
[
"2026-08-12",
47.4,
"Motif 3",
"Motif Technologies",
1
],
[
"2026-08-12",
57.7,
"Qwen3.8 2.4T A95B",
"Alibaba",
1
],
[
"2026-08-12",
60.9,
"Grok 4.6",
"xAI",
0
],
[
"2026-08-13",
53.2,
"DeepSeek V4 Pro 0813",
"DeepSeek",
1
],
[
"2026-08-13",
56.0,
"Gemini 3.7 Flash",
"Google",
0
],
[
"2026-08-14",
52.0,
"Qwen3.8 27B",
"Alibaba",
1
],
[
"2026-08-18",
59.5,
"GLM-5.3",
"Z.ai",
0
],
[
"2026-08-21",
51.5,
"DeepSeek V4 Flash Vision",
"DeepSeek",
0
],
[
"2026-08-25",
14.3,
"Granite 4.2 3B",
"IBM",
1
],
[
"2026-08-25",
19.6,
"Granite 4.2 8B",
"IBM",
1
],
[
"2026-08-25",
23.7,
"Granite 4.2 30B",
"IBM",
1
],
[
"2026-08-26",
49.1,
"Agnes 2.5 Pro Beta",
"Sapiens AI",
0
],
[
"2026-08-26",
55.8,
"Qwen3.8-Flash-Next",
"Alibaba",
1
],
[
"2026-08-26",
57.5,
"GLM-5.3-Flash",
"Z.ai",
1
],
[
"2026-08-30",
44.0,
"Apodex 1.1",
"Apodex",
0
],
[
"2026-09-01",
65.7,
"Claude Fable 5.1",
"Anthropic",
0
],
[
"2026-09-02",
58.7,
"Gemini 3.8 Flash",
"Google",
0
],
[
"2026-09-02",
62.1,
"Muse Spark 1.3",
"Meta",
0
],
[
"2026-09-03",
47.3,
"K2 Horizon 375B A23B",
"MBZUAI Institute of Foundation Models",
1
],
[
"2026-09-03",
61.2,
"GPT-6 Astra",
"OpenAI",
0
]
]

L = {'ylabel': 'Intelligence Index (Artificial Analysis)', 'title': 'AI Model Capability Over Time — A Single Number', 'sub': 'independent measurement of {n} models ({lo} – {hi})  ·  yellow: best available  ·  green dashed: best open-weights', 'growth': 'Last 12 months\n{a:.0f} → {b:.0f}  ({k:.1f}x)\nBest open-weights: {ao:.0f}  (gap {fark:.1f})', 'cloud': 'other measured models', 'front_all': 'best at the time', 'front_open': 'best open-weights', 'credit': 'Source: artificialanalysis.ai  ·  Compiled by Prof. Dr. Oğuz Ergin'}

COLORS = {"Anthropic": "#d97757", "OpenAI": "#10a37f", "Google": "#4285F4", "xAI": "#1da1f2",
          "Meta": "#0668E1", "DeepSeek": "#ef4444", "Alibaba": "#7C3AED", "Moonshot": "#14B8A6",
          "Z.ai": "#BE185D", "MiniMax": "#C77DFF", "Mistral": "#fa8005", "ByteDance": "#22D3EE",
          "Microsoft": "#F25022", "Amazon": "#ff9900", "NVIDIA": "#76b900"}
OTHER = "#4a5160"

df = pd.DataFrame(DATA, columns=["date", "ii", "name", "comp", "open"])
df["Date"] = pd.to_datetime(df["date"])
df = df.sort_values("Date").reset_index(drop=True)

# --- sinir: o gune kadarki en iyi ---
front, best = [], -1
for _, r in df.iterrows():
    if r["ii"] > best:
        best = r["ii"]
        if front and front[-1]["Date"] == r["Date"]:
            front[-1] = r
        else:
            front.append(r)
fr = pd.DataFrame(front).reset_index(drop=True)

# --- acik agirlik siniri ---
fo, best_o = [], -1
for _, r in df[df["open"] == 1].iterrows():
    if r["ii"] > best_o:
        best_o = r["ii"]
        if fo and fo[-1]["Date"] == r["Date"]: fo[-1] = r
        else: fo.append(r)
fro = pd.DataFrame(fo).reset_index(drop=True)
OPEN_C = "#3fb950"

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")

# arka plan bulutu
ax.scatter(df["Date"], df["ii"], s=34, c="#262c36", alpha=.85, edgecolors="none", zorder=1)

# sinir merdiveni
ax.step(fr["Date"], fr["ii"], where="post", color="#ffd166", lw=3.0, zorder=3, alpha=.95)
ax.fill_between(fr["Date"], fr["ii"], step="post", color="#ffd166", alpha=.05, zorder=2)
ax.step(fro["Date"], fro["ii"], where="post", color=OPEN_C, lw=2.6, zorder=3, alpha=.95, linestyle=(0, (6, 2)))
for _, r in fr.iterrows():
    ax.scatter([r["Date"]], [r["ii"]], s=230, c=COLORS.get(r["comp"], OTHER),
               edgecolors="white", linewidths=2.0, zorder=5)
for _, r in fro.iterrows():
    ax.scatter([r["Date"]], [r["ii"]], s=150, c=COLORS.get(r["comp"], OTHER),
               edgecolors=OPEN_C, linewidths=2.4, zorder=4)

# sinir etiketleri: PIKSEL uzayinda 2 boyutlu cakisma kontrolu
# (kademe farki tek basina yetmiyor: noktalarin kendi yuksekligi de degisiyor)
PX_DAY = (26 * 105 * 0.93) / max(1, (df["Date"].max() - df["Date"].min()).days)
YLIM = fr["ii"].max() * 1.20
PX_UNIT = (14 * 105 * 0.78) / YLIM
TIERS = [26, -34, 66, -74, 106, -114, 146, -154, 186, -194, 226, -234]
x0 = df["Date"].min().toordinal()
boxes = []
_seen = set()
_items = [(r, False) for _, r in fr.iterrows()] + [(r, True) for _, r in fro.iterrows()]
_items = [(r, o) for r, o in _items if not ((r["Date"], r["name"]) in _seen or _seen.add((r["Date"], r["name"])))]
_items.sort(key=lambda x: x[0]["Date"])
for r, _is_open in _items:
    cx = (r["Date"].toordinal() - x0) * PX_DAY
    hw = len(r["name"]) * 4.5 + 17
    tier, en_iyi = None, (-1, TIERS[-1])
    for t in TIERS:
        cy = r["ii"] * PX_UNIT + t
        if all(abs(cx - bx) > (hw + bw) or abs(cy - by) > 58 for bx, by, bw in boxes):
            tier = t; break
        # yer yoksa: en az cakisan kademeyi akilda tut
        pay = min(((abs(cx - bx) - (hw + bw)) if abs(cy - by) <= 58 else 9999) for bx, by, bw in boxes)
        if pay > en_iyi[0]: en_iyi = (pay, t)
    if tier is None: tier = en_iyi[1]
    boxes.append((cx, r["ii"] * PX_UNIT + tier, hw))
    ax.annotate(r["name"], (r["Date"], r["ii"]), xytext=(0, tier), textcoords="offset points",
                fontsize=12, color="#e6edf3", fontweight="bold", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.30", fc="#161b22",
                          ec=(OPEN_C if _is_open else COLORS.get(r["comp"], OTHER)), lw=1.6, alpha=.96),
                arrowprops=dict(arrowstyle="-", color=(OPEN_C if _is_open else COLORS.get(r["comp"], OTHER)),
                                lw=1.1, alpha=.55, shrinkA=2, shrinkB=6))

ax.set_ylabel(L["ylabel"], fontsize=17, color="#8b949e", labelpad=16)
ax.set_ylim(0, YLIM)
ax.grid(True, axis="y", color="#21262d", lw=1.0)
ax.grid(True, axis="x", color="#161b22", lw=.7)
for s in ax.spines.values(): s.set_color("#30363d")
ax.tick_params(colors="#8b949e", labelsize=14)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

# baslik + alt baslik
son = fr.iloc[-1]
bir_yil = fr[fr["Date"] <= son["Date"] - pd.Timedelta(days=365)]["ii"].max()
plt.title(L["title"], fontsize=30, color="white", pad=54, fontweight="bold")
ax.text(0.5, 1.045, L["sub"].format(n=len(df), lo=df["Date"].min().strftime("%b %Y"),
                                    hi=df["Date"].max().strftime("%b %Y")),
        transform=ax.transAxes, ha="center", fontsize=15, color="#8b949e", style="italic")

# buyume kutusu
son_o = fro.iloc[-1]
ax.text(0.015, 0.965, L["growth"].format(a=bir_yil, b=son["ii"], k=son["ii"] / bir_yil,
                                         ao=son_o["ii"], fark=son["ii"] - son_o["ii"]),
        transform=ax.transAxes, ha="left", va="top", fontsize=17, color="#ffd166", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", fc="#161b22", ec="#ffd166", lw=1.8, alpha=.95))

# lejant (sinirdaki sirketler)
import matplotlib.lines as mlines
comps = list(dict.fromkeys(fr["comp"]))
handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=13, markerfacecolor=COLORS.get(c, OTHER),
                         markeredgecolor="white", label=c) for c in comps]
handles.append(mlines.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="#262c36",
                             markeredgecolor="none", label=L["cloud"]))
handles.insert(0, mlines.Line2D([], [], color="#ffd166", lw=3, label=L["front_all"]))
handles.insert(1, mlines.Line2D([], [], color=OPEN_C, lw=2.6, linestyle=(0, (6, 2)), label=L["front_open"]))
ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#161b22", edgecolor="#30363d",
          fontsize=14, labelcolor="#c9d1d9", ncol=2)

ax.text(0.995, -0.115, L["credit"], transform=ax.transAxes, ha="right", fontsize=13,
        color="#6e7681", style="italic")
plt.tight_layout()
plt.savefig("intelligence_index.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: intelligence_index.png  |  model:", len(df), " sinir:", len(fr))
