# -*- coding: utf-8 -*-
# Yapay zeka basarimi tek sayiyla: Artificial Analysis Zeka Endeksi (Intelligence Index)
# Veri kaynagi: artificialanalysis.ai  (472 model, 2022-11-30 - 2026-09-03)
# Bu dosya scratchpad/make_ii_scripts.py ile uretildi; veri asagida GOMULU.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = [
[
"2022-11-30",
5.5,
"GPT-3.5 Turbo",
"OpenAI",
0
],
[
"2023-02-24",
5.0,
"Llama 65B",
"Meta",
1
],
[
"2023-03-14",
5.0,
"Claude Instant",
"Anthropic",
0
],
[
"2023-03-14",
6.7,
"GPT-4",
"OpenAI",
0
],
[
"2023-05-10",
5.4,
"PALM-2",
"Google",
0
],
[
"2023-07-11",
5.5,
"Claude 2.0",
"Anthropic",
0
],
[
"2023-07-18",
5.3,
"Llama 2 Chat 13B",
"Meta",
1
],
[
"2023-07-18",
5.3,
"Llama 2 Chat 70B",
"Meta",
1
],
[
"2023-07-18",
5.7,
"Llama 2 Chat 7B",
"Meta",
1
],
[
"2023-09-25",
5.0,
"Qwen Chat 14B",
"Alibaba",
1
],
[
"2023-09-27",
5.0,
"Mistral 7B Instruct",
"Mistral",
1
],
[
"2023-11-06",
7.0,
"GPT-4 Turbo",
"OpenAI",
0
],
[
"2023-11-21",
5.6,
"Claude 2.1",
"Anthropic",
0
],
[
"2023-11-29",
5.3,
"DeepSeek LLM 67B Chat",
"DeepSeek",
1
],
[
"2023-11-30",
5.4,
"Qwen Chat 72B",
"Alibaba",
1
],
[
"2023-12-06",
5.3,
"Gemini 1.0 Pro",
"Google",
0
],
[
"2023-12-06",
5.8,
"Gemini 1.0 Ultra",
"Google",
0
],
[
"2023-12-11",
5.1,
"Mixtral 8x7B Instruct",
"Mistral",
1
],
[
"2023-12-11",
5.5,
"Mistral Medium",
"Mistral",
0
],
[
"2023-12-18",
5.3,
"OpenChat 3.5",
"OpenChat",
1
],
[
"2024-01-25",
6.4,
"Solar Mini",
"Upstage",
1
],
[
"2024-02-26",
5.5,
"Mistral Small",
"Mistral",
0
],
[
"2024-02-26",
5.8,
"Mistral Large",
"Mistral",
0
],
[
"2024-02-26",
6.3,
"Phi-4 Mini Instruct",
"Microsoft",
1
],
[
"2024-03-04",
5.6,
"Claude 3 Haiku",
"Anthropic",
0
],
[
"2024-03-04",
5.9,
"Claude 3 Sonnet",
"Anthropic",
0
],
[
"2024-03-04",
8.7,
"Claude 3 Opus",
"Anthropic",
0
],
[
"2024-03-12",
5.0,
"Command-R",
"Cohere",
1
],
[
"2024-03-17",
6.3,
"Grok-1",
"xAI",
1
],
[
"2024-03-27",
5.3,
"DBRX Instruct",
"Databricks",
1
],
[
"2024-04-04",
5.3,
"Command-R+",
"Cohere",
1
],
[
"2024-04-17",
5.7,
"Mixtral 8x22B Instruct",
"Mistral",
1
],
[
"2024-04-18",
4.8,
"Llama 3 Instruct 8B",
"Meta",
1
],
[
"2024-04-18",
5.5,
"Llama 3 Instruct 70B",
"Meta",
1
],
[
"2024-04-23",
5.8,
"Phi-3 Mini Instruct 3.8B",
"Microsoft",
1
],
[
"2024-04-24",
5.4,
"Arctic Instruct",
"Snowflake",
1
],
[
"2024-04-25",
5.7,
"Qwen1.5 Chat 110B",
"Alibaba",
1
],
[
"2024-05-06",
5.5,
"DeepSeek-V2-Chat",
"DeepSeek",
1
],
[
"2024-05-13",
7.3,
"GPT-4o",
"OpenAI",
0
],
[
"2024-05-14",
5.9,
"Gemini 1.5 Flash",
"Google",
0
],
[
"2024-05-15",
6.4,
"Gemini 1.5 Pro",
"Google",
0
],
[
"2024-06-07",
6.3,
"Qwen2 Instruct 72B",
"Alibaba",
1
],
[
"2024-06-17",
5.3,
"DeepSeek Coder V2 Lite Instruct",
"DeepSeek",
1
],
[
"2024-06-17",
6.0,
"DeepSeek-Coder-V2",
"DeepSeek",
1
],
[
"2024-06-21",
7.2,
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
6.6,
"Llama 3.1 Instruct 70B",
"Meta",
1
],
[
"2024-07-23",
6.9,
"Llama 3.1 Instruct 8B",
"Meta",
1
],
[
"2024-07-23",
7.3,
"Llama 3.1 Instruct 405B",
"Meta",
1
],
[
"2024-07-24",
6.8,
"Mistral Large 2",
"Mistral",
1
],
[
"2024-08-06",
7.7,
"GPT-4o",
"OpenAI",
0
],
[
"2024-08-13",
6.9,
"Grok Beta",
"xAI",
0
],
[
"2024-08-15",
6.0,
"Hermes 3 - Llama-3.1 70B",
"Nous Research",
1
],
[
"2024-08-22",
5.2,
"Jamba 1.5 Mini",
"AI21 Labs",
1
],
[
"2024-08-22",
6.0,
"Jamba 1.5 Large",
"AI21 Labs",
1
],
[
"2024-09-06",
6.6,
"DeepSeek-V2.5",
"DeepSeek",
1
],
[
"2024-09-12",
9.8,
"o1-mini",
"OpenAI",
0
],
[
"2024-09-12",
11.4,
"o1-preview",
"OpenAI",
0
],
[
"2024-09-17",
5.8,
"Mistral Small",
"Mistral",
1
],
[
"2024-09-19",
5.8,
"Qwen2.5 Coder Instruct 7B",
"Alibaba",
1
],
[
"2024-09-19",
6.9,
"Qwen2.5 Instruct 32B",
"Alibaba",
1
],
[
"2024-09-19",
7.7,
"Qwen2.5 Instruct 72B",
"Alibaba",
1
],
[
"2024-09-24",
7.1,
"Gemini 1.5 Flash",
"Google",
0
],
[
"2024-09-24",
7.9,
"Gemini 1.5 Pro",
"Google",
0
],
[
"2024-09-25",
4.8,
"Llama 3.2 Instruct 1B",
"Meta",
1
],
[
"2024-09-25",
5.4,
"Llama 3.2 Instruct 11B",
"Meta",
1
],
[
"2024-09-25",
5.6,
"Molmo 7B-D",
"Allen Institute for AI",
1
],
[
"2024-09-25",
5.7,
"Llama 3.2 Instruct 3B",
"Meta",
1
],
[
"2024-09-25",
6.4,
"Llama 3.2 Instruct 90B",
"Meta",
1
],
[
"2024-09-30",
5.4,
"LFM 40B",
"Liquid AI",
0
],
[
"2024-10-03",
6.2,
"Gemini 1.5 Flash-8B",
"Google",
0
],
[
"2024-10-04",
6.4,
"Reka Flash",
"Reka AI",
0
],
[
"2024-10-15",
6.9,
"Llama 3.1 Nemotron Instruct 70B",
"NVIDIA",
1
],
[
"2024-10-22",
7.9,
"Claude 3.5 Sonnet",
"Anthropic",
0
],
[
"2024-10-22",
8.9,
"Claude 3.5 Haiku",
"Anthropic",
0
],
[
"2024-11-11",
6.7,
"Qwen2.5 Coder Instruct 32B",
"Alibaba",
1
],
[
"2024-11-18",
6.4,
"Qwen2.5 Turbo",
"Alibaba",
0
],
[
"2024-11-18",
7.1,
"Pixtral Large",
"Mistral",
1
],
[
"2024-11-18",
7.6,
"Mistral Large 2",
"Mistral",
1
],
[
"2024-11-20",
8.4,
"GPT-4o",
"OpenAI",
0
],
[
"2024-11-26",
5.6,
"OLMo 2 7B",
"Allen Institute for AI",
1
],
[
"2024-11-27",
7.6,
"QwQ 32B-Preview",
"Alibaba",
1
],
[
"2024-12-03",
5.9,
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
7.0,
"Nova Pro",
"Amazon",
0
],
[
"2024-12-05",
15.2,
"o1",
"OpenAI",
0
],
[
"2024-12-06",
7.7,
"Llama 3.3 Instruct 70B",
"Meta",
1
],
[
"2024-12-10",
6.6,
"DeepSeek-V2.5",
"DeepSeek",
1
],
[
"2024-12-11",
8.2,
"Gemini 2.0 Flash",
"Google",
0
],
[
"2024-12-12",
5.9,
"Phi-4",
"Microsoft",
1
],
[
"2024-12-12",
7.1,
"Grok 2",
"xAI",
1
],
[
"2024-12-19",
6.6,
"Gemini 2.0 Flash Thinking Experimental",
"Google",
0
],
[
"2024-12-26",
8.5,
"DeepSeek V3",
"DeepSeek",
1
],
[
"2025-01-20",
5.5,
"DeepSeek R1 Distill Qwen 1.5B",
"DeepSeek",
1
],
[
"2025-01-20",
6.5,
"DeepSeek R1 Distill Llama 8B",
"DeepSeek",
1
],
[
"2025-01-20",
7.8,
"DeepSeek R1 Distill Qwen 14B",
"DeepSeek",
1
],
[
"2025-01-20",
7.9,
"DeepSeek R1 Distill Llama 70B",
"DeepSeek",
1
],
[
"2025-01-20",
8.4,
"DeepSeek R1 Distill Qwen 32B",
"DeepSeek",
1
],
[
"2025-01-20",
11.4,
"DeepSeek R1",
"DeepSeek",
1
],
[
"2025-01-21",
7.6,
"Sonar Pro",
"Perplexity",
0
],
[
"2025-01-21",
7.7,
"Sonar",
"Perplexity",
0
],
[
"2025-01-21",
9.4,
"Gemini 2.0 Flash Thinking Experimental",
"Google",
0
],
[
"2025-01-28",
8.0,
"Qwen2.5 Max",
"Alibaba",
0
],
[
"2025-01-28",
8.7,
"Sonar Reasoning",
"Perplexity",
0
],
[
"2025-01-28",
11.8,
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
7.2,
"Llama 3.1 Tulu3 405B",
"Allen Institute for AI",
1
],
[
"2025-01-31",
12.5,
"o3-mini",
"OpenAI",
0
],
[
"2025-02-05",
7.3,
"Gemini 2.0 Flash-Lite",
"Google",
0
],
[
"2025-02-05",
8.7,
"Gemini 2.0 Pro Experimental",
"Google",
0
],
[
"2025-02-05",
8.9,
"Gemini 2.0 Flash",
"Google",
0
],
[
"2025-02-13",
5.1,
"DeepHermes 3 - Llama-3.1 8B Preview",
"Nous Research",
1
],
[
"2025-02-15",
7.2,
"GPT-4o",
"OpenAI",
0
],
[
"2025-02-17",
6.5,
"Mistral Saba",
"Mistral",
0
],
[
"2025-02-18",
6.4,
"R1 1776",
"Perplexity",
1
],
[
"2025-02-19",
10.4,
"Grok 3 Reasoning Beta",
"xAI",
0
],
[
"2025-02-19",
12.1,
"Grok 3",
"xAI",
0
],
[
"2025-02-19",
14.6,
"Grok 3 mini Reasoning",
"xAI",
0
],
[
"2025-02-24",
17.7,
"Claude 3.7 Sonnet",
"Anthropic",
0
],
[
"2025-02-25",
7.4,
"Gemini 2.0 Flash-Lite",
"Google",
0
],
[
"2025-02-26",
5.8,
"Phi-4 Multimodal Instruct",
"Microsoft",
1
],
[
"2025-02-27",
9.6,
"GPT-4.5",
"OpenAI",
0
],
[
"2025-03-05",
9.5,
"QwQ 32B",
"Alibaba",
1
],
[
"2025-03-06",
5.2,
"Jamba 1.6 Mini",
"AI21 Labs",
1
],
[
"2025-03-06",
6.0,
"Jamba 1.6 Large",
"AI21 Labs",
1
],
[
"2025-03-10",
5.6,
"Reka Flash 3",
"Reka AI",
1
],
[
"2025-03-12",
3.8,
"Gemma 3 12B Instruct",
"Google",
1
],
[
"2025-03-12",
4.8,
"Gemma 3 4B Instruct",
"Google",
1
],
[
"2025-03-12",
4.9,
"Gemma 3 27B Instruct",
"Google",
1
],
[
"2025-03-13",
4.8,
"Gemma 3 1B Instruct",
"Google",
1
],
[
"2025-03-13",
6.0,
"OLMo 2 32B",
"Allen Institute for AI",
1
],
[
"2025-03-13",
6.1,
"DeepHermes 3 - Mistral 24B Preview",
"Nous Research",
1
],
[
"2025-03-13",
7.0,
"Command A",
"Cohere",
1
],
[
"2025-03-17",
7.4,
"Mistral Small 3.1",
"Mistral",
1
],
[
"2025-03-18",
8.9,
"Llama 3.3 Nemotron Super 49B v1",
"NVIDIA",
1
],
[
"2025-03-19",
12.4,
"o1-pro",
"OpenAI",
0
],
[
"2025-03-25",
9.7,
"DeepSeek V3 0324",
"DeepSeek",
1
],
[
"2025-03-25",
15.0,
"Gemini 2.5 Pro Preview",
"Google",
0
],
[
"2025-03-27",
9.0,
"GPT-4o",
"OpenAI",
0
],
[
"2025-04-05",
6.5,
"Llama 4 Scout",
"Meta",
1
],
[
"2025-04-05",
9.3,
"Llama 4 Maverick",
"Meta",
1
],
[
"2025-04-07",
7.5,
"Llama 3.1 Nemotron Ultra 253B v1",
"NVIDIA",
1
],
[
"2025-04-14",
7.8,
"GPT-4.1 nano",
"OpenAI",
0
],
[
"2025-04-14",
10.2,
"GPT-4.1 mini",
"OpenAI",
0
],
[
"2025-04-14",
12.7,
"GPT-4.1",
"OpenAI",
0
],
[
"2025-04-16",
4.9,
"Granite 3.3 8B",
"IBM",
1
],
[
"2025-04-16",
16.7,
"o4-mini",
"OpenAI",
0
],
[
"2025-04-16",
20.2,
"o3",
"OpenAI",
0
],
[
"2025-04-17",
11.7,
"Gemini 2.5 Flash Preview",
"Google",
0
],
[
"2025-04-28",
4.8,
"Qwen3 0.6B",
"Alibaba",
1
],
[
"2025-04-28",
5.2,
"Qwen3 1.7B",
"Alibaba",
1
],
[
"2025-04-28",
6.0,
"Qwen3 8B",
"Alibaba",
1
],
[
"2025-04-28",
6.7,
"Qwen3 14B",
"Alibaba",
1
],
[
"2025-04-28",
7.2,
"Qwen3 4B",
"Alibaba",
1
],
[
"2025-04-28",
7.3,
"Qwen3 32B",
"Alibaba",
1
],
[
"2025-04-28",
7.6,
"Qwen3 30B A3B",
"Alibaba",
1
],
[
"2025-04-28",
9.5,
"Qwen3 235B A22B",
"Alibaba",
1
],
[
"2025-04-30",
9.2,
"Nova Premier",
"Amazon",
0
],
[
"2025-05-06",
14.5,
"Gemini 2.5 Pro Preview",
"Google",
0
],
[
"2025-05-07",
9.0,
"Mistral Medium 3",
"Mistral",
0
],
[
"2025-05-20",
5.8,
"Gemma 3n E4B Instruct Preview",
"Google",
1
],
[
"2025-05-20",
7.3,
"Llama 3.1 Nemotron Nano 4B v1.1",
"NVIDIA",
1
],
[
"2025-05-20",
9.1,
"Solar Pro 2",
"Upstage",
0
],
[
"2025-05-20",
13.1,
"Gemini 2.5 Flash",
"Google",
0
],
[
"2025-05-21",
8.7,
"Devstral Small",
"Mistral",
1
],
[
"2025-05-22",
18.9,
"Claude 4 Sonnet",
"Anthropic",
0
],
[
"2025-05-22",
20.6,
"Claude 4 Opus",
"Anthropic",
0
],
[
"2025-05-23",
5.3,
"Sarvam M",
"Sarvam",
1
],
[
"2025-05-28",
13.1,
"DeepSeek R1 0528",
"DeepSeek",
1
],
[
"2025-05-29",
8.1,
"DeepSeek R1 0528 Qwen3 8B",
"DeepSeek",
1
],
[
"2025-06-05",
16.7,
"Gemini 2.5 Pro",
"Google",
0
],
[
"2025-06-10",
8.2,
"Magistral Small 1",
"Mistral",
1
],
[
"2025-06-10",
9.1,
"Magistral Medium 1",
"Mistral",
0
],
[
"2025-06-10",
21.9,
"o3-pro",
"OpenAI",
0
],
[
"2025-06-17",
8.5,
"Gemini 2.5 Flash-Lite",
"Google",
0
],
[
"2025-06-17",
10.0,
"MiniMax M1 40k",
"MiniMax",
1
],
[
"2025-06-17",
11.7,
"MiniMax M1 80k",
"MiniMax",
1
],
[
"2025-06-20",
7.0,
"Mistral Small 3.2",
"Mistral",
1
],
[
"2025-06-26",
4.8,
"Gemma 3n E2B Instruct",
"Google",
1
],
[
"2025-06-26",
4.8,
"Gemma 3n E4B Instruct",
"Google",
1
],
[
"2025-06-30",
7.5,
"ERNIE 4.5 300B A47B",
"Baidu",
1
],
[
"2025-07-07",
5.2,
"Jamba 1.7 Mini",
"AI21 Labs",
1
],
[
"2025-07-07",
6.1,
"Jamba 1.7 Large",
"AI21 Labs",
1
],
[
"2025-07-09",
7.5,
"Solar Pro 2",
"Upstage",
0
],
[
"2025-07-10",
4.8,
"LFM2 1.2B",
"Liquid AI",
1
],
[
"2025-07-10",
7.6,
"Devstral Small",
"Mistral",
1
],
[
"2025-07-10",
9.0,
"Devstral Medium",
"Mistral",
0
],
[
"2025-07-10",
22.5,
"Grok 4",
"xAI",
0
],
[
"2025-07-11",
12.7,
"Kimi K2",
"Kimi",
1
],
[
"2025-07-15",
5.3,
"Exaone 4.0 1.2B",
"LG AI Research",
1
],
[
"2025-07-15",
8.2,
"EXAONE 4.0 32B",
"LG AI Research",
1
],
[
"2025-07-21",
12.0,
"Qwen3 235B A22B 2507 Instruct",
"Alibaba",
1
],
[
"2025-07-22",
11.9,
"Qwen3 Coder 480B A35B Instruct",
"Alibaba",
1
],
[
"2025-07-25",
9.0,
"Llama Nemotron Super 49B v1.5",
"NVIDIA",
1
],
[
"2025-07-25",
12.7,
"Qwen3 235B A22B 2507",
"Alibaba",
1
],
[
"2025-07-28",
11.1,
"GLM-4.5-Air",
"Z.ai",
1
],
[
"2025-07-28",
12.8,
"GLM-4.5",
"Z.ai",
1
],
[
"2025-07-29",
7.5,
"Qwen3 30B A3B 2507 Instruct",
"Alibaba",
1
],
[
"2025-07-30",
9.8,
"Qwen3 30B A3B 2507",
"Alibaba",
1
],
[
"2025-07-31",
9.6,
"Qwen3 Coder 30B A3B Instruct",
"Alibaba",
1
],
[
"2025-08-05",
10.0,
"gpt-oss-20b",
"OpenAI",
1
],
[
"2025-08-05",
12.3,
"gpt-oss-120b",
"OpenAI",
1
],
[
"2025-08-05",
22.8,
"Claude 4.1 Opus",
"Anthropic",
0
],
[
"2025-08-06",
6.7,
"Qwen3 4B 2507 Instruct",
"Alibaba",
1
],
[
"2025-08-06",
8.8,
"Qwen3 4B 2507",
"Alibaba",
1
],
[
"2025-08-07",
13.0,
"GPT-5 nano",
"OpenAI",
0
],
[
"2025-08-07",
20.6,
"GPT-5 mini",
"OpenAI",
0
],
[
"2025-08-07",
23.0,
"GPT-5",
"OpenAI",
0
],
[
"2025-08-11",
7.6,
"GLM-4.5V",
"Z.ai",
1
],
[
"2025-08-12",
9.9,
"Mistral Medium 3.1",
"Mistral",
0
],
[
"2025-08-14",
5.1,
"Gemma 3 270M",
"Google",
1
],
[
"2025-08-18",
7.4,
"NVIDIA Nemotron Nano 9B V2",
"NVIDIA",
1
],
[
"2025-08-20",
12.1,
"Seed-OSS-36B-Instruct",
"ByteDance",
1
],
[
"2025-08-21",
13.7,
"DeepSeek V3.1",
"DeepSeek",
1
],
[
"2025-08-27",
7.5,
"Hermes 4 - Llama-3.1 405B",
"Nous Research",
1
],
[
"2025-08-27",
7.9,
"Hermes 4 - Llama-3.1 70B",
"Nous Research",
1
],
[
"2025-08-28",
14.1,
"Grok Code Fast 1",
"xAI",
0
],
[
"2025-09-02",
4.8,
"Apertus 8B Instruct",
"Swiss AI Initiative",
1
],
[
"2025-09-02",
5.1,
"Apertus 70B Instruct",
"Swiss AI Initiative",
1
],
[
"2025-09-05",
12.6,
"Qwen3 Max",
"Alibaba",
0
],
[
"2025-09-05",
15.3,
"Kimi K2 0905",
"Kimi",
1
],
[
"2025-09-08",
10.4,
"Gemini 2.5 Flash-Lite Preview",
"Google",
0
],
[
"2025-09-09",
5.5,
"Ling-mini-2.0",
"InclusionAI",
1
],
[
"2025-09-11",
9.6,
"Qwen3 Next 80B A3B Instruct",
"Alibaba",
1
],
[
"2025-09-11",
11.2,
"Qwen3 Next 80B A3B",
"Alibaba",
1
],
[
"2025-09-17",
7.8,
"Ling-flash-2.0",
"InclusionAI",
1
],
[
"2025-09-17",
8.6,
"Magistral Small 1.2",
"Mistral",
1
],
[
"2025-09-18",
11.8,
"Magistral Medium 1.2",
"Mistral",
0
],
[
"2025-09-19",
7.2,
"Ring-flash-2.0",
"InclusionAI",
1
],
[
"2025-09-19",
17.9,
"Grok 4 Fast",
"xAI",
0
],
[
"2025-09-22",
5.1,
"Granite 4.0 Micro",
"IBM",
1
],
[
"2025-09-22",
6.0,
"Granite 4.0 H Small",
"IBM",
1
],
[
"2025-09-22",
6.0,
"Qwen3 Omni 30B A3B Instruct",
"Alibaba",
1
],
[
"2025-09-22",
7.8,
"Qwen3 Omni 30B A3B",
"Alibaba",
1
],
[
"2025-09-22",
15.4,
"DeepSeek V3.1 Terminus",
"DeepSeek",
1
],
[
"2025-09-23",
5.2,
"LFM2 2.6B",
"Liquid AI",
1
],
[
"2025-09-23",
9.9,
"Qwen3 VL 235B A22B Instruct",
"Alibaba",
1
],
[
"2025-09-23",
13.4,
"Qwen3 VL 235B A22B",
"Alibaba",
1
],
[
"2025-09-23",
15.6,
"Qwen3 Max",
"Alibaba",
0
],
[
"2025-09-23",
24.9,
"GPT-5 Codex",
"OpenAI",
0
],
[
"2025-09-25",
9.3,
"Gemini 2.5 Flash-Lite Preview",
"Google",
0
],
[
"2025-09-25",
15.5,
"Gemini 2.5 Flash Preview",
"Google",
0
],
[
"2025-09-29",
16.6,
"DeepSeek V3.2 Exp",
"DeepSeek",
1
],
[
"2025-09-29",
21.2,
"Claude 4.5 Sonnet",
"Anthropic",
0
],
[
"2025-09-30",
13.8,
"Apriel-v1.5-15B-Thinker",
"ServiceNow",
1
],
[
"2025-09-30",
18.5,
"GLM-4.6",
"Z.ai",
1
],
[
"2025-10-03",
7.9,
"Qwen3 VL 30B A3B Instruct",
"Alibaba",
1
],
[
"2025-10-03",
9.5,
"Qwen3 VL 30B A3B",
"Alibaba",
1
],
[
"2025-10-07",
4.9,
"LFM2 8B A1B",
"Liquid AI",
1
],
[
"2025-10-08",
5.7,
"Jamba Reasoning 3B",
"AI21 Labs",
1
],
[
"2025-10-08",
9.2,
"Ling-1T",
"InclusionAI",
1
],
[
"2025-10-13",
10.9,
"Ring-1T",
"InclusionAI",
1
],
[
"2025-10-14",
5.7,
"Qwen3 VL 4B Instruct",
"Alibaba",
1
],
[
"2025-10-14",
7.0,
"Qwen3 VL 4B",
"Alibaba",
1
],
[
"2025-10-14",
7.3,
"Qwen3 VL 8B Instruct",
"Alibaba",
1
],
[
"2025-10-14",
8.2,
"Qwen3 VL 8B",
"Alibaba",
1
],
[
"2025-10-15",
17.6,
"Claude 4.5 Haiku",
"Anthropic",
0
],
[
"2025-10-21",
8.4,
"Qwen3 VL 32B Instruct",
"Alibaba",
1
],
[
"2025-10-21",
11.9,
"Qwen3 VL 32B",
"Alibaba",
1
],
[
"2025-10-26",
18.6,
"MiniMax-M2",
"MiniMax",
1
],
[
"2025-10-28",
4.8,
"Granite 4.0 350M",
"IBM",
1
],
[
"2025-10-28",
4.8,
"Granite 4.0 H 350M",
"IBM",
1
],
[
"2025-10-28",
5.0,
"Granite 4.0 1B",
"IBM",
1
],
[
"2025-10-28",
5.2,
"Granite 4.0 H 1B",
"IBM",
1
],
[
"2025-10-28",
7.5,
"NVIDIA Nemotron Nano 12B v2 VL",
"NVIDIA",
1
],
[
"2025-10-29",
13.4,
"Nova 2.0 Lite",
"Amazon",
0
],
[
"2025-10-30",
7.3,
"Kimi Linear 48B A3B Instruct",
"Kimi",
1
],
[
"2025-11-03",
16.3,
"Qwen3 Max Thinking",
"Alibaba",
0
],
[
"2025-11-06",
22.0,
"Kimi K2 Thinking",
"Kimi",
1
],
[
"2025-11-11",
16.9,
"Doubao Seed Code",
"ByteDance",
0
],
[
"2025-11-11",
18.6,
"KAT-Coder-Pro V1",
"KwaiKAT",
0
],
[
"2025-11-13",
14.3,
"ERNIE 5.0 Thinking Preview",
"Baidu",
0
],
[
"2025-11-13",
20.4,
"GPT-5.1 Codex mini",
"OpenAI",
0
],
[
"2025-11-13",
23.7,
"GPT-5.1 Codex",
"OpenAI",
0
],
[
"2025-11-13",
24.7,
"GPT-5.1",
"OpenAI",
0
],
[
"2025-11-18",
28.0,
"Gemini 3 Pro Preview",
"Google",
0
],
[
"2025-11-19",
20.4,
"Grok 4.1 Fast",
"xAI",
0
],
[
"2025-11-20",
5.2,
"Olmo 3 7B Instruct",
"Allen Institute for AI",
1
],
[
"2025-11-20",
5.6,
"Olmo 3 7B Think",
"Allen Institute for AI",
1
],
[
"2025-11-20",
6.5,
"Olmo 3 32B Think",
"Allen Institute for AI",
1
],
[
"2025-11-24",
29.1,
"Claude Opus 4.5",
"Anthropic",
0
],
[
"2025-11-25",
13.4,
"Apriel-v1.6-15B-Thinker",
"ServiceNow",
1
],
[
"2025-11-26",
13.6,
"Nova 2.0 Omni",
"Amazon",
0
],
[
"2025-11-27",
10.6,
"INTELLECT-3",
"Prime Intellect",
1
],
[
"2025-11-27",
14.2,
"Nova 2.0 Pro Preview",
"Amazon",
0
],
[
"2025-12-01",
14.5,
"DeepSeek V3.2 Speciale",
"DeepSeek",
1
],
[
"2025-12-01",
21.5,
"DeepSeek V3.2",
"DeepSeek",
1
],
[
"2025-12-02",
4.8,
"Ministral 3 3B",
"Mistral",
1
],
[
"2025-12-02",
5.5,
"Ministral 3 8B",
"Mistral",
1
],
[
"2025-12-02",
6.0,
"Ministral 3 14B",
"Mistral",
1
],
[
"2025-12-02",
9.7,
"Mistral Large 3",
"Mistral",
1
],
[
"2025-12-04",
9.2,
"Motif-2-12.7B-Reasoning",
"Motif Technologies",
0
],
[
"2025-12-05",
9.9,
"K2-V2",
"MBZUAI Institute of Foundation Models",
1
],
[
"2025-12-08",
11.2,
"GLM-4.6V",
"Z.ai",
1
],
[
"2025-12-09",
8.4,
"Devstral Small 2",
"Mistral",
1
],
[
"2025-12-09",
9.4,
"Devstral 2",
"Mistral",
1
],
[
"2025-12-11",
5.0,
"Molmo2-8B",
"Allen Institute for AI",
1
],
[
"2025-12-11",
11.0,
"Mi:dm K 2.5 Pro",
"Korea Telecom",
0
],
[
"2025-12-11",
28.5,
"GPT-5.2 Codex",
"OpenAI",
0
],
[
"2025-12-11",
30.4,
"GPT-5.2",
"OpenAI",
0
],
[
"2025-12-12",
7.1,
"Olmo 3.1 32B Think",
"Allen Institute for AI",
1
],
[
"2025-12-15",
8.9,
"NVIDIA Nemotron 3 Nano 30B A3B",
"NVIDIA",
1
],
[
"2025-12-15",
11.5,
"K2 Think V2",
"MBZUAI Institute of Foundation Models",
1
],
[
"2025-12-16",
22.4,
"MiMo-V2-Flash",
"Xiaomi",
1
],
[
"2025-12-17",
10.4,
"Solar Open 100B",
"Upstage",
1
],
[
"2025-12-17",
26.3,
"Gemini 3 Flash Preview",
"Google",
0
],
[
"2025-12-22",
22.2,
"GLM-4.7",
"Z.ai",
1
],
[
"2025-12-23",
20.9,
"MiniMax-M2.1",
"MiniMax",
1
],
[
"2025-12-26",
11.4,
"HyperCLOVA X SEED Think",
"Naver",
1
],
[
"2025-12-31",
14.4,
"K-EXAONE",
"LG AI Research",
1
],
[
"2026-01-04",
7.8,
"Falcon-H1R-7B",
"TII UAE",
1
],
[
"2026-01-05",
4.8,
"LFM2.5-VL-1.6B",
"Liquid AI",
1
],
[
"2026-01-05",
5.2,
"LFM2.5-1.2B-Instruct",
"Liquid AI",
1
],
[
"2026-01-13",
6.5,
"Olmo 3.1 32B Instruct",
"Allen Institute for AI",
1
],
[
"2026-01-19",
14.9,
"GLM-4.7-Flash",
"Z.ai",
1
],
[
"2026-01-20",
5.2,
"LFM2.5-1.2B-Thinking",
"Liquid AI",
1
],
[
"2026-01-20",
7.7,
"Step3 VL 10B",
"StepFun",
1
],
[
"2026-01-26",
21.3,
"Qwen3 Max Thinking",
"Alibaba",
0
],
[
"2026-01-27",
23.5,
"Kimi K2.5",
"Kimi",
1
],
[
"2026-01-28",
11.5,
"LongCat Flash Lite",
"LongCat",
1
],
[
"2026-02-02",
16.6,
"Step 3.5 Flash",
"StepFun",
1
],
[
"2026-02-03",
10.1,
"Qwen3 Coder Next",
"Alibaba",
1
],
[
"2026-02-05",
31.9,
"Claude Opus 4.6",
"Anthropic",
0
],
[
"2026-02-05",
32.5,
"GPT-5.3 Codex",
"OpenAI",
0
],
[
"2026-02-10",
9.0,
"Tri-21B-Think",
"Trillion Labs",
1
],
[
"2026-02-10",
9.6,
"Tri-21B-think Preview",
"Trillion Labs",
1
],
[
"2026-02-11",
8.4,
"Nanbeige4.1-3B",
"Nanbeige",
1
],
[
"2026-02-11",
27.9,
"GLM-5",
"Z.ai",
1
],
[
"2026-02-12",
22.8,
"MiniMax-M2.5",
"MiniMax",
1
],
[
"2026-02-16",
21.4,
"Qwen3.5 397B A17B",
"Alibaba",
1
],
[
"2026-02-17",
4.8,
"Tiny Aya Global",
"Cohere",
1
],
[
"2026-02-17",
30.5,
"Claude Sonnet 4.6",
"Anthropic",
0
],
[
"2026-02-19",
30.4,
"Gemini 3.1 Pro Preview",
"Google",
0
],
[
"2026-02-20",
11.5,
"Mercury 2",
"Inception",
0
],
[
"2026-02-24",
17.7,
"Qwen3.5 122B A10B",
"Alibaba",
1
],
[
"2026-02-24",
19.3,
"Qwen3.5 35B A3B",
"Alibaba",
1
],
[
"2026-02-24",
22.9,
"Qwen3.5 27B",
"Alibaba",
1
],
[
"2026-02-25",
5.9,
"LFM2 24B A2B",
"Liquid AI",
1
],
[
"2026-03-02",
6.1,
"Qwen3.5 0.8B",
"Alibaba",
1
],
[
"2026-03-02",
6.9,
"Qwen3.5 2B",
"Alibaba",
1
],
[
"2026-03-02",
13.1,
"Qwen3.5 4B",
"Alibaba",
1
],
[
"2026-03-02",
13.7,
"Qwen3.5 9B",
"Alibaba",
1
],
[
"2026-03-03",
16.0,
"Gemini 3.1 Flash-Lite",
"Google",
0
],
[
"2026-03-05",
39.0,
"GPT-5.4",
"OpenAI",
0
],
[
"2026-03-06",
6.6,
"Sarvam 30B",
"Sarvam",
1
],
[
"2026-03-06",
8.8,
"Sarvam 105B",
"Sarvam",
1
],
[
"2026-03-10",
25.2,
"Grok 4.20 0309",
"xAI",
0
],
[
"2026-03-11",
13.6,
"Nemotron 3 Super 120B A12B",
"NVIDIA",
1
],
[
"2026-03-15",
26.6,
"GLM-5-Turbo",
"Z.ai",
0
],
[
"2026-03-16",
7.4,
"NVIDIA Nemotron 3 Nano 4B",
"NVIDIA",
1
],
[
"2026-03-16",
11.5,
"Mistral Small 4",
"Mistral",
1
],
[
"2026-03-17",
21.2,
"GPT-5.4 nano",
"OpenAI",
0
],
[
"2026-03-17",
24.6,
"GPT-5.4 mini",
"OpenAI",
0
],
[
"2026-03-18",
23.2,
"MiniMax-M2.7",
"MiniMax",
1
],
[
"2026-03-18",
28.6,
"MiMo-V2-Pro",
"Xiaomi",
0
],
[
"2026-03-19",
11.7,
"Nemotron Cascade 2 30B A3B",
"NVIDIA",
1
],
[
"2026-03-19",
23.9,
"MiMo-V2-Omni",
"Xiaomi",
0
],
[
"2026-03-27",
21.7,
"KAT Coder Pro V2",
"KwaiKAT",
0
],
[
"2026-03-27",
25.1,
"MiMo-V2-Omni-0327",
"Xiaomi",
0
],
[
"2026-03-30",
12.5,
"Qwen3.5 Omni Flash",
"Alibaba",
0
],
[
"2026-03-30",
20.4,
"Qwen3.5 Omni Plus",
"Alibaba",
0
],
[
"2026-04-01",
10.9,
"Trinity Large Thinking",
"Arcee AI",
1
],
[
"2026-04-01",
23.5,
"GLM 5V Turbo",
"Z.ai",
0
],
[
"2026-04-02",
7.8,
"Gemma 4 E2B",
"Google",
1
],
[
"2026-04-02",
15.4,
"Gemma 4 31B",
"Google",
1
],
[
"2026-04-02",
16.7,
"Gemma 4 26B A4B",
"Google",
1
],
[
"2026-04-02",
17.0,
"Step 3.5 Flash 2603",
"StepFun",
0
],
[
"2026-04-02",
27.0,
"Qwen3.6 Plus",
"Alibaba",
0
],
[
"2026-04-03",
8.9,
"Gemma 4 E4B",
"Google",
1
],
[
"2026-04-06",
7.8,
"Solar Pro 3",
"Upstage",
0
],
[
"2026-04-07",
25.7,
"Grok 4.20 0309 v2",
"xAI",
0
],
[
"2026-04-07",
27.4,
"GLM-5.1",
"Z.ai",
1
],
[
"2026-04-08",
31.3,
"Muse Spark",
"Meta",
0
],
[
"2026-04-09",
13.2,
"EXAONE 4.5 33B",
"LG AI Research",
1
],
[
"2026-04-15",
12.2,
"JT-MINI",
"China Mobile",
0
],
[
"2026-04-16",
22.3,
"Qwen3.6 35B A3B",
"Alibaba",
1
],
[
"2026-04-16",
40.7,
"Claude Opus 4.7",
"Anthropic",
0
],
[
"2026-04-20",
28.4,
"Qwen3.6 Max Preview",
"Alibaba",
0
],
[
"2026-04-20",
31.3,
"Kimi K2.6",
"Kimi",
1
],
[
"2026-04-21",
9.7,
"Ling 2.6 Flash",
"InclusionAI",
1
],
[
"2026-04-22",
21.9,
"Qwen3.6 27B",
"Alibaba",
1
],
[
"2026-04-22",
22.3,
"MiMo-V2.5",
"Xiaomi",
1
],
[
"2026-04-22",
26.4,
"MiMo-V2.5-Pro",
"Xiaomi",
1
],
[
"2026-04-23",
17.0,
"Ling-2.6-1T",
"InclusionAI",
1
],
[
"2026-04-23",
22.7,
"Hy3-preview",
"Tencent",
1
],
[
"2026-04-23",
38.6,
"GPT-5.5",
"OpenAI",
0
],
[
"2026-04-24",
24.8,
"DeepSeek V4 Flash",
"DeepSeek",
1
],
[
"2026-04-24",
30.9,
"DeepSeek V4 Pro",
"DeepSeek",
1
],
[
"2026-04-29",
5.9,
"Granite 4.1 3B",
"IBM",
1
],
[
"2026-04-29",
6.6,
"Granite 4.1 8B",
"IBM",
1
],
[
"2026-04-29",
7.4,
"Granite 4.1 30B",
"IBM",
1
],
[
"2026-04-29",
10.3,
"Nemotron 3 Nano Omni 30B A3B Reasoning",
"NVIDIA",
1
],
[
"2026-04-29",
14.9,
"Mistral Medium 3.5",
"Mistral",
1
],
[
"2026-04-30",
25.4,
"Grok 4.3",
"xAI",
0
],
[
"2026-05-05",
22.7,
"GPT-5.5 Instant",
"OpenAI",
0
],
[
"2026-05-08",
17.3,
"Ring-2.6-1T",
"InclusionAI",
1
],
[
"2026-05-11",
5.7,
"MiniCPM-V 4.6 1.3B",
"OpenBMB",
1
],
[
"2026-05-14",
18.7,
"JT-35B-Flash",
"China Mobile",
0
],
[
"2026-05-19",
29.9,
"Qwen3.7 Max",
"Alibaba",
0
],
[
"2026-05-19",
33.6,
"Gemini 3.5 Flash",
"Google",
0
],
[
"2026-05-20",
13.9,
"Command A+",
"Cohere",
1
],
[
"2026-05-25",
8.8,
"MiniCPM5-1B",
"OpenBMB",
1
],
[
"2026-05-26",
11.7,
"HyperNova 60B 2605",
"Multiverse Computing",
1
],
[
"2026-05-28",
7.2,
"LFM2.5-8B-A1B",
"Liquid AI",
1
],
[
"2026-05-28",
42.0,
"Claude Opus 4.8",
"Anthropic",
0
],
[
"2026-05-29",
19.5,
"Step 3.7 Flash",
"StepFun",
1
],
[
"2026-06-01",
25.8,
"Qwen3.7 Plus",
"Alibaba",
0
],
[
"2026-06-01",
29.6,
"MiniMax-M3",
"MiniMax",
1
],
[
"2026-06-02",
28.2,
"Nex-N2-Pro",
"Nex AGI",
1
],
[
"2026-06-03",
14.2,
"Gemma 4 12B",
"Google",
1
],
[
"2026-06-04",
23.4,
"Nemotron 3 Ultra 550B A55B",
"NVIDIA",
1
],
[
"2026-06-09",
12.8,
"North Mini Code",
"Cohere",
1
],
[
"2026-06-09",
49.7,
"Claude Fable 5",
"Anthropic",
0
],
[
"2026-06-10",
9.5,
"DiffusionGemma 26B A4B",
"Google",
1
],
[
"2026-06-12",
26.3,
"Kimi K2.7 Code",
"Kimi",
1
],
[
"2026-06-16",
27.2,
"Grok Build 0.1 0616",
"xAI",
0
],
[
"2026-06-16",
38.6,
"GLM-5.2",
"Z.ai",
1
],
[
"2026-06-25",
26.8,
"GPT-5.5 Instant",
"OpenAI",
0
],
[
"2026-06-29",
19.7,
"LongCat 2.0",
"LongCat",
1
],
[
"2026-06-30",
38.4,
"Claude Sonnet 5",
"Anthropic",
0
],
[
"2026-07-06",
25.8,
"Hy3",
"Tencent",
1
],
[
"2026-07-08",
39.1,
"Grok 4.5",
"xAI",
0
],
[
"2026-07-09",
27.3,
"JT-4.1 Flash 236B A21B",
"China Mobile",
0
],
[
"2026-07-09",
34.3,
"Muse Spark 1.1",
"Meta",
0
],
[
"2026-07-09",
37.5,
"GPT-5.6 Luna",
"OpenAI",
0
],
[
"2026-07-09",
42.3,
"GPT-5.6 Terra",
"OpenAI",
0
],
[
"2026-07-09",
47.1,
"GPT-5.6 Sol",
"OpenAI",
0
],
[
"2026-07-14",
32.3,
"Motif 3",
"Motif Technologies",
0
],
[
"2026-07-15",
25.5,
"Inkling",
"Thinking Machines",
1
],
[
"2026-07-16",
43.8,
"Kimi K3",
"Kimi",
1
],
[
"2026-07-21",
22.7,
"Gemini 3.5 Flash-Lite",
"Google",
0
],
[
"2026-07-21",
34.3,
"Gemini 3.6 Flash",
"Google",
0
],
[
"2026-07-23",
10.8,
"G9v3-3B",
"AI9Stars",
1
],
[
"2026-07-24",
6.3,
"Celeris-1",
"Celeris",
0
],
[
"2026-07-24",
26.8,
"Agnes 2.5 Pro Alpha",
"Sapiens AI",
0
],
[
"2026-07-24",
50.7,
"Claude Opus 5",
"Anthropic",
0
],
[
"2026-07-30",
26.1,
"Inkling Small",
"Thinking Machines",
1
],
[
"2026-07-31",
34.5,
"DeepSeek V4 Flash 0731",
"DeepSeek",
1
],
[
"2026-08-03",
21.8,
"G9v3-39A5B",
"AI9Stars",
1
],
[
"2026-08-03",
40.3,
"Qwen3.8 Max",
"Alibaba",
0
],
[
"2026-08-04",
8.4,
"LFM2.5-2.6B",
"Liquid AI",
1
],
[
"2026-08-04",
24.9,
"Ling 3.0 Flash",
"InclusionAI",
1
],
[
"2026-08-05",
39.8,
"Muse Spark 1.2",
"Meta",
0
],
[
"2026-08-06",
11.9,
"Ling 3.0 Tiny",
"InclusionAI",
0
],
[
"2026-08-06",
28.2,
"Solar Pro 4",
"Upstage",
0
],
[
"2026-08-10",
18.1,
"Muse Glimmer",
"Meta",
1
],
[
"2026-08-10",
27.1,
"Quasar 438B",
"Multiverse Computing",
0
],
[
"2026-08-11",
13.6,
"Nemotron 3.5 Lightning",
"NVIDIA",
1
],
[
"2026-08-12",
19.7,
"K-EXAONE 2.0",
"LG AI Research",
1
],
[
"2026-08-12",
19.7,
"K-EXAONE 2.0 0803",
"LG AI Research",
1
],
[
"2026-08-12",
22.7,
"A.X-K2",
"SK Telecom",
1
],
[
"2026-08-12",
24.7,
"Solar Open2 250B",
"Upstage",
1
],
[
"2026-08-12",
33.6,
"Motif 3",
"Motif Technologies",
1
],
[
"2026-08-12",
40.0,
"Qwen3.8 2.4T A95B",
"Alibaba",
1
],
[
"2026-08-12",
44.4,
"Grok 4.6",
"xAI",
0
],
[
"2026-08-13",
36.3,
"DeepSeek V4 Pro 0813",
"DeepSeek",
1
],
[
"2026-08-13",
39.6,
"Gemini 3.7 Flash",
"Google",
0
],
[
"2026-08-14",
33.9,
"Qwen3.8 27B",
"Alibaba",
1
],
[
"2026-08-18",
44.9,
"GLM-5.3",
"Z.ai",
0
],
[
"2026-08-21",
35.0,
"DeepSeek V4 Flash Vision",
"DeepSeek",
0
],
[
"2026-08-25",
9.1,
"Granite 4.2 3B",
"IBM",
1
],
[
"2026-08-25",
12.4,
"Granite 4.2 8B",
"IBM",
1
],
[
"2026-08-25",
14.8,
"Granite 4.2 30B",
"IBM",
1
],
[
"2026-08-26",
35.2,
"Agnes 2.5 Pro Beta",
"Sapiens AI",
0
],
[
"2026-08-26",
41.9,
"GLM-5.3-Flash",
"Z.ai",
1
],
[
"2026-08-26",
42.2,
"Qwen3.8-Flash-Next",
"Alibaba",
1
],
[
"2026-08-30",
30.4,
"Apodex 1.1",
"Apodex",
0
],
[
"2026-09-01",
53.4,
"Claude Fable 5.1",
"Anthropic",
0
],
[
"2026-09-02",
41.2,
"Gemini 3.8 Flash",
"Google",
0
],
[
"2026-09-02",
48.2,
"Muse Spark 1.3",
"Meta",
0
],
[
"2026-09-03",
33.9,
"K2 Horizon 375B A23B",
"MBZUAI Institute of Foundation Models",
1
],
[
"2026-09-03",
52.8,
"GPT-6 Astra",
"OpenAI",
0
]
]

L = {'months': None, 'ylabel': 'Intelligence Index (Artificial Analysis)', 'title': 'AI Model Capability Over Time — A Single Number', 'sub': 'independent measurement of {n} models ({lo} – {hi})  ·  yellow: best available  ·  green dashed: best open-weights', 'growth': 'Last 12 months\n{a:.0f} → {b:.0f}  ({k:.1f}x)\nBest open-weights: {ao:.0f}  (gap {fark:.1f})', 'cloud': 'other measured models', 'front_all': 'best at the time', 'front_open': 'best open-weights', 'credit': 'Source: artificialanalysis.ai  ·  Compiled by Prof. Dr. Oğuz Ergin'}

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
# Ay adlari dile bagli. Onceden her iki cizelgede de %b kullaniliyordu,
# Turkce cizelgenin ekseni ve alt basligi "Nov 2022 - Sep 2026" diyordu.
if L.get("months"):
    _AY = L["months"]
    def _ay(ts): return "%s %d" % (_AY[ts.month - 1], ts.year)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, pos: _ay(mdates.num2date(x))))
else:
    def _ay(ts): return ts.strftime("%b %Y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

# baslik + alt baslik
son = fr.iloc[-1]
bir_yil = fr[fr["Date"] <= son["Date"] - pd.Timedelta(days=365)]["ii"].max()
plt.title(L["title"], fontsize=30, color="white", pad=54, fontweight="bold")
ax.text(0.5, 1.045, L["sub"].format(n=len(df), lo=_ay(df["Date"].min()),
                                    hi=_ay(df["Date"].max())),
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
