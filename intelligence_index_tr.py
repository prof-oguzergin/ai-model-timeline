# -*- coding: utf-8 -*-
# Yapay zeka basarimi tek sayiyla: Artificial Analysis Zeka Endeksi (Intelligence Index)
# Veri kaynagi: artificialanalysis.ai  (435 model, 2022-11-30 - 2026-07-24)
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
3.6,
"GPT-3.5 Turbo",
"OpenAI"
],
[
"2023-02-24",
2.1,
"Llama 65B",
"Meta"
],
[
"2023-03-14",
2.1,
"Claude Instant",
"Anthropic"
],
[
"2023-03-14",
7.0,
"GPT-4",
"OpenAI"
],
[
"2023-05-10",
3.2,
"PALM-2",
"Google"
],
[
"2023-07-11",
3.6,
"Claude 2.0",
"Anthropic"
],
[
"2023-07-18",
3.0,
"Llama 2 Chat 13B",
"Meta"
],
[
"2023-07-18",
3.0,
"Llama 2 Chat 70B",
"Meta"
],
[
"2023-07-18",
4.3,
"Llama 2 Chat 7B",
"Meta"
],
[
"2023-09-25",
2.1,
"Qwen Chat 14B",
"Alibaba"
],
[
"2023-09-27",
2.1,
"Mistral 7B Instruct",
"Mistral"
],
[
"2023-11-06",
7.9,
"GPT-4 Turbo",
"OpenAI"
],
[
"2023-11-21",
3.9,
"Claude 2.1",
"Anthropic"
],
[
"2023-11-29",
3.0,
"DeepSeek LLM 67B Chat",
"DeepSeek"
],
[
"2023-11-30",
3.4,
"Qwen Chat 72B",
"Alibaba"
],
[
"2023-12-06",
3.1,
"Gemini 1.0 Pro",
"Google"
],
[
"2023-12-06",
4.6,
"Gemini 1.0 Ultra",
"Google"
],
[
"2023-12-11",
2.4,
"Mixtral 8x7B Instruct",
"Mistral"
],
[
"2023-12-11",
3.6,
"Mistral Medium",
"Mistral"
],
[
"2023-12-18",
3.0,
"OpenChat 3.5",
"OpenChat"
],
[
"2024-01-25",
6.2,
"Solar Mini",
"Upstage"
],
[
"2024-02-26",
3.6,
"Mistral Small",
"Mistral"
],
[
"2024-02-26",
4.4,
"Mistral Large",
"Mistral"
],
[
"2024-02-26",
6.0,
"Phi-4 Mini Instruct",
"Microsoft"
],
[
"2024-03-04",
3.9,
"Claude 3 Haiku",
"Anthropic"
],
[
"2024-03-04",
4.7,
"Claude 3 Sonnet",
"Anthropic"
],
[
"2024-03-04",
11.8,
"Claude 3 Opus",
"Anthropic"
],
[
"2024-03-12",
2.1,
"Command-R",
"Cohere"
],
[
"2024-03-17",
6.0,
"Grok-1",
"xAI"
],
[
"2024-03-27",
3.0,
"DBRX Instruct",
"Databricks"
],
[
"2024-04-04",
3.0,
"Command-R+",
"Cohere"
],
[
"2024-04-17",
4.4,
"Mixtral 8x22B Instruct",
"Mistral"
],
[
"2024-04-18",
1.2,
"Llama 3 Instruct 8B",
"Meta"
],
[
"2024-04-18",
3.5,
"Llama 3 Instruct 70B",
"Meta"
],
[
"2024-04-23",
4.6,
"Phi-3 Mini Instruct 3.8B",
"Microsoft"
],
[
"2024-04-24",
3.4,
"Arctic Instruct",
"Snowflake"
],
[
"2024-04-25",
4.1,
"Qwen1.5 Chat 110B",
"Alibaba"
],
[
"2024-05-06",
3.6,
"DeepSeek-V2-Chat",
"DeepSeek"
],
[
"2024-05-13",
8.6,
"GPT-4o",
"OpenAI"
],
[
"2024-05-14",
4.9,
"Gemini 1.5 Flash",
"Google"
],
[
"2024-05-15",
6.3,
"Gemini 1.5 Pro",
"Google"
],
[
"2024-06-07",
6.0,
"Qwen2 Instruct 72B",
"Alibaba"
],
[
"2024-06-17",
3.1,
"DeepSeek Coder V2 Lite Instruct",
"DeepSeek"
],
[
"2024-06-17",
5.1,
"DeepSeek-Coder-V2",
"DeepSeek"
],
[
"2024-06-21",
8.3,
"Claude 3.5 Sonnet",
"Anthropic"
],
[
"2024-07-18",
6.9,
"GPT-4o mini",
"OpenAI"
],
[
"2024-07-23",
6.8,
"Llama 3.1 Instruct 70B",
"Meta"
],
[
"2024-07-23",
7.6,
"Llama 3.1 Instruct 8B",
"Meta"
],
[
"2024-07-23",
8.5,
"Llama 3.1 Instruct 405B",
"Meta"
],
[
"2024-07-24",
7.3,
"Mistral Large 2",
"Mistral"
],
[
"2024-08-06",
9.6,
"GPT-4o",
"OpenAI"
],
[
"2024-08-13",
7.5,
"Grok Beta",
"xAI"
],
[
"2024-08-15",
5.1,
"Hermes 3 - Llama-3.1 70B",
"Nous Research"
],
[
"2024-08-22",
2.7,
"Jamba 1.5 Mini",
"AI21 Labs"
],
[
"2024-08-22",
5.1,
"Jamba 1.5 Large",
"AI21 Labs"
],
[
"2024-09-06",
6.6,
"DeepSeek-V2.5",
"DeepSeek"
],
[
"2024-09-12",
14.0,
"o1-mini",
"OpenAI"
],
[
"2024-09-12",
17.0,
"o1-preview",
"OpenAI"
],
[
"2024-09-17",
4.7,
"Mistral Small",
"Mistral"
],
[
"2024-09-19",
4.5,
"Qwen2.5 Coder Instruct 7B",
"Alibaba"
],
[
"2024-09-19",
7.5,
"Qwen2.5 Instruct 32B",
"Alibaba"
],
[
"2024-09-19",
9.6,
"Qwen2.5 Instruct 72B",
"Alibaba"
],
[
"2024-09-24",
8.0,
"Gemini 1.5 Flash",
"Google"
],
[
"2024-09-24",
10.0,
"Gemini 1.5 Pro",
"Google"
],
[
"2024-09-25",
1.1,
"Llama 3.2 Instruct 1B",
"Meta"
],
[
"2024-09-25",
3.3,
"Llama 3.2 Instruct 11B",
"Meta"
],
[
"2024-09-25",
3.8,
"Molmo 7B-D",
"Allen Institute for AI"
],
[
"2024-09-25",
4.2,
"Llama 3.2 Instruct 3B",
"Meta"
],
[
"2024-09-25",
6.2,
"Llama 3.2 Instruct 90B",
"Meta"
],
[
"2024-09-30",
3.4,
"LFM 40B",
"Liquid AI"
],
[
"2024-10-03",
5.5,
"Gemini 1.5 Flash-8B",
"Google"
],
[
"2024-10-04",
6.3,
"Reka Flash",
"Reka AI"
],
[
"2024-10-15",
7.6,
"Llama 3.1 Nemotron Instruct 70B",
"NVIDIA"
],
[
"2024-10-22",
9.9,
"Claude 3.5 Sonnet",
"Anthropic"
],
[
"2024-10-22",
12.3,
"Claude 3.5 Haiku",
"Anthropic"
],
[
"2024-11-11",
7.1,
"Qwen2.5 Coder Instruct 32B",
"Alibaba"
],
[
"2024-11-18",
6.3,
"Qwen2.5 Turbo",
"Alibaba"
],
[
"2024-11-18",
8.1,
"Pixtral Large",
"Mistral"
],
[
"2024-11-18",
9.1,
"Mistral Large 2",
"Mistral"
],
[
"2024-11-20",
11.2,
"GPT-4o",
"OpenAI"
],
[
"2024-11-26",
3.9,
"OLMo 2 7B",
"Allen Institute for AI"
],
[
"2024-11-27",
9.2,
"QwQ 32B-Preview",
"Alibaba"
],
[
"2024-12-03",
4.7,
"Nova Micro",
"Amazon"
],
[
"2024-12-03",
6.9,
"Nova Lite",
"Amazon"
],
[
"2024-12-03",
7.7,
"Nova Pro",
"Amazon"
],
[
"2024-12-05",
23.4,
"o1",
"OpenAI"
],
[
"2024-12-06",
9.4,
"Llama 3.3 Instruct 70B",
"Meta"
],
[
"2024-12-10",
6.8,
"DeepSeek-V2.5",
"DeepSeek"
],
[
"2024-12-11",
10.7,
"Gemini 2.0 Flash",
"Google"
],
[
"2024-12-12",
4.9,
"Phi-4",
"Microsoft"
],
[
"2024-12-12",
8.0,
"Grok 2",
"xAI"
],
[
"2024-12-19",
6.6,
"Gemini 2.0 Flash Thinking Experimental",
"Google"
],
[
"2024-12-26",
14.2,
"DeepSeek V3",
"DeepSeek"
],
[
"2025-01-20",
3.7,
"DeepSeek R1 Distill Qwen 1.5B",
"DeepSeek"
],
[
"2025-01-20",
6.4,
"DeepSeek R1 Distill Llama 8B",
"DeepSeek"
],
[
"2025-01-20",
9.8,
"DeepSeek R1 Distill Qwen 14B",
"DeepSeek"
],
[
"2025-01-20",
9.9,
"DeepSeek R1 Distill Llama 70B",
"DeepSeek"
],
[
"2025-01-20",
11.0,
"DeepSeek R1 Distill Qwen 32B",
"DeepSeek"
],
[
"2025-01-20",
18.5,
"DeepSeek R1",
"DeepSeek"
],
[
"2025-01-21",
9.3,
"Sonar Pro",
"Perplexity"
],
[
"2025-01-21",
9.5,
"Sonar",
"Perplexity"
],
[
"2025-01-21",
13.3,
"Gemini 2.0 Flash Thinking Experimental",
"Google"
],
[
"2025-01-28",
10.2,
"Qwen2.5 Max",
"Alibaba"
],
[
"2025-01-28",
11.7,
"Sonar Reasoning",
"Perplexity"
],
[
"2025-01-28",
17.8,
"Sonar Reasoning Pro",
"Perplexity"
],
[
"2025-01-30",
6.9,
"Mistral Small 3",
"Mistral"
],
[
"2025-01-30",
8.3,
"Llama 3.1 Tulu3 405B",
"Allen Institute for AI"
],
[
"2025-01-31",
19.0,
"o3-mini",
"OpenAI"
],
[
"2025-02-05",
8.6,
"Gemini 2.0 Flash-Lite",
"Google"
],
[
"2025-02-05",
11.8,
"Gemini 2.0 Pro Experimental",
"Google"
],
[
"2025-02-05",
12.3,
"Gemini 2.0 Flash",
"Google"
],
[
"2025-02-13",
2.3,
"DeepHermes 3 - Llama-3.1 8B Preview",
"Nous Research"
],
[
"2025-02-15",
8.2,
"GPT-4o",
"OpenAI"
],
[
"2025-02-17",
6.4,
"Mistral Saba",
"Mistral"
],
[
"2025-02-18",
6.3,
"R1 1776",
"Perplexity"
],
[
"2025-02-19",
15.1,
"Grok 3 Reasoning Beta",
"xAI"
],
[
"2025-02-19",
18.4,
"Grok 3",
"xAI"
],
[
"2025-02-19",
22.5,
"Grok 3 mini Reasoning",
"xAI"
],
[
"2025-02-24",
27.1,
"Claude 3.7 Sonnet",
"Anthropic"
],
[
"2025-02-25",
8.8,
"Gemini 2.0 Flash-Lite",
"Google"
],
[
"2025-02-26",
4.5,
"Phi-4 Multimodal Instruct",
"Microsoft"
],
[
"2025-02-27",
13.6,
"GPT-4.5",
"OpenAI"
],
[
"2025-03-05",
13.4,
"QwQ 32B",
"Alibaba"
],
[
"2025-03-06",
2.6,
"Jamba 1.6 Mini",
"AI21 Labs"
],
[
"2025-03-06",
5.0,
"Jamba 1.6 Large",
"AI21 Labs"
],
[
"2025-03-10",
4.1,
"Reka Flash 3",
"Reka AI"
],
[
"2025-03-12",
1.1,
"Gemma 3 4B Instruct",
"Google"
],
[
"2025-03-12",
5.5,
"Gemma 3 12B Instruct",
"Google"
],
[
"2025-03-12",
7.4,
"Gemma 3 27B Instruct",
"Google"
],
[
"2025-03-13",
1.0,
"Gemma 3 1B Instruct",
"Google"
],
[
"2025-03-13",
5.0,
"OLMo 2 32B",
"Allen Institute for AI"
],
[
"2025-03-13",
5.3,
"DeepHermes 3 - Mistral 24B Preview",
"Nous Research"
],
[
"2025-03-13",
7.7,
"Command A",
"Cohere"
],
[
"2025-03-17",
14.7,
"Mistral Small 3.1",
"Mistral"
],
[
"2025-03-18",
12.2,
"Llama 3.3 Nemotron Super 49B v1",
"NVIDIA"
],
[
"2025-03-19",
18.9,
"o1-pro",
"OpenAI"
],
[
"2025-03-25",
15.4,
"DeepSeek V3 0324",
"DeepSeek"
],
[
"2025-03-25",
23.0,
"Gemini 2.5 Pro Preview",
"Google"
],
[
"2025-03-27",
12.3,
"GPT-4o",
"OpenAI"
],
[
"2025-04-05",
10.0,
"Llama 4 Scout",
"Meta"
],
[
"2025-04-05",
14.3,
"Llama 4 Maverick",
"Meta"
],
[
"2025-04-07",
9.1,
"Llama 3.1 Nemotron Ultra 253B v1",
"NVIDIA"
],
[
"2025-04-14",
9.6,
"GPT-4.1 nano",
"OpenAI"
],
[
"2025-04-14",
14.8,
"GPT-4.1 mini",
"OpenAI"
],
[
"2025-04-14",
19.4,
"GPT-4.1",
"OpenAI"
],
[
"2025-04-16",
1.8,
"Granite 3.3 8B",
"IBM"
],
[
"2025-04-16",
25.6,
"o4-mini",
"OpenAI"
],
[
"2025-04-16",
30.4,
"o3",
"OpenAI"
],
[
"2025-04-17",
17.5,
"Gemini 2.5 Flash Preview",
"Google"
],
[
"2025-04-28",
1.3,
"Qwen3 0.6B",
"Alibaba"
],
[
"2025-04-28",
2.6,
"Qwen3 1.7B",
"Alibaba"
],
[
"2025-04-28",
8.3,
"Qwen3 8B",
"Alibaba"
],
[
"2025-04-28",
8.4,
"Qwen3 4B",
"Alibaba"
],
[
"2025-04-28",
9.3,
"Qwen3 30B A3B",
"Alibaba"
],
[
"2025-04-28",
10.4,
"Qwen3 14B",
"Alibaba"
],
[
"2025-04-28",
11.5,
"Qwen3 32B",
"Alibaba"
],
[
"2025-04-28",
13.4,
"Qwen3 235B A22B",
"Alibaba"
],
[
"2025-04-30",
12.7,
"Nova Premier",
"Amazon"
],
[
"2025-05-06",
22.3,
"Gemini 2.5 Pro Preview",
"Google"
],
[
"2025-05-07",
12.5,
"Mistral Medium 3",
"Mistral"
],
[
"2025-05-20",
4.6,
"Gemma 3n E4B Instruct Preview",
"Google"
],
[
"2025-05-20",
8.5,
"Llama 3.1 Nemotron Nano 4B v1.1",
"NVIDIA"
],
[
"2025-05-20",
12.5,
"Solar Pro 2",
"Upstage"
],
[
"2025-05-20",
20.1,
"Gemini 2.5 Flash",
"Google"
],
[
"2025-05-21",
11.8,
"Devstral Small",
"Mistral"
],
[
"2025-05-22",
28.9,
"Claude 4 Sonnet",
"Anthropic"
],
[
"2025-05-22",
31.0,
"Claude 4 Opus",
"Anthropic"
],
[
"2025-05-23",
3.0,
"Sarvam M",
"Sarvam"
],
[
"2025-05-28",
20.1,
"DeepSeek R1 0528",
"DeepSeek"
],
[
"2025-05-29",
10.4,
"DeepSeek R1 0528 Qwen3 8B",
"DeepSeek"
],
[
"2025-06-05",
25.8,
"Gemini 2.5 Pro",
"Google"
],
[
"2025-06-10",
10.7,
"Magistral Small 1",
"Mistral"
],
[
"2025-06-10",
12.5,
"Magistral Medium 1",
"Mistral"
],
[
"2025-06-10",
32.5,
"o3-pro",
"OpenAI"
],
[
"2025-06-17",
11.4,
"Gemini 2.5 Flash-Lite",
"Google"
],
[
"2025-06-17",
14.4,
"MiniMax M1 40k",
"MiniMax"
],
[
"2025-06-17",
17.7,
"MiniMax M1 80k",
"MiniMax"
],
[
"2025-06-20",
10.6,
"Mistral Small 3.2",
"Mistral"
],
[
"2025-06-26",
1.0,
"Gemma 3n E2B Instruct",
"Google"
],
[
"2025-06-26",
1.2,
"Gemma 3n E4B Instruct",
"Google"
],
[
"2025-06-30",
9.0,
"ERNIE 4.5 300B A47B",
"Baidu"
],
[
"2025-07-07",
2.7,
"Jamba 1.7 Mini",
"AI21 Labs"
],
[
"2025-07-07",
5.3,
"Jamba 1.7 Large",
"AI21 Labs"
],
[
"2025-07-09",
9.0,
"Solar Pro 2",
"Upstage"
],
[
"2025-07-10",
1.1,
"LFM2 1.2B",
"Liquid AI"
],
[
"2025-07-10",
9.3,
"Devstral Small",
"Mistral"
],
[
"2025-07-10",
12.4,
"Devstral Medium",
"Mistral"
],
[
"2025-07-10",
33.3,
"Grok 4",
"xAI"
],
[
"2025-07-11",
19.4,
"Kimi K2",
"Kimi"
],
[
"2025-07-15",
2.9,
"Exaone 4.0 1.2B",
"LG AI Research"
],
[
"2025-07-15",
10.6,
"EXAONE 4.0 32B",
"LG AI Research"
],
[
"2025-07-21",
18.2,
"Qwen3 235B A22B 2507 Instruct",
"Alibaba"
],
[
"2025-07-22",
18.0,
"Qwen3 Coder 480B A35B Instruct",
"Alibaba"
],
[
"2025-07-25",
12.4,
"Llama Nemotron Super 49B v1.5",
"NVIDIA"
],
[
"2025-07-25",
19.6,
"Qwen3 235B A22B 2507",
"Alibaba"
],
[
"2025-07-28",
16.5,
"GLM-4.5-Air",
"Z.ai"
],
[
"2025-07-28",
19.5,
"GLM-4.5",
"Z.ai"
],
[
"2025-07-29",
9.1,
"Qwen3 30B A3B 2507 Instruct",
"Alibaba"
],
[
"2025-07-30",
14.4,
"Qwen3 30B A3B 2507",
"Alibaba"
],
[
"2025-07-31",
13.6,
"Qwen3 Coder 30B A3B Instruct",
"Alibaba"
],
[
"2025-08-05",
14.9,
"gpt-oss-20b",
"OpenAI"
],
[
"2025-08-05",
23.8,
"gpt-oss-120b",
"OpenAI"
],
[
"2025-08-05",
33.7,
"Claude 4.1 Opus",
"Anthropic"
],
[
"2025-08-06",
7.1,
"Qwen3 4B 2507 Instruct",
"Alibaba"
],
[
"2025-08-06",
12.0,
"Qwen3 4B 2507",
"Alibaba"
],
[
"2025-08-07",
19.9,
"GPT-5 nano",
"OpenAI"
],
[
"2025-08-07",
30.9,
"GPT-5 mini",
"OpenAI"
],
[
"2025-08-07",
34.7,
"GPT-5",
"OpenAI"
],
[
"2025-08-11",
9.1,
"GLM-4.5V",
"Z.ai"
],
[
"2025-08-12",
14.7,
"Mistral Medium 3.1",
"Mistral"
],
[
"2025-08-14",
2.4,
"Gemma 3 270M",
"Google"
],
[
"2025-08-18",
8.8,
"NVIDIA Nemotron Nano 9B V2",
"NVIDIA"
],
[
"2025-08-20",
18.3,
"Seed-OSS-36B-Instruct",
"ByteDance"
],
[
"2025-08-21",
21.0,
"DeepSeek V3.1",
"DeepSeek"
],
[
"2025-08-27",
9.0,
"Hermes 4 - Llama-3.1 405B",
"Nous Research"
],
[
"2025-08-27",
10.0,
"Hermes 4 - Llama-3.1 70B",
"Nous Research"
],
[
"2025-08-28",
21.6,
"Grok Code Fast 1",
"xAI"
],
[
"2025-09-02",
1.0,
"Apertus 8B Instruct",
"Swiss AI Initiative"
],
[
"2025-09-02",
2.4,
"Apertus 70B Instruct",
"Swiss AI Initiative"
],
[
"2025-09-05",
19.2,
"Qwen3 Max",
"Alibaba"
],
[
"2025-09-05",
23.5,
"Kimi K2 0905",
"Kimi"
],
[
"2025-09-08",
15.1,
"Gemini 2.5 Flash-Lite Preview",
"Google"
],
[
"2025-09-09",
3.8,
"Ling-mini-2.0",
"InclusionAI"
],
[
"2025-09-11",
13.7,
"Qwen3 Next 80B A3B Instruct",
"Alibaba"
],
[
"2025-09-11",
16.7,
"Qwen3 Next 80B A3B",
"Alibaba"
],
[
"2025-09-17",
9.7,
"Ling-flash-2.0",
"InclusionAI"
],
[
"2025-09-17",
11.3,
"Magistral Small 1.2",
"Mistral"
],
[
"2025-09-18",
17.9,
"Magistral Medium 1.2",
"Mistral"
],
[
"2025-09-19",
8.2,
"Ring-flash-2.0",
"InclusionAI"
],
[
"2025-09-19",
27.4,
"Grok 4 Fast",
"xAI"
],
[
"2025-09-22",
2.4,
"Granite 4.0 Micro",
"IBM"
],
[
"2025-09-22",
5.1,
"Qwen3 Omni 30B A3B Instruct",
"Alibaba"
],
[
"2025-09-22",
5.2,
"Granite 4.0 H Small",
"IBM"
],
[
"2025-09-22",
9.6,
"Qwen3 Omni 30B A3B",
"Alibaba"
],
[
"2025-09-22",
30.4,
"DeepSeek V3.1 Terminus",
"DeepSeek"
],
[
"2025-09-23",
2.7,
"LFM2 2.6B",
"Liquid AI"
],
[
"2025-09-23",
14.3,
"Qwen3 VL 235B A22B Instruct",
"Alibaba"
],
[
"2025-09-23",
20.6,
"Qwen3 VL 235B A22B",
"Alibaba"
],
[
"2025-09-23",
24.0,
"Qwen3 Max",
"Alibaba"
],
[
"2025-09-23",
36.1,
"GPT-5 Codex",
"OpenAI"
],
[
"2025-09-25",
13.1,
"Gemini 2.5 Flash-Lite Preview",
"Google"
],
[
"2025-09-25",
23.8,
"Gemini 2.5 Flash Preview",
"Google"
],
[
"2025-09-29",
25.4,
"DeepSeek V3.2 Exp",
"DeepSeek"
],
[
"2025-09-29",
36.4,
"Claude 4.5 Sonnet",
"Anthropic"
],
[
"2025-09-30",
21.2,
"Apriel-v1.5-15B-Thinker",
"ServiceNow"
],
[
"2025-09-30",
28.7,
"GLM-4.6",
"Z.ai"
],
[
"2025-10-03",
10.0,
"Qwen3 VL 30B A3B Instruct",
"Alibaba"
],
[
"2025-10-03",
13.3,
"Qwen3 VL 30B A3B",
"Alibaba"
],
[
"2025-10-07",
1.8,
"LFM2 8B A1B",
"Liquid AI"
],
[
"2025-10-08",
4.1,
"Jamba Reasoning 3B",
"AI21 Labs"
],
[
"2025-10-08",
12.8,
"Ling-1T",
"InclusionAI"
],
[
"2025-10-13",
16.2,
"Ring-1T",
"InclusionAI"
],
[
"2025-10-14",
4.1,
"Qwen3 VL 4B Instruct",
"Alibaba"
],
[
"2025-10-14",
7.9,
"Qwen3 VL 4B",
"Alibaba"
],
[
"2025-10-14",
8.4,
"Qwen3 VL 8B Instruct",
"Alibaba"
],
[
"2025-10-14",
10.6,
"Qwen3 VL 8B",
"Alibaba"
],
[
"2025-10-15",
29.6,
"Claude 4.5 Haiku",
"Anthropic"
],
[
"2025-10-21",
11.1,
"Qwen3 VL 32B Instruct",
"Alibaba"
],
[
"2025-10-21",
17.9,
"Qwen3 VL 32B",
"Alibaba"
],
[
"2025-10-26",
28.3,
"MiniMax-M2",
"MiniMax"
],
[
"2025-10-28",
1.0,
"Granite 4.0 350M",
"IBM"
],
[
"2025-10-28",
1.0,
"Granite 4.0 H 350M",
"IBM"
],
[
"2025-10-28",
2.1,
"Granite 4.0 1B",
"IBM"
],
[
"2025-10-28",
2.7,
"Granite 4.0 H 1B",
"IBM"
],
[
"2025-10-28",
9.0,
"NVIDIA Nemotron Nano 12B v2 VL",
"NVIDIA"
],
[
"2025-10-29",
19.0,
"Nova 2.0 Lite",
"Amazon"
],
[
"2025-10-30",
8.5,
"Kimi Linear 48B A3B Instruct",
"Kimi"
],
[
"2025-11-03",
25.0,
"Qwen3 Max Thinking",
"Alibaba"
],
[
"2025-11-06",
32.7,
"Kimi K2 Thinking",
"Kimi"
],
[
"2025-11-11",
26.0,
"Doubao Seed Code",
"ByteDance"
],
[
"2025-11-11",
28.3,
"KAT-Coder-Pro V1",
"KwaiKAT"
],
[
"2025-11-13",
21.9,
"ERNIE 5.0 Thinking Preview",
"Baidu"
],
[
"2025-11-13",
30.6,
"GPT-5.1 Codex mini",
"OpenAI"
],
[
"2025-11-13",
34.7,
"GPT-5.1 Codex",
"OpenAI"
],
[
"2025-11-13",
36.9,
"GPT-5.1",
"OpenAI"
],
[
"2025-11-18",
39.6,
"Gemini 3 Pro Preview",
"Google"
],
[
"2025-11-19",
30.6,
"Grok 4.1 Fast",
"xAI"
],
[
"2025-11-20",
2.8,
"Olmo 3 7B Instruct",
"Allen Institute for AI"
],
[
"2025-11-20",
4.0,
"Olmo 3 7B Think",
"Allen Institute for AI"
],
[
"2025-11-20",
6.4,
"Olmo 3 32B Think",
"Allen Institute for AI"
],
[
"2025-11-24",
40.8,
"Claude Opus 4.5",
"Anthropic"
],
[
"2025-11-25",
20.5,
"Apriel-v1.6-15B-Thinker",
"ServiceNow"
],
[
"2025-11-26",
20.9,
"Nova 2.0 Omni",
"Amazon"
],
[
"2025-11-27",
15.6,
"INTELLECT-3",
"Prime Intellect"
],
[
"2025-11-27",
21.8,
"Nova 2.0 Pro Preview",
"Amazon"
],
[
"2025-12-01",
22.2,
"DeepSeek V3.2 Speciale",
"DeepSeek"
],
[
"2025-12-01",
32.0,
"DeepSeek V3.2",
"DeepSeek"
],
[
"2025-12-02",
6.3,
"Ministral 3 3B",
"Mistral"
],
[
"2025-12-02",
9.0,
"Ministral 3 8B",
"Mistral"
],
[
"2025-12-02",
11.1,
"Ministral 3 14B",
"Mistral"
],
[
"2025-12-02",
15.9,
"Mistral Large 3",
"Mistral"
],
[
"2025-12-04",
12.8,
"Motif-2-12.7B-Reasoning",
"Motif Technologies"
],
[
"2025-12-05",
14.2,
"K2-V2",
"MBZUAI Institute of Foundation Models"
],
[
"2025-12-08",
16.8,
"GLM-4.6V",
"Z.ai"
],
[
"2025-12-09",
17.4,
"Devstral Small 2",
"Mistral"
],
[
"2025-12-09",
19.2,
"Devstral 2",
"Mistral"
],
[
"2025-12-11",
2.0,
"Molmo2-8B",
"Allen Institute for AI"
],
[
"2025-12-11",
16.4,
"Mi:dm K 2.5 Pro",
"Korea Telecom"
],
[
"2025-12-11",
40.1,
"GPT-5.2 Codex",
"OpenAI"
],
[
"2025-12-11",
42.2,
"GPT-5.2",
"OpenAI"
],
[
"2025-12-12",
8.1,
"Olmo 3.1 32B Think",
"Allen Institute for AI"
],
[
"2025-12-15",
14.2,
"NVIDIA Nemotron 3 Nano 30B A3B",
"NVIDIA"
],
[
"2025-12-15",
17.3,
"K2 Think V2",
"MBZUAI Institute of Foundation Models"
],
[
"2025-12-16",
33.2,
"MiMo-V2-Flash",
"Xiaomi"
],
[
"2025-12-17",
15.1,
"Solar Open 100B",
"Upstage"
],
[
"2025-12-17",
37.8,
"Gemini 3 Flash Preview",
"Google"
],
[
"2025-12-22",
33.7,
"GLM-4.7",
"Z.ai"
],
[
"2025-12-23",
31.4,
"MiniMax-M2.1",
"MiniMax"
],
[
"2025-12-26",
17.0,
"HyperCLOVA X SEED Think",
"Naver"
],
[
"2025-12-31",
22.1,
"K-EXAONE",
"LG AI Research"
],
[
"2026-01-04",
9.8,
"Falcon-H1R-7B",
"TII UAE"
],
[
"2026-01-05",
1.0,
"LFM2.5-VL-1.6B",
"Liquid AI"
],
[
"2026-01-05",
2.7,
"LFM2.5-1.2B-Instruct",
"Liquid AI"
],
[
"2026-01-13",
6.5,
"Olmo 3.1 32B Instruct",
"Allen Institute for AI"
],
[
"2026-01-19",
22.9,
"GLM-4.7-Flash",
"Z.ai"
],
[
"2026-01-20",
2.7,
"LFM2.5-1.2B-Thinking",
"Liquid AI"
],
[
"2026-01-20",
9.5,
"Step3 VL 10B",
"StepFun"
],
[
"2026-01-26",
31.7,
"Qwen3 Max Thinking",
"Alibaba"
],
[
"2026-01-27",
35.4,
"Kimi K2.5",
"Kimi"
],
[
"2026-01-28",
17.2,
"LongCat Flash Lite",
"LongCat"
],
[
"2026-02-02",
25.5,
"Step 3.5 Flash",
"StepFun"
],
[
"2026-02-03",
21.1,
"Qwen3 Coder Next",
"Alibaba"
],
[
"2026-02-05",
43.7,
"Claude Opus 4.6",
"Anthropic"
],
[
"2026-02-05",
44.3,
"GPT-5.3 Codex",
"OpenAI"
],
[
"2026-02-10",
12.4,
"Tri-21B-Think",
"Trillion Labs"
],
[
"2026-02-10",
13.6,
"Tri-21B-think Preview",
"Trillion Labs"
],
[
"2026-02-11",
11.1,
"Nanbeige4.1-3B",
"Nanbeige"
],
[
"2026-02-11",
39.5,
"GLM-5",
"Z.ai"
],
[
"2026-02-12",
33.7,
"MiniMax-M2.5",
"MiniMax"
],
[
"2026-02-16",
33.7,
"Qwen3.5 397B A17B",
"Alibaba"
],
[
"2026-02-17",
1.0,
"Tiny Aya Global",
"Cohere"
],
[
"2026-02-17",
47.2,
"Claude Sonnet 4.6",
"Anthropic"
],
[
"2026-02-19",
46.5,
"Gemini 3.1 Pro Preview",
"Google"
],
[
"2026-02-20",
21.4,
"Mercury 2",
"Inception"
],
[
"2026-02-24",
29.3,
"Qwen3.5 35B A3B",
"Alibaba"
],
[
"2026-02-24",
32.3,
"Qwen3.5 122B A10B",
"Alibaba"
],
[
"2026-02-24",
33.8,
"Qwen3.5 27B",
"Alibaba"
],
[
"2026-02-25",
4.9,
"LFM2 24B A2B",
"Liquid AI"
],
[
"2026-03-02",
5.3,
"Qwen3.5 0.8B",
"Alibaba"
],
[
"2026-03-02",
7.1,
"Qwen3.5 2B",
"Alibaba"
],
[
"2026-03-02",
20.1,
"Qwen3.5 4B",
"Alibaba"
],
[
"2026-03-02",
21.4,
"Qwen3.5 9B",
"Alibaba"
],
[
"2026-03-03",
25.0,
"Gemini 3.1 Flash-Lite",
"Google"
],
[
"2026-03-05",
51.4,
"GPT-5.4",
"OpenAI"
],
[
"2026-03-06",
6.6,
"Sarvam 30B",
"Sarvam"
],
[
"2026-03-06",
11.9,
"Sarvam 105B",
"Sarvam"
],
[
"2026-03-10",
36.5,
"Grok 4.20 0309",
"xAI"
],
[
"2026-03-11",
25.4,
"NVIDIA Nemotron 3 Super 120B A12B",
"NVIDIA"
],
[
"2026-03-15",
38.1,
"GLM-5-Turbo",
"Z.ai"
],
[
"2026-03-16",
8.7,
"NVIDIA Nemotron 3 Nano 4B",
"NVIDIA"
],
[
"2026-03-16",
19.6,
"Mistral Small 4",
"Mistral"
],
[
"2026-03-17",
38.2,
"GPT-5.4 nano",
"OpenAI"
],
[
"2026-03-17",
40.0,
"GPT-5.4 mini",
"OpenAI"
],
[
"2026-03-18",
38.1,
"MiniMax-M2.7",
"MiniMax"
],
[
"2026-03-18",
40.3,
"MiMo-V2-Pro",
"Xiaomi"
],
[
"2026-03-19",
17.6,
"Nemotron Cascade 2 30B A3B",
"NVIDIA"
],
[
"2026-03-19",
35.0,
"MiMo-V2-Omni",
"Xiaomi"
],
[
"2026-03-27",
33.7,
"KAT Coder Pro V2",
"KwaiKAT"
],
[
"2026-03-27",
36.4,
"MiMo-V2-Omni-0327",
"Xiaomi"
],
[
"2026-03-30",
19.0,
"Qwen3.5 Omni Flash",
"Alibaba"
],
[
"2026-03-30",
30.6,
"Qwen3.5 Omni Plus",
"Alibaba"
],
[
"2026-04-01",
18.2,
"Trinity Large Thinking",
"Arcee AI"
],
[
"2026-04-01",
34.5,
"GLM 5V Turbo",
"Z.ai"
],
[
"2026-04-02",
9.4,
"Gemma 4 E2B",
"Google"
],
[
"2026-04-02",
25.7,
"Gemma 4 26B A4B",
"Google"
],
[
"2026-04-02",
26.0,
"Step 3.5 Flash 2603",
"StepFun"
],
[
"2026-04-02",
29.4,
"Gemma 4 31B",
"Google"
],
[
"2026-04-02",
39.6,
"Qwen3.6 Plus",
"Alibaba"
],
[
"2026-04-03",
11.9,
"Gemma 4 E4B",
"Google"
],
[
"2026-04-06",
14.1,
"Solar Pro 3",
"Upstage"
],
[
"2026-04-07",
37.0,
"Grok 4.20 0309 v2",
"xAI"
],
[
"2026-04-07",
40.2,
"GLM-5.1",
"Z.ai"
],
[
"2026-04-08",
43.1,
"Muse Spark",
"Meta"
],
[
"2026-04-09",
20.2,
"EXAONE 4.5 33B",
"LG AI Research"
],
[
"2026-04-15",
18.5,
"JT-MINI",
"China Mobile"
],
[
"2026-04-16",
31.6,
"Qwen3.6 35B A3B",
"Alibaba"
],
[
"2026-04-16",
53.5,
"Claude Opus 4.7",
"Anthropic"
],
[
"2026-04-20",
40.0,
"Qwen3.6 Max Preview",
"Alibaba"
],
[
"2026-04-20",
44.2,
"Kimi K2.6",
"Kimi"
],
[
"2026-04-21",
14.1,
"Ling 2.6 Flash",
"InclusionAI"
],
[
"2026-04-22",
37.1,
"Qwen3.6 27B",
"Alibaba"
],
[
"2026-04-22",
37.2,
"MiMo-V2.5",
"Xiaomi"
],
[
"2026-04-22",
42.2,
"MiMo-V2.5-Pro",
"Xiaomi"
],
[
"2026-04-23",
26.1,
"Ling-2.6-1T",
"InclusionAI"
],
[
"2026-04-23",
33.6,
"Hy3-preview",
"Tencent"
],
[
"2026-04-23",
54.8,
"GPT-5.5",
"OpenAI"
],
[
"2026-04-24",
40.3,
"DeepSeek V4 Flash",
"DeepSeek"
],
[
"2026-04-24",
44.3,
"DeepSeek V4 Pro",
"DeepSeek"
],
[
"2026-04-29",
4.7,
"Granite 4.1 3B",
"IBM"
],
[
"2026-04-29",
6.7,
"Granite 4.1 8B",
"IBM"
],
[
"2026-04-29",
8.9,
"Granite 4.1 30B",
"IBM"
],
[
"2026-04-29",
14.9,
"Nemotron 3 Nano Omni 30B A3B Reasoning",
"NVIDIA"
],
[
"2026-04-29",
29.9,
"Mistral Medium 3.5",
"Mistral"
],
[
"2026-04-30",
37.6,
"Grok 4.3",
"xAI"
],
[
"2026-05-05",
33.5,
"GPT-5.5 Instant",
"OpenAI"
],
[
"2026-05-08",
30.6,
"Ring-2.6-1T",
"InclusionAI"
],
[
"2026-05-11",
4.2,
"MiniCPM-V 4.6 1.3B",
"OpenBMB"
],
[
"2026-05-14",
28.4,
"JT-35B-Flash",
"China Mobile"
],
[
"2026-05-19",
46.0,
"Qwen3.7 Max",
"Alibaba"
],
[
"2026-05-19",
50.2,
"Gemini 3.5 Flash",
"Google"
],
[
"2026-05-20",
22.5,
"Command A+",
"Cohere"
],
[
"2026-05-25",
12.0,
"MiniCPM5-1B",
"OpenBMB"
],
[
"2026-05-26",
17.8,
"HyperNova 60B 2605",
"Multiverse Computing"
],
[
"2026-05-28",
8.3,
"LFM2.5-8B-A1B",
"Liquid AI"
],
[
"2026-05-28",
55.7,
"Claude Opus 4.8",
"Anthropic"
],
[
"2026-05-29",
30.3,
"Step 3.7 Flash",
"StepFun"
],
[
"2026-06-01",
39.0,
"Qwen3.7 Plus",
"Alibaba"
],
[
"2026-06-01",
44.4,
"MiniMax-M3",
"MiniMax"
],
[
"2026-06-02",
41.0,
"Nex-N2-Pro",
"Nex AGI"
],
[
"2026-06-03",
21.8,
"Gemma 4 12B",
"Google"
],
[
"2026-06-04",
37.8,
"Nemotron 3 Ultra 550B A55B",
"NVIDIA"
],
[
"2026-06-09",
19.8,
"North Mini Code",
"Cohere"
],
[
"2026-06-09",
59.9,
"Claude Fable 5",
"Anthropic"
],
[
"2026-06-10",
13.5,
"DiffusionGemma 26B A4B",
"Google"
],
[
"2026-06-12",
41.9,
"Kimi K2.7 Code",
"Kimi"
],
[
"2026-06-16",
39.8,
"Grok Build 0.1 0616",
"xAI"
],
[
"2026-06-16",
51.1,
"GLM-5.2",
"Z.ai"
],
[
"2026-06-25",
28.9,
"GPT-5.5 Instant",
"OpenAI"
],
[
"2026-06-29",
33.5,
"LongCat 2.0",
"LongCat"
],
[
"2026-06-30",
53.4,
"Claude Sonnet 5",
"Anthropic"
],
[
"2026-07-06",
41.2,
"Hy3",
"Tencent"
],
[
"2026-07-08",
53.8,
"Grok 4.5",
"xAI"
],
[
"2026-07-09",
38.8,
"JT-4.1 Flash 236B A21B",
"China Mobile"
],
[
"2026-07-09",
50.6,
"Muse Spark 1.1",
"Meta"
],
[
"2026-07-09",
51.2,
"GPT-5.6 Luna",
"OpenAI"
],
[
"2026-07-09",
55.0,
"GPT-5.6 Terra",
"OpenAI"
],
[
"2026-07-09",
58.9,
"GPT-5.6 Sol",
"OpenAI"
],
[
"2026-07-14",
44.1,
"Motif 3",
"Motif Technologies"
],
[
"2026-07-15",
40.7,
"Inkling",
"Thinking Machines"
],
[
"2026-07-16",
57.1,
"Kimi K3",
"Kimi"
],
[
"2026-07-21",
36.5,
"Gemini 3.5 Flash-Lite",
"Google"
],
[
"2026-07-21",
50.1,
"Gemini 3.6 Flash",
"Google"
],
[
"2026-07-23",
16.1,
"G9v3-3B",
"AI9Stars"
],
[
"2026-07-24",
38.8,
"Agnes 2.5 Pro Alpha",
"Sapiens AI"
],
[
"2026-07-24",
60.7,
"Claude Opus 5",
"Anthropic"
]
]

L = {'ylabel': 'Zekâ Endeksi (Artificial Analysis)', 'title': 'Yapay Zeka Modellerinin Başarımı — Tek Sayıyla', 'sub': '{n} modelin bağımsız ölçümü ({lo} – {hi})  ·  sarı merdiven: o güne kadarki en iyi', 'growth': 'Son 12 ayda\n{a:.0f} → {b:.0f}  ({k:.1f}×)', 'cloud': 'diğer ölçülen modeller', 'credit': 'Kaynak: artificialanalysis.ai  ·  Derleyen: Prof. Dr. Oğuz Ergin'}

COLORS = {"Anthropic": "#d97757", "OpenAI": "#10a37f", "Google": "#4285F4", "xAI": "#1da1f2",
          "Meta": "#0668E1", "DeepSeek": "#ef4444", "Alibaba": "#7C3AED", "Moonshot": "#14B8A6",
          "Z.ai": "#BE185D", "MiniMax": "#C77DFF", "Mistral": "#fa8005", "ByteDance": "#22D3EE",
          "Microsoft": "#F25022", "Amazon": "#ff9900", "NVIDIA": "#76b900"}
OTHER = "#4a5160"

df = pd.DataFrame(DATA, columns=["date", "ii", "name", "comp"])
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

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")

# arka plan bulutu
ax.scatter(df["Date"], df["ii"], s=34, c="#262c36", alpha=.85, edgecolors="none", zorder=1)

# sinir merdiveni
ax.step(fr["Date"], fr["ii"], where="post", color="#ffd166", lw=3.0, zorder=3, alpha=.95)
ax.fill_between(fr["Date"], fr["ii"], step="post", color="#ffd166", alpha=.05, zorder=2)
for _, r in fr.iterrows():
    ax.scatter([r["Date"]], [r["ii"]], s=230, c=COLORS.get(r["comp"], OTHER),
               edgecolors="white", linewidths=2.0, zorder=5)

# sinir etiketleri: PIKSEL uzayinda 2 boyutlu cakisma kontrolu
# (kademe farki tek basina yetmiyor: noktalarin kendi yuksekligi de degisiyor)
PX_DAY = (26 * 105 * 0.93) / max(1, (df["Date"].max() - df["Date"].min()).days)
YLIM = fr["ii"].max() * 1.20
PX_UNIT = (14 * 105 * 0.78) / YLIM
TIERS = [26, -34, 66, -74, 106, -114, 146, -154]
x0 = df["Date"].min().toordinal()
boxes = []
for _, r in fr.iterrows():
    cx = (r["Date"].toordinal() - x0) * PX_DAY
    hw = len(r["name"]) * 4.8 + 18
    tier = TIERS[-1]
    for t in TIERS:
        cy = r["ii"] * PX_UNIT + t
        if all(abs(cx - bx) > (hw + bw) or abs(cy - by) > 34 for bx, by, bw in boxes):
            tier = t; break
    boxes.append((cx, r["ii"] * PX_UNIT + tier, hw))
    ax.annotate(r["name"], (r["Date"], r["ii"]), xytext=(0, tier), textcoords="offset points",
                fontsize=13, color="#e6edf3", fontweight="bold", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.30", fc="#161b22", ec=COLORS.get(r["comp"], OTHER), lw=1.6, alpha=.96),
                arrowprops=dict(arrowstyle="-", color=COLORS.get(r["comp"], OTHER), lw=1.1, alpha=.55,
                                shrinkA=2, shrinkB=6))

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
ax.text(0.015, 0.965, L["growth"].format(a=bir_yil, b=son["ii"], k=son["ii"] / bir_yil),
        transform=ax.transAxes, ha="left", va="top", fontsize=17, color="#ffd166", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", fc="#161b22", ec="#ffd166", lw=1.8, alpha=.95))

# lejant (sinirdaki sirketler)
import matplotlib.lines as mlines
comps = list(dict.fromkeys(fr["comp"]))
handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=13, markerfacecolor=COLORS.get(c, OTHER),
                         markeredgecolor="white", label=c) for c in comps]
handles.append(mlines.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="#262c36",
                             markeredgecolor="none", label=L["cloud"]))
ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#161b22", edgecolor="#30363d",
          fontsize=14, labelcolor="#c9d1d9", ncol=2)

ax.text(0.995, -0.115, L["credit"], transform=ax.transAxes, ha="right", fontsize=13,
        color="#6e7681", style="italic")
plt.tight_layout()
plt.savefig("intelligence_index_tr.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: intelligence_index_tr.png  |  model:", len(df), " sinir:", len(fr))
