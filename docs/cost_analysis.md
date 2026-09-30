# Query Cost Analysis

**Measured:** 30 Sep 2026, against the live system (ReAssure 2.0 / 3.0 wording + CIS indexed, S3 Vectors `us-east-1`, Databricks pay-per-token).
**Scope:** the per-query cost of `POST /query`. Ingestion and fixed infrastructure are summarised at the end.

---

## TL;DR

| | Per query | Per 1,000 queries |
|---|---|---|
| **Typical query** (mean of 20 measured) | **~$0.0015** | **~$1.49** |
| Median query | ~$0.0014 | ~$1.36 |
| List question with a long answer | ~$0.0032 | ~$3.23 |

- **The LLM is ~99.7% of the cost:** input 77%, output 23%. Embedding and vector search are rounding errors.
- **Output is 23% of the cost**, driven by list questions, which produce 300–900 tokens.
- **Switching models moves the cost the most:** from $0.35 (GPT-6 Luna) or $0.48 (gpt-oss-120b, on the same Databricks account) up to $5.93 (Grok 4.7) per 1,000 typical queries. See section 3b.

---

## 1. Typical query: input tokens

Every query sends one prompt to Llama 3.3 70B. The token counts are what Databricks reported for real requests (`usage.prompt_tokens` / `completion_tokens`), not estimates. Policy prefix `ReAssure 3.0` searches the wording and the CIS, and likewise for 2.0.

| Question | Policy | Input tokens | Output tokens | Parents sent |
|---|---|---:|---:|---:|
| which expenses are not covered in this policy | ReAssure 3.0 | 3,765 | 895 | 10 |
| what are the standard exclusions | ReAssure 3.0 | 2,419 | 285 | 9 |
| what are the specific exclusions | ReAssure 3.0 | 2,163 | 415 | 9 |
| what is the waiting period for cataract | ReAssure 3.0 | 2,875 | 67 | 9 |
| what is the room rent limit | ReAssure 3.0 | 512 | 67 | 2 |
| how do I cancel the policy | ReAssure 3.0 | 1,870 | 97 | 8 |
| what is the free look period | ReAssure 3.0 | 2,322 | 117 | 8 |
| what is the moratorium period | ReAssure 3.0 | 2,761 | 67 | 10 |
| what is the definition of a hospital | ReAssure 3.0 | 1,471 | 196 | 10 |
| is air ambulance covered and up to what limit | ReAssure 3.0 | 3,894 | 290 | 9 |
| which expenses are not covered in this policy | ReAssure 2.0 | 1,582 | 760 | 10 |
| what are the standard exclusions | ReAssure 2.0 | 2,618 | 269 | 9 |
| what are the specific exclusions | ReAssure 2.0 | 2,469 | 334 | 9 |
| what is the waiting period for cataract | ReAssure 2.0 | 3,085 | 51 | 9 |
| what is the room rent limit | ReAssure 2.0 | 2,658 | 43 | 10 |
| how do I cancel the policy | ReAssure 2.0 | 2,091 | 106 | 9 |
| what is the free look period | ReAssure 2.0 | 1,945 | 86 | 7 |
| what is the moratorium period | ReAssure 2.0 | 2,870 | 84 | 10 |
| what is the definition of a hospital | ReAssure 2.0 | 1,593 | 198 | 10 |
| is air ambulance covered and up to what limit | ReAssure 2.0 | 805 | 78 | 10 |
| **Mean** | | **2,288** | **225** | 9 |
| **Median** | | **2,371** | **112** | 9 |

**What drives input size:** each matched sub-clause is sent as its own parent, 20–300 tokens including its clause path. Most queries send 8–10 small parents, 0.5K–3.9K tokens in total.

**Where large inputs come from:** tables, lists and CIS rows are sent whole. For example, the Annexure III critical illness table is ~3.9K tokens.

**Output:** single-fact questions produce 40–200 tokens. List questions produce 270–900.

---

## 2. Breakdown by component

### 2a. Where the input tokens go

Measured by re-sending each prompt in parts (system prompt only, plus the question, plus the context tags with empty clauses) and taking the differences:

| Component | Mean tokens | Share of input |
|---|---:|---:|
| Retrieved clause text (incl. clause path lines) | 1,829 | **79.9%** |
| System prompt (rules + chat template) | 248 | 10.8% |
| Context tags (`<context>`, `<document clause=… pages=…>`) | 198 | 8.6% |
| User question (+ `<question>` tags) | 14 | 0.6% |
| **Total** | **2,288** | 100% |

Each of the ~9 parents sent has its own `<document clause pages>` tag. The fixed overhead (system prompt + tags + question ≈ 460 tokens) is about a fifth of the input.

### 2b. Where the money goes (typical query)

| Component | Service | Unit price | Usage per query | Cost per query | Share |
|---|---|---|---|---:|---:|
| LLM input | Databricks, Llama 3.3 70B | $0.50 / 1M tokens | 2,288 tokens | $0.001144 | **77.0%** |
| LLM output | Databricks, Llama 3.3 70B | $1.50 / 1M tokens | 225 tokens | $0.000338 | **22.7%** |
| Query vector search | S3 Vectors `QueryVectors` | $2.50 / 1M requests (+ data processed) | 1 request | $0.0000025 | 0.17% |
| Query embedding | Databricks, GTE Large | $0.13 / 1M tokens | ~14 tokens | $0.0000018 | 0.12% |
| Postgres lookups | self-hosted / RDS | fixed cost | 2 small queries | – | – |
| **Total** | | | | **~$0.00149** | 100% |

Notes on the smaller lines:
- **S3 Vectors data processing** is charged on the size of the index searched ($0.004/TB under 100K vectors). With about 1,000 vectors of ~4.3 KB each, that's around $10⁻⁸ per query. The data returned (10 results, a few KB) is inside the free 512 KB per query.
- **Cross-region transfer** (API in `ap-south-1`, services in the US): each query sends a few tens of KB out. At typical egress rates that's well under $0.01 per 1,000 queries.

---

## 3. Total cost for 1,000 requests

| Scenario | Input / output tokens per query | **Total / 1,000** |
|---|---|---:|
| **Typical (measured mean)** | 2,288 / 225 | **≈ $1.49** |
| Median question | 2,371 / 112 | ≈ $1.36 |
| Largest input observed | 3,894 / 290 | ≈ $2.39 |
| List question with a long answer | 3,765 / 895 | ≈ $3.23 |
| *What-if: Bedrock Claude Haiku 4.5 at $1 / $5 per 1M* ¹ | 2,288 / 225 | *≈ $3.42* |
| *What-if: GPT-6 Luna* ² | 2,288 / 225 | *≈ $0.35* |

Embedding and vector search add about $0.004 per 1,000 queries in every scenario (already included).

¹ Anthropic API list price for Claude Haiku 4.5. The Bedrock on-demand rate for Haiku 4.5 wasn't shown on the AWS pricing page when checked, so confirm it. Claude's tokenizer also counts the same text differently from Llama's.

² See section 3a. GPT-6 Luna is not available in the current Databricks workspace. It would need an external-model endpoint or a new `LLMClient`.

**Scaling:** the cost is linear in query volume. 100K queries/month ≈ $150 at the typical rate. Fixed costs are extra (section 5).

### 3a. GPT-6 Luna compared with Llama 3.3 70B

GPT-6 Luna (OpenAI, standard tier): **$0.10 / 1M input** and **$0.50 / 1M output**. That's **80% cheaper on input** and **67% cheaper on output** than Llama 3.3 on Databricks ($0.50 / $1.50), so it's cheaper for every kind of question.

Per 1,000 queries, using the same measured token counts (embedding + search included):

| Scenario (input / output tokens) | Llama 3.3 70B (current) | GPT-6 Luna | Saving | GPT-6 Luna, +500 output tokens / query ³ | GPT-6 Luna, +1,000 output tokens / query ³ |
|---|---:|---:|---:|---:|---:|
| **Typical** (2,288 / 225) | **$1.49** | **$0.35** | **−77%** | $0.60 | $0.85 |
| Median (2,371 / 112) | $1.36 | $0.30 | −78% | $0.55 | $0.80 |
| Largest input (3,894 / 290) | $2.39 | $0.54 | −77% | $0.79 | $1.04 |
| List question, long answer (3,765 / 895) | $3.23 | $0.83 | −74% | $1.08 | $1.33 |

At 100K queries/month: **≈ $35** with GPT-6 Luna, against ≈ $149 with Llama 3.3.

**What this shows:**
- **About 4× cheaper across the board.** The saving barely changes between short and long answers, because both input and output are much cheaper.
- **The cost split stays similar:** about 66% input and 33% output. Embedding and vector search become ~1% of the total, which is still negligible.
- **Even 1,000 extra output tokens per query** (e.g. if the model produces reasoning tokens) leaves it 41–59% cheaper than Llama 3.3 today.
- **Cached input ($0.01 / 1M) and batch pricing (50% off) exist,** but help little here. The only prefix shared by every request is the ~250-token system prompt, and batch doesn't suit interactive chat.

**Not covered here:**
- **Reasoning tokens:** it isn't confirmed whether GPT-6 Luna produces billed reasoning tokens on this kind of request. If it does, use the right-hand columns as a guide and measure the real output count.
- **Tokenizer:** GPT-6 Luna counts the same text differently from Llama. These figures reuse Llama's measured counts, so expect a ±10–20% difference until measured with the model itself.
- **Integration:** not in your Databricks workspace. It would need a Databricks *external model* endpoint pointing at OpenAI (then only `DATABRICKS_LLM_MODEL` changes), or a new `LLMClient`. Policyholder data would then go to OpenAI.
- **Answer quality and latency:** not tested. Compare answers side by side before switching.

³ Illustration, not a measurement.

### 3b. Other LLMs compared

The same measured token counts, priced per model. The prices are list prices for standard (non-batch, uncached) requests as of 30 Sep 2026. Embedding and vector search (~$0.004 per 1,000) are included. Sorted by typical cost.

| Model | Input $/1M | Output $/1M | **Typical / 1,000** | Median / 1,000 | List question / 1,000 | vs current | Output share |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Luna (OpenAI) | $0.10 | $0.50 | **$0.35** | $0.30 | $0.83 | −77% | 33% |
| GLM-5.3-Flash (Z.ai) ⁵ | $0.15 | $0.50 | **$0.46** | $0.42 | $1.02 | −69% | 24% |
| gpt-oss-120b (Databricks / Groq / Together) | $0.15 | $0.60 | **$0.48** | $0.43 | $1.11 | −68% | 28% |
| DeepSeek V4.1 Flash, off-peak ⁶ | $0.15 | $0.60 | **$0.48** | $0.43 | $1.11 | −68% | 28% |
| Gemini 3.1 Flash-Lite | $0.25 | $1.50 | **$0.91** | $0.76 | $2.29 | −38% | 37% |
| DeepSeek V4.1 Flash, peak ⁶ | $0.30 | $1.20 | **$0.96** | $0.85 | $2.21 | −35% | 28% |
| Gemini 2.5 Flash, thinking off | $0.30 | $2.50 | **$1.25** | $0.99 | $3.37 | −16% | 45% |
| **Llama 3.3 70B, Databricks (current)** | **$0.50** | **$1.50** | **$1.49** | **$1.36** | **$3.23** | – | 23% |
| Gemini 3.6 Flash (intro price, to 31 Dec 2026) ⁷ | $0.75 | $3.75 | **$2.57** | $2.20 | $6.18 | +73% | 33% |
| Gemini 3.8 Flash (intro price, to 31 Dec 2026) ⁷ | $0.75 | $3.75 | **$2.57** | $2.20 | $6.18 | +73% | 33% |
| Claude Haiku 4.5 (Anthropic list price) | $1.00 | $5.00 | **$3.42** | $2.93 | $8.24 | +130% | 33% |
| Gemini 3.6 / 3.8 Flash (from 1 Jan 2027) ⁷ | $1.50 | $7.50 | **$5.13** | $4.40 | $12.36 | +245% | 33% |
| Grok 4.7 (xAI, prompts < 200K) | $2.00 | $6.00 | **$5.93** | $5.41 | $12.90 | +299% | 23% |

At 100K queries/month, the typical cost ranges from about **$35** (GPT-6 Luna) to about **$593** (Grok 4.7), against **$149** today.

**What this shows:**
- **Four models cut the current LLM bill by about two thirds or more:** GPT-6 Luna, GLM-5.3-Flash, gpt-oss-120b and off-peak DeepSeek V4.1 Flash. At this point retrieval, not the model, stops being a meaningful cost driver.
- **gpt-oss-120b is the easiest to try.** It's served on Databricks, the platform already in use, so switching should mainly be a matter of changing `DATABRICKS_LLM_MODEL`. It's a reasoning model, though, so check how its streamed output separates reasoning from the answer before relying on the existing client.
- **Gemini 3.6 / 3.8 Flash cost 73% more than today at their introductory price, and about 3.5× from 2027.**
- **Grok 4.7 and Haiku 4.5 are the most expensive options** for this workload.

**Before choosing on price:**
- **Reasoning / thinking tokens are billed as output** by most of these models (gpt-oss-120b, the Gemini Flash models, DeepSeek and GLM thinking modes). The table assumes none. Set thinking or reasoning effort to its lowest setting, and measure the real output count: 500 extra output tokens per query adds $0.25–$3.75 per 1,000, depending on the model's output price.
- **Tokenizers differ**, so each model counts the same prompt differently. Expect roughly ±10–20% until measured with the model itself.
- **Integration:** only Databricks-hosted models (Llama 3.3 today, gpt-oss-120b) and Bedrock (Claude) have clients in the code. The others need a new `LLMClient`.
- **Data residency and compliance:** the DeepSeek and Z.ai (GLM) APIs are operated from China. For policyholder questions, check whether that's acceptable, or use a host that serves the same open-weight models from elsewhere, at different prices.
- **Answer quality and latency** were not tested for any of these. Output speed (tokens/second) matters as much as price, because generation is most of the response time.

⁵ GLM-5.3-Flash: a price tracker describes a "promo cliff", i.e. an introductory rate that may end. Confirm the current rate with Z.ai.
⁶ DeepSeek V4.1 Flash peak hours are Mon–Fri 09:00–12:00 and 14:00–18:00 Beijing time, which is **06:30–09:30 and 11:30–15:30 IST**. Most Indian business-hours traffic would pay the peak rate; evenings and weekends get off-peak.
⁷ Gemini 3.6 Flash and 3.8 Flash are both listed at an introductory $0.75 / $3.75 until 31 Dec 2026, rising to $1.50 / $7.50 on 1 Jan 2027.

---

## 4. Cost levers, ranked

| Lever | Effect on cost | Trade-off / status |
|---|---|---|
| Trim output on list questions (summarise by category) | Output is 23% of cost; list answers use 300–900 tokens | Less detail in answers |
| Shorten the system prompt / context tags | ~460 fixed tokens ≈ 15% of the typical cost | Small |
| Change LLM model | From −77% (GPT-6 Luna) and −68% (gpt-oss-120b on Databricks) to +299% (Grok 4.7). See section 3b | gpt-oss-120b needs no new client; the others do. Quality, latency and data residency still to be compared |
| Cache repeated questions | Up to −100% for exact repeats | Only helps repeated questions |
| Change embedding model or vector store | < 0.3% of cost | Not worth it for cost alone |

---

## 5. Not included in per-query cost

**One-time per uploaded document** (e.g. ReAssure 3.0 wording: 313 chunks):
- **Embedding the chunks:** ~40K tokens × $0.13/1M ≈ **$0.005**.
- **S3 Vectors writes:** ~16 PUT calls (batches of 20, 128 KB minimum billed each) ≈ **$0.0004**.
- **Docling parsing:** runs on the API host's CPU (no API charge).

**Fixed / monthly:**
- **S3 Vectors storage:** $0.06/GB-month. 10K vectors ≈ 43 MB ≈ $0.003/month.
- **Raw PDFs and reports in S3:** a few MB per document, negligible.
- **PostgreSQL** (e.g. RDS), and hosting for the API and Streamlit. These are fixed monthly instance costs.
- **Databricks:** pay-per-token has no idle cost. Provisioned throughput, if adopted for latency, is billed per hour instead.

---

## Method, sources and caveats

**Method:**
- The same 10 questions were sent through the real `Generator` (`build_default()`) against policy prefixes `ReAssure 3.0` and `ReAssure 2.0` (20 queries), with provider-reported token usage from the `done` event.
- The component split came from sending three extra one-token requests per question (system + question, system + tags with empty clauses) and subtracting.
- Script: `cost_measure.py` (session scratchpad, not committed).

**Prices** (checked 30 Sep 2026):
- [Databricks Foundation Model Serving pricing](https://www.databricks.com/product/pricing/foundation-model-serving): Llama 3.3 70B at 7.143 DBU / 1M input and 21.429 DBU / 1M output tokens; GTE at 1.857 DBU / 1M tokens.
- **DBU rate:** $0.07/DBU for model serving (US East, Premium). Source: [Databricks AI Search pricing](https://www.databricks.com/product/pricing/ai-search), where 28.57 DBU = $2.00 at $0.07/DBU in US East, and [mammoth.io's summary of DBU rates](https://mammoth.io/blog/databricks-pricing/).
- **Rate assumption:** Enterprise tiers, committed-use discounts and other regions change the $/DBU. Every Databricks line above scales linearly with it.
- [Amazon S3 pricing, S3 Vectors section](https://aws.amazon.com/s3/pricing/) (US East): storage $0.06/GB-month, PUT $0.20/GB, $2.50 per 1M query requests, data processing $0.004/TB under 100K vectors, first 512 KB returned per query free.
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing): Gemini 2.5 Flash, paid standard tier, $0.30 / 1M input (text) and $2.50 / 1M output, **including thinking tokens**. The batch tier is half price but isn't usable for interactive chat.
- **Claude Haiku 4.5:** $1 / $5 per 1M tokens, Anthropic API list price. The [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) didn't list Haiku 4.5 when checked.
- **Section 3b models:** taken from third-party price trackers, not the providers' own pages, so **confirm each with the provider before deciding**. "groq 4.7" was taken to mean xAI's Grok 4.7.
  - GPT-6 Luna: [eesel.ai](https://www.eesel.ai/blog/gpt-6-luna-pricing), standard tier $0.10 / $0.50; cached input $0.01; batch 50% off
  - Gemini 3.1 Flash-Lite: [pricepertoken](https://pricepertoken.com/pricing-page/model/google-gemini-3.1-flash-lite)
  - gpt-oss-120b: [computeprices (Groq)](https://computeprices.com/providers/groq/models/gpt-oss-120b); the same rate is reported for Databricks and Together
  - Gemini 3.6 Flash: [anotherwrapper](https://anotherwrapper.com/llm-pricing/gemini-3.6-flash)
  - Gemini 3.8 Flash: [eesel.ai](https://eesel.ai/blog/gemini-3-8-flash-pricing)
  - Grok 4.7: [eesel.ai](https://www.eesel.ai/blog/grok-4-7-pricing)
  - GLM-5.3-Flash: [eesel.ai](https://www.eesel.ai/blog/glm-5-3-flash-pricing)
  - DeepSeek V4.1 Flash: [propakistani](https://propakistani.pk/2026/09/11/deepseek-v4-1-flash-launches-with-lower-prices-and-native-vision/)

**Caveats:**
- **Sample:** 20 questions on two policies. Tables and lists are sent whole, so questions that hit large tables cost more than the mean.
- **Output varies** between runs even at temperature 0. The same list question has produced 799–1,024 output tokens; 1,024 is the cap, and hitting it truncates the answer.
- **Nothing counted for failed requests:** retries after a 429 or a timeout are not billed as completed tokens. A stream that fails midway may be billed for the tokens it produced.
