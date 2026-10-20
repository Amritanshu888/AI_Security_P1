#!/usr/bin/env python
# coding: utf-8

# # 🌐 LLM Gateway with Portkey: Zero to Production
# ### A Complete Hands-On Guide for Enterprise AI Systems
# 
# ---
# 
# > **What you'll build:** A progressively enhanced LLM setup — starting from a raw direct API call, then routing every request through a production-grade gateway with observability, retries, fallbacks, caching, and more.
# 
# | Experiment | What We Build | New Concept |
# |---|---|---|
# | 🔴 Baseline | Raw Groq call, no gateway | The problem |
# | 🟡 Exp 1 | Route through Portkey | Gateway basics |
# | 🟡 Exp 2 | + Metadata & Observability | Request tagging |
# | 🟡 Exp 3 | + Automatic Retries | Resilience |
# | 🟡 Exp 4 | + Request Timeouts | Latency control |
# | 🟢 Exp 5 | + Fallbacks | Multi-provider routing |
# | 🟢 Exp 6 | + Load Balancing | Traffic splitting |
# | 🟢 Exp 7 | + Caching | Cost & speed |
# | 🟢 Exp 8 | + Saved Config from Dashboard | Production configs |
# | 🏭 Exp 9 | LangChain Drop-in | RAG integration |
# | 🏭 Exp 10 | Full Production Gateway | Everything combined |

# ---
# # PART 1 — Theory: What Is an LLM Gateway and Why Do We Need It?
# 
# ## The Problem With Direct LLM Calls
# 
# When you call an LLM API directly:
# 
# ```python
# # What most teams start with
# response = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", ...)
# ```
# 
# This works — until it doesn't. Real production problems:
# 
# | Problem | Impact |
# |---|---|
# | Groq rate-limits you at 3 AM | App breaks for hours with no fallback |
# | You switch providers | Rewrite every API call across 10+ files |
# | Nobody knows what prompts are sent | Zero visibility, impossible to debug |
# | Same question asked 1000 times | You pay for 1000 LLM calls |
# | Groq goes down | Full outage — no automatic recovery |
# | Latency spikes | No timeout, users wait forever |
# 
# An LLM Gateway solves all of these with **zero changes to your business logic**.
# 
# ## What Is a Gateway?
# 
# A gateway is a **proxy layer** between your application and any LLM API:
# 
# ```
# Your App  →  [LLM Gateway]  →  Groq / NVIDIA / OpenAI / Anthropic
# ```
# 
# The gateway intercepts every request and can:
# 1. **Route** to the right provider
# 2. **Retry** automatically on transient failures
# 3. **Fallback** to a different provider if the primary fails
# 4. **Balance** load across multiple models by weight
# 5. **Cache** responses so identical requests never hit the LLM twice
# 6. **Log** everything with full request/response detail
# 7. **Tag** requests with metadata (user, session, feature) for analytics
# 8. **Enforce** timeouts and kill slow requests
# 
# ## Why Portkey?
# 
# | Feature | Portkey | LiteLLM | Direct SDK |
# |---|---|---|---|
# | 250+ models unified | ✅ | ✅ | ❌ one provider |
# | Automatic fallbacks | ✅ | ✅ | ❌ manual |
# | Request caching | ✅ | ✅ | ❌ manual |
# | Observability dashboard | ✅ beautiful UI | ⚠️ basic | ❌ none |
# | LangChain drop-in | ✅ | ✅ | ✅ native |
# | Config-as-code | ✅ JSON/YAML | ❌ | ❌ |
# | Open source | ✅ Apache 2.0 | ✅ MIT | N/A |
# | Overhead | ~20–40ms | ~30ms | 0ms |
# 
# Portkey handles **25M+ daily requests** with **99.99% uptime**.

# In[12]:


import os
import time
import uuid
import json
from dotenv import load_dotenv
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL

load_dotenv(dotenv_path="../.env")

# Portkey API key — authenticates your app to the Portkey gateway
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "b3dPe0oypNaOP8ZNNPxvojGWGyoA")

# Provider slug — the name you gave when adding Groq in the Portkey dashboard
# Model format: @<slug>/<model-name>
GROQ_SLUG    =  "flight-policsy"  # your Groq integration slug
GROQ_MODEL   = f"@{GROQ_SLUG}/llama-3.3-70b-versatile"

# Second Groq integration for multi-provider experiments (5, 6, 10)
# Uses a smaller/faster model as the fallback target
GROQ_SLUG_2      =  "flight-policy"           # your second Groq slug
GROQ_MODEL_SMALL = f"@{GROQ_SLUG_2}/llama-3.1-8b-instant"

# Keep GROQ_API_KEY for the LangChain experiment (Exp 9)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

def show(q, answer, ms, label=""):
    bar = chr(9472) * 62
    print(f"\n{bar}")
    print(f"Q: {q}")
    print(f"A: {answer[:260]}{'...' if len(answer) > 260 else ''}")
    note = f" | {label}" if label else ""
    print(f"⏱  {ms:.0f}ms{note}")
    print(bar)

# Main Portkey client — used for all simple experiments
portkey = Portkey(api_key=PORTKEY_API_KEY)

print("Setup complete!")
print(f"  Portkey API Key : {'OK' if PORTKEY_API_KEY else 'MISSING'}")
print(f"  Groq slug       : {GROQ_SLUG}")
print(f"  Groq model ref  : {GROQ_MODEL}")
print(f"  Groq slug 2     : {GROQ_SLUG_2}")
print(f"  Small model ref : {GROQ_MODEL_SMALL}")
print(f"\nPortkey Gateway : {PORTKEY_GATEWAY_URL}")


# ## ⚠️ Note on Variable Names
# 
# You had:
# ```python
# PORTKEY_CONFIG_ID = "flight-policsy"
# ```
# 
# **What it actually is:** a **provider slug** (the integration name you gave when adding Groq).
# It is NOT a config ID. Config IDs are a different thing in Portkey — they start with `pc-` and refer to saved routing strategies (fallback/load-balance configs you build in the dashboard).
# 
# | Thing | Looks like | Used for |
# |---|---|---|
# | **Provider slug** | `flight-policsy` | `@flight-policsy/model-name` |
# | **Config ID** | `pc-abc123` | `Portkey(config="pc-abc123")` |
# 
# These are two separate concepts. We use both in this notebook.
# 
# ---
# ## 🔧 Setup for Multi-Provider Experiments (5, 6, 10)
# 
# These experiments use **two Groq integrations** (different model sizes, same provider):
# - **Primary**: `GROQ_SLUG` (`flight-policsy`) → `llama-3.3-70b-versatile` (large, capable)
# - **Fallback**: `GROQ_SLUG_2` (`flight-policy`) → `llama-3.1-8b-instant` (small, fast)
# 
# This is a realistic production pattern: fall back to a faster/cheaper model under rate limits.
# Both slugs are already set in `c04`. All experiments work right now.

# ---
# # BASELINE — Direct LLM Call (No Gateway)
# 
# **Goal:** See what a raw LLM call looks like — no routing, no logging, no resilience.

# In[2]:


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

raw_groq = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)

section("BASELINE — Direct Groq Call")

questions = [
    "What is Kubernetes in one sentence?",
    "What is Intel SRIOV?",
]

for q in questions:
    t0 = time.time()
    r = raw_groq.invoke([HumanMessage(content=q)])
    show(q, r.content, (time.time()-t0)*1000, label="direct Groq — no gateway")

print("\nMissing: logging, retries, fallback, caching, metadata.")


# ---
# # EXPERIMENT 1 — Route Through the Gateway
# 
# **Goal:** Same call, same answer — but now every request is logged in your Portkey dashboard.
# 
# **New concepts:**
# - `Portkey(api_key=...)` — the gateway client
# - `model="@slug/model-name"` — tells Portkey which provider to use
# - `response.choices[0].message.content` — standard OpenAI response format

# In[3]:


section("EXP 1 — Basic Gateway Call")

for q in questions:
    t0 = time.time()
    r = portkey.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": q}]
    )
    show(q, r.choices[0].message.content, (time.time()-t0)*1000,
         label="routed via Portkey gateway")

print("\n✅ Check portkey.ai → Logs to see both requests fully logged!")
print("   Token count, cost, latency — all tracked. Zero extra code.")


# ### What changed?
# 
# ```
# Before:  Your App  →  Groq directly
# After:   Your App  →  Portkey  →  Groq (via @flight-policsy integration)
# ```
# 
# - Portkey adds ~20–40ms then forwards to Groq using stored credentials
# - Full request + response logged automatically — no extra code
# - **You changed 3 lines. Your business logic is identical.**
# 
# ---
# # EXPERIMENT 2 — Metadata & Observability
# 
# **Goal:** Tag every request with user, session, and feature info so you can filter and analyse in the dashboard.
# 
# **New concept:** `portkey.with_options(metadata={...})` — attaches tags to a single request.
# The special key `_user` powers per-user analytics.

# In[4]:


section("EXP 2 — Metadata & Observability")

session = str(uuid.uuid4())[:8]

scenarios = [
    ("alice", "enterprise-rag",   "What is Kubernetes RBAC?"),
    ("bob",   "docs-chatbot",     "How does BGP path selection work?"),
    ("carol", "support-bot",      "What is SRIOV virtualization?"),
    ("alice", "enterprise-rag",   "Explain Kubernetes NetworkPolicy"),   # same user, diff Q
]

for user, feature, q in scenarios:
    t0 = time.time()
    r = portkey.with_options(
        metadata={
            "_user":       user,          # powers per-user analytics in dashboard
            "session_id":  session,
            "feature":     feature,
            "environment": "notebook"
        }
    ).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": q}]
    )
    ms = (time.time() - t0) * 1000
    print(f"\n👤 {user:8s} | 🔧 {feature:18s} | {ms:.0f}ms")
    print(f"  Q: {q}")
    print(f"  A: {r.choices[0].message.content[:120]}...")

print("\n✅ Dashboard now shows: cost per user, calls per feature, session grouping")


# ### Why metadata matters in production
# 
# With `_user` tagging you can answer:
# - *Which users generate the most cost?*
# - *Which feature uses the most tokens?*
# - *What did Alice's full session look like?*
# - *Is the RAG pipeline slower than the support bot?*
# 
# All answered from the Portkey dashboard — no extra logging code.
# 
# ---
# # EXPERIMENT 3 — Automatic Retries
# 
# **Goal:** Portkey automatically retries failed requests with exponential backoff. App code never sees transient 429/5xx errors.
# 
# **New concept:** Pass a `config` dict to `Portkey(...)` with retry settings.

# In[5]:


retry_config = {
    "retry": {
        "attempts": 3,
        "on_status_codes": [429, 500, 502, 503, 504]
    }
}

portkey_retry = Portkey(api_key=PORTKEY_API_KEY, config=retry_config)

section("EXP 3 — Automatic Retries")
print("Config: 3 retry attempts on [429, 500, 502, 503, 504]")
print("Retries fire automatically on failure — transparent to your code\n")

try:
    t0 = time.time()
    r = portkey_retry.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "What is a Kubernetes DaemonSet?"}]
    )
    ms = (time.time() - t0) * 1000
    print(f"✅ Succeeded in {ms:.0f}ms")
    print(f"   {r.choices[0].message.content[:300]}")
    print("\nRetry sequence if Groq had failed:")
    print("  Attempt 1 → 429 → wait 1s → Attempt 2 → 429 → wait 2s → Attempt 3")
    print("  Your code only sees the final success or the last failure")
except Exception as e:
    print(f"❌ All attempts failed: {e}")


# ---
# # EXPERIMENT 4 — Request Timeouts
# 
# **Goal:** Kill requests that take too long. An LLM stall blocks your FastAPI worker indefinitely without a timeout.
# 
# **New concept:** `request_timeout` in milliseconds in the config dict. Portkey returns **HTTP 408** on timeout. Pair with fallbacks for auto-recovery.

# In[6]:


# Config-level timeout (cleanest approach — works inside any routing config)
timeout_config = {"request_timeout": 10000}   # 10 seconds in ms

portkey_timeout = Portkey(api_key=PORTKEY_API_KEY, config=timeout_config)

section("EXP 4 — Request Timeouts")
print("Timeout: 10,000ms (10 seconds). Portkey returns HTTP 408 if exceeded.\n")

try:
    t0 = time.time()
    r = portkey_timeout.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "Explain Kubernetes networking in 2 sentences."}]
    )
    ms = (time.time() - t0) * 1000
    print(f"✅ Response in {ms:.0f}ms (within 10s timeout)")
    print(f"   {r.choices[0].message.content}")
except Exception as e:
    print(f"⏱  Timed out: {e}")
    print("   Portkey issued a 408. Pair with fallback to auto-switch providers on timeout.")

print("\n--- Combining timeout + retry (production pattern) ---")
combined = {
    "request_timeout": 10000,
    "retry": {"attempts": 2, "on_status_codes": [408, 429, 503]}
}
print(json.dumps(combined, indent=2))


# ---
# # EXPERIMENT 5 — Fallbacks
# 
# **Goal:** If the primary Groq model fails, automatically switch to the smaller Groq fallback. Users never see the failure.
# 
# **New concept:** `strategy.mode = "fallback"` with an ordered `targets` list. First target is primary, rest are fallbacks.
# Both targets use Groq — one large model (70b), one small model (8b).

# In[7]:


fallback_config = {
    "strategy": {"mode": "fallback"},
    "targets": [
        {"override_params": {"model": GROQ_MODEL}},     # primary
        {"override_params": {"model": GROQ_MODEL_SMALL}}    # fallback if primary fails
    ]
}

portkey_fallback = Portkey(api_key=PORTKEY_API_KEY, config=fallback_config)

section("EXP 5 — Fallback Routing")
print(f"Strategy: {GROQ_MODEL} → {GROQ_MODEL_SMALL} on failure\n")

fallback_questions = [
    "What is Intel QuickAssist Technology?",
    "Explain Kubernetes persistent volume claims.",
]

for q in fallback_questions:
    try:
        t0 = time.time()
        r = portkey_fallback.chat.completions.create(
            messages=[{"role": "user", "content": q}]
        )
        ms = (time.time() - t0) * 1000
        show(q, r.choices[0].message.content, ms, label="fallback ready")
    except Exception as e:
        print(f"❌ {e}")
        print("   → Check that both GROQ_SLUG and GROQ_SLUG_2 are set correctly in c04")

print("\nFallback chain:")
print("  Groq 2xx   → return immediately")
print("  Groq 4xx/5xx → try small Groq model (8b) automatically")
print("  Check Portkey Logs to see fallback activations")


# ### Narrow the fallback trigger
# 
# Default fires on any non-2xx. Narrow it to avoid accidental fallbacks on bad requests:
# 
# ```python
# # Only fall back on rate limits and server errors
# "strategy": {"mode": "fallback", "on_status_codes": [429, 503]}
# ```
# 
# ---
# # EXPERIMENT 6 — Load Balancing
# 
# **Goal:** Split traffic between providers by weight. Use for gradual migration, A/B testing, or cost control.
# 
# **New concept:** `strategy.mode = "loadbalance"` with `weight` per target. Portkey normalises weights to percentages.

# In[8]:


load_balance_config = {
    "strategy": {"mode": "loadbalance"},
    "targets": [
        {"override_params": {"model": GROQ_MODEL},   "weight": 0.7},   # 70%
        {"override_params": {"model": GROQ_MODEL_SMALL}, "weight": 0.3}    # 30%
    ]
}

portkey_lb = Portkey(api_key=PORTKEY_API_KEY, config=load_balance_config)

section("EXP 6 — Load Balancing (70% large / 30% small)")

lb_questions = [
    "What is a Kubernetes Ingress resource?",
    "How does OSPF differ from BGP?",
    "What is Intel FPGA acceleration?",
    "Explain Kubernetes HPA.",
    "What is a VLAN trunk?",
    "How does Kubernetes etcd work?",
]

print("Sending 6 requests. Expect ~4 on large model (70b), ~2 on small model (8b) (probabilistic).\n")

for i, q in enumerate(lb_questions, 1):
    try:
        t0 = time.time()
        r = portkey_lb.chat.completions.create(
            messages=[{"role": "user", "content": q}]
        )
        ms = (time.time() - t0) * 1000
        print(f"Req {i} [{ms:.0f}ms]: {q}")
        print(f"         {r.choices[0].message.content[:120]}...")
    except Exception as e:
        print(f"Req {i}: ERROR — {e}")

print("\n✅ Check Portkey Logs to see which provider served each request")
print("   Set weight=0 to pause a target without removing it from the config")


# ### Load balancing use cases
# 
# | Scenario | Config |
# |---|---|
# | **Gradual migration** | Start 95/5, shift to 50/50, then 0/100 over weeks |
# | **A/B testing** | 50/50 — measure quality + user satisfaction per model |
# | **Cost control** | Route more traffic to the cheaper model |
# | **Maintenance** | `weight: 0` pauses a target without removing it |
# 
# ---
# # EXPERIMENT 7 — Request Caching
# 
# **Goal:** Cache responses to identical questions. Second call is instant and costs nothing.
# 
# **New concept:** `cache.mode = "simple"` does exact-match caching. Identical request → served from cache, no LLM call.

# In[ ]:


cache_config = {"cache": {"mode": "simple"}}
# cache_config = {"cache": {"mode": "semantic"}}   enterprise tier

portkey_cached = Portkey(api_key=PORTKEY_API_KEY, config=cache_config)

section("EXP 7 — Simple Caching")

# Use a short, precise question with temperature=0 to maximise cache-key stability
q = "Define Kubernetes ConfigMap in one sentence."
call_params = dict(
    model=GROQ_MODEL,
    messages=[{"role": "user", "content": q}],
    temperature=0,
    max_tokens=120,
)

print("--- CALL 1: Cache MISS — Portkey forwards to Groq ---")
t0 = time.time()
r1 = portkey_cached.chat.completions.create(**call_params)
t1 = (time.time() - t0) * 1000
ans1 = r1.choices[0].message.content.strip()
print(f"Answer  : {ans1[:200]}")
print(f"Latency : {t1:.0f}ms | Cost: normal token price")

print("CALL 2: Same request — should be a Cache HIT")
t0 = time.time()
r2 = portkey_cached.chat.completions.create(**call_params)
t2 = (time.time() - t0) * 1000
ans2 = r2.choices[0].message.content.strip()
print(f"Answer  : {ans2[:200]}")
print(f"Latency : {t2:.0f}ms")

identical = ans1 == ans2
speedup   = t1 / t2 if t2 > 0 else 999
print()
print(f"Identical answers : {identical}  ← True = cache served exact stored response")
print(f"Speedup           : {speedup:.1f}x")
if t2 < 200:
    print("✅ Cache HIT confirmed — response served from Portkey in <200ms")
elif identical:
    print("✅ Answers match — cache likely hit but gateway latency added round-trip time")
else:
    print("⚠️  Answers differ — Portkey may have missed cache due to header variation.")
    print("   Verify in Portkey Logs: look for cache_status = HIT on call 2.")
print()
print("To force a fresh response (e.g. after updating docs):")
print("  portkey_cached.with_options(cache_force_refresh=True).chat.completions.create(...)")
print("Note: Portkey Logs → each request shows cache_status (HIT / MISS) in the detail panel.")


# ---
# # EXPERIMENT 8 — Saved Config from Dashboard
# 
# **Goal:** Store your routing strategy in the Portkey dashboard once. Reference it by config ID everywhere. Change strategy centrally without redeploying.
# 
# **New concept:** Config IDs (start with `pc-`). You build the config in the dashboard UI — fallbacks, retries, weights — save it, and get a stable `pc-xxxxx` ID.

# In[9]:


# How to get a config ID:
# 1. portkey.ai → Configs → Create Config
# 2. Add your fallback/retry/cache strategy visually
# 3. Save → copy the ID (starts with pc-xxxxx)
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_SAVED_CONFIG_ID", "pc-ssss-d8a1e4")

portkey_from_config = Portkey(
    api_key=PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_ID    # "pc-xxxxx"  ← different from provider slug!
)

section("EXP 8 — Saved Config from Dashboard")
print(f"Using config ID: {PORTKEY_CONFIG_ID}")
print("All routing rules live in the dashboard. Edit there, no redeploy needed.\n")

try:
    t0 = time.time()
    r = portkey_from_config.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "What is Kubernetes HPA?"}]
    )
    ms = (time.time() - t0) * 1000
    print(f"✅ Config applied ({ms:.0f}ms)")
    print(f"   {r.choices[0].message.content[:300]}")
    print("\n💡 To update routing: edit config in dashboard → Save. Zero redeployment.")
except Exception as e:
    print(f"❌ Config error: {e}")
    print("   → Config IDs start with pc- and come from Portkey dashboard → Configs")


# ### Provider slug vs Config ID — the difference
# 
# ```
# @flight-policsy/llama-3.3-70b-versatile
#         ↑
#    Provider slug — identifies WHICH provider to use
#    (you set this when adding the integration)
# 
# pc-abc123
#    ↑
#    Config ID — identifies a saved routing STRATEGY
#    (fallback, retry rules, load balance weights, etc.)
#    (you get this from Portkey → Configs → your config)
# ```
# 
# Both are useful. Provider slugs select the LLM. Config IDs apply complex routing rules.
# 
# ---
# # EXPERIMENT 8 — LangChain Drop-In Integration
# 
# **Goal:** Slot Portkey into the existing LangChain pipeline with **zero changes to chain logic**. This is the integration point for `app/agents/nodes/`.

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Portkey exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL
# Pass PORTKEY_API_KEY as the api_key; the @slug/model handles provider routing
portkey_llm = ChatOpenAI(
    api_key=PORTKEY_API_KEY,          # Portkey API key (not Groq key)
    base_url=PORTKEY_GATEWAY_URL,     # Portkey gateway endpoint
    model=GROQ_MODEL,                 # "@rag/llama-3.3-70b-versatile"
    temperature=0,
    default_headers=createHeaders(    # adds x-portkey-* headers
        api_key=PORTKEY_API_KEY,
        metadata={
            "_user":       "rag-pipeline",
            "environment": "notebook",
            "feature":     "langchain-integration"
        }
    )
)

section("EXP 9 — LangChain Drop-in")

# Test 1: Direct invoke (like app/agents/nodes/planner.py)
print("--- Test 1: Direct invoke (like planner node) ---")
t0 = time.time()
r = portkey_llm.invoke([
    SystemMessage(content="You are an Enterprise IT Assistant."),
    HumanMessage(content="What is the difference between a Deployment and a StatefulSet?")
])
ms = (time.time() - t0) * 1000
print(f"✅ {ms:.0f}ms")
print(r.content[:250])

# Test 2: LCEL chain (like app/agents/nodes/responder.py)
print("\n--- Test 2: LCEL chain (like responder node) ---")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Enterprise IT expert. Be concise."),
    ("human",  "{question}")
])
chain = prompt | portkey_llm | StrOutputParser()

t0 = time.time()
answer = chain.invoke({"question": "Explain Kubernetes pod affinity rules."})
ms = (time.time() - t0) * 1000
print(f"✅ {ms:.0f}ms")
print(answer[:250])

print("\n→ This is a drop-in for ChatGroq in app/agents/nodes/*.py")
print("→ Every RAG pipeline call is now logged in Portkey with metadata")


# ### Migration path for the RAG system
# 
# In `app/agents/nodes/planner.py`:
# 
# ```python
# # Before
# from langchain_groq import ChatGroq
# llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
# 
# # After
# from langchain_openai import ChatOpenAI
# from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
# 
# llm = ChatOpenAI(
#     api_key=PORTKEY_API_KEY,
#     base_url=PORTKEY_GATEWAY_URL,
#     model="@flight-policsy/llama-3.3-70b-versatile",
#     default_headers=createHeaders(
#         api_key=PORTKEY_API_KEY,
#         metadata={"feature": "planner"}
#     )
# )
# ```
# 
# **Zero changes** to node logic, chain structure, or LangGraph wiring.
# 
# ---
# # EXPERIMENT 10 — Full Production Gateway
# 
# **Goal:** Combine everything — fallbacks + retries + timeout + caching + metadata. The config you'd use in `app/main.py`.

# In[11]:


PRODUCTION_CONFIG = {
    "strategy":       {"mode": "fallback"},
    "request_timeout": 30000,              # 30s hard cap
    "retry": {
        "attempts":        2,
        "on_status_codes": [429, 500, 503]
    },
    "cache": {"mode": "simple"},           # free repeated queries
    "targets": [
        {"override_params": {"model": GROQ_MODEL},   "weight": 0.8},
        {"override_params": {"model": GROQ_MODEL_SMALL}, "weight": 0.2}
    ]
}

production_gateway = Portkey(api_key=PORTKEY_API_KEY, config=PRODUCTION_CONFIG)

section("EXP 10 — Full Production Gateway")
print("⚙️  Active:")
print("   Fallback  : large Groq 70b (80%) → small Groq 8b (20%) on failure")
print("   Retry     : 2 attempts on 429/500/503")
print("   Timeout   : 30 seconds")
print("   Cache     : simple exact match")

test_suite = [
    ("alice", "enterprise-rag",   "What is Kubernetes RBAC?"),
    ("bob",   "support-chat",     "How does Intel SRIOV work?"),
    ("carol", "docs-lookup",      "Explain BGP route reflectors."),
    ("alice", "enterprise-rag",   "What is Kubernetes RBAC?"),  # cache hit!
    ("dave",  "support-chat",     "What is a Kubernetes operator pattern?"),
]

for user, feature, q in test_suite:
    try:
        t0 = time.time()
        r = production_gateway.with_options(
            metadata={
                "_user":       user,
                "feature":     feature,
                "session_id":  str(uuid.uuid4())[:8],
                "environment": "production"
            }
        ).chat.completions.create(
            messages=[{"role": "user", "content": q}]
        )
        ms = (time.time() - t0) * 1000
        cache_hint = " — CACHE HIT!" if ms < 100 else ""
        print(f"\n👤 {user:8s} | {feature:18s} | {ms:.0f}ms{cache_hint}")
        print(f"   Q: {q}")
        print(f"   A: {r.choices[0].message.content[:150]}...")
    except Exception as e:
        print(f"   ERROR: {e}")

print("\n✅ Full observability, resilience, and cost control on every call.")


# ---
# # Connecting to the RAG System
# 
# ```python
# # app/main.py
# from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
# from langchain_openai import ChatOpenAI
# 
# PRODUCTION_CONFIG = {
#     "strategy":        {"mode": "fallback"},
#     "request_timeout": 30000,
#     "retry":           {"attempts": 2, "on_status_codes": [429, 500, 503]},
#     "cache":           {"mode": "simple"},
#     "targets": [
#         {"override_params": {"model": "@flight-policsy/llama-3.3-70b-versatile"}},
#         {"override_params": {"model": "@flight-policy/llama-3.1-8b-instant"}}
#     ]
# }
# 
# # Drop-in LLM for all LangGraph nodes (planner, retriever, responder)
# gateway_llm = ChatOpenAI(
#     api_key=PORTKEY_API_KEY,
#     base_url=PORTKEY_GATEWAY_URL,
#     model="@flight-policsy/llama-3.3-70b-versatile",
#     default_headers=createHeaders(
#         api_key=PORTKEY_API_KEY,
#         config=PRODUCTION_CONFIG
#     )
# )
# 
# @app.post("/query")
# def query(request: QueryRequest):
#     # Per-request metadata tags the session user in Portkey logs
#     tagged_headers = createHeaders(
#         api_key=PORTKEY_API_KEY,
#         config=PRODUCTION_CONFIG,
#         metadata={"_user": request.thread_id, "feature": "rag-query"}
#     )
#     ...
# ```

# ---
# # SUMMARY
# 
# ## What Is an LLM Gateway?
# 
# A proxy between your app and any LLM API that adds **resilience, observability, and cost control** without touching business logic.
# 
# Think of it as a **circuit breaker + load balancer + cache + logger** — all for LLM calls.
# 
# ## Why Use One in Production?
# 
# | Without Gateway | With Gateway |
# |---|---|
# | Provider outage = full downtime | Auto-fallback to backup provider |
# | Rate limits crash your app | Retries with exponential backoff |
# | No visibility into prompts/costs | Full log + cost tracking per request |
# | Same query paid for 1000 times | Cache hit = free + instant |
# | Changing providers = rewrite code | One config change, no redeploy |
# | No per-user analytics | `_user` metadata → dashboard analytics |
# 
# ## Framework Comparison
# 
# | Framework | Approach | Dashboard | Best For |
# |---|---|---|---|
# | **Portkey** | Managed proxy + Python SDK | ✅ Beautiful | Enterprise observability + configs |
# | **LiteLLM** | Python library | ⚠️ Basic | Pure code, no dashboard needed |
# | **Azure AI Gateway** | Azure managed | ✅ Azure Portal | Azure-native workloads |
# | **AWS Bedrock** | AWS managed | ✅ CloudWatch | AWS-native workloads |
# | **Traefik AI** | HTTP proxy | ✅ | Kubernetes infrastructure teams |
# 
# ## Full Terminology Glossary
# 
# | Term | Plain English Meaning |
# |---|---|
# | **LLM Gateway** | Proxy layer between your app and any LLM provider |
# | **Portkey** | Open-source managed LLM gateway with Python SDK + dashboard |
# | **Provider slug** | Short name you give an LLM integration, e.g. `flight-policsy` |
# | **`@slug/model`** | Model reference format: `@flight-policsy/llama-3.3-70b-versatile` |
# | **Config** | JSON routing strategy: fallback order, retry rules, cache, timeout |
# | **Config ID** | Saved config reference from dashboard. Always starts with `pc-`. |
# | **`PORTKEY_GATEWAY_URL`** | `https://api.portkey.ai/v1` — the proxy endpoint |
# | **`createHeaders()`** | Builds `x-portkey-*` headers for OpenAI-compatible clients |
# | **`Portkey(config=...)`** | Python client with inline or saved config applied |
# | **`with_options()`** | Override settings for a single request (e.g. per-request metadata) |
# | **Fallback** | Auto-switch to backup provider on non-2xx error |
# | **Load Balancing** | Probabilistic traffic split by `weight` |
# | **Retry** | Re-send N times with exponential backoff |
# | **Timeout** | Kill request after N milliseconds. Portkey returns HTTP 408. |
# | **Simple Cache** | Exact-match store. Identical query = instant free hit. |
# | **Semantic Cache** | Similarity-match cache. Similar queries also hit the cache. Enterprise only. |
# | **Metadata** | Key-value tags on a request. `_user` powers per-user analytics. |
# | **408** | Portkey's HTTP status when `request_timeout` is exceeded. |
# | **429** | Rate limit from LLM provider. Triggers retry/fallback. |
# 
# ---
# 
# ## 30-Second Cheat Sheet
# 
# ```python
# from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
# 
# # Basic call — just route through the gateway
# portkey = Portkey(api_key=PORTKEY_API_KEY)
# r = portkey.chat.completions.create(
#     model="@flight-policsy/llama-3.3-70b-versatile",
#     messages=[{"role": "user", "content": "Hello"}]
# )
# 
# # With per-request metadata
# r = portkey.with_options(
#     metadata={"_user": "alice", "feature": "rag"}
# ).chat.completions.create(model="@flight-policsy/...", messages=[...])
# 
# # With inline config (retry + cache + fallback)
# portkey = Portkey(
#     api_key=PORTKEY_API_KEY,
#     config={
#         "strategy": {"mode": "fallback"},
#         "retry":    {"attempts": 3},
#         "cache":    {"mode": "simple"},
#         "targets":  [
#             {"override_params": {"model": "@groq-slug/llama-3.3-70b-versatile"}},
#             {"override_params": {"model": "@flight-policy/llama-3.1-8b-instant"}}
#         ]
#     }
# )
# 
# # With saved config from dashboard
# portkey = Portkey(api_key=PORTKEY_API_KEY, config="pc-your-config-id")
# 
# # LangChain drop-in
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(
#     api_key=PORTKEY_API_KEY,
#     base_url=PORTKEY_GATEWAY_URL,
#     model="@flight-policsy/llama-3.3-70b-versatile",
#     default_headers=createHeaders(api_key=PORTKEY_API_KEY)
# )
# ```
# 
# ## See Also
# - `DOCS/15_GUARDRAILS.md` — NeMo Guardrails (safety layer, pairs with gateway)
# - `notebooks/01_guardrails.ipynb` — live guardrails experiments
# - `app/main.py` — FastAPI entry point where the gateway integrates
# - `app/agents/nodes/` — swap `ChatGroq` for `portkey_llm` here
