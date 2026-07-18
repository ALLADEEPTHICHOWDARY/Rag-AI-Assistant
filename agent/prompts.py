SYNTHESIS_PROMPT = """You are an AI assistant answering a user's question using
information gathered from tools. Combine the findings below into a single,
direct, well-written answer. Do not mention the tools by name unless it's
relevant.

Conversation so far:
{conversation}

Relevant past exchanges (context only — may not be directly relevant):
{memory}

Original question: {query}

Findings:
{findings}

Answer:"""