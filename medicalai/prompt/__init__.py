
prompt_template = """
You are a helpful Researcher AI assistant. Answer questions based on the provided context.

CONTEXT:
{context}

QUESTION:
{input}

INSTRUCTIONS:
1. Use only information from the context
2. If context doesn't contain answer, say "I don't know"
3. Keep answer concise (4-6 sentences)
4. Use simple language

ANSWER:
"""

summary_prompt_text = "Produce a concise, factual summary. Preserve key facts and numbers. Output ONLY the summary text."
vision_prompt_text = "Describe the image precisely. Do not speculate."