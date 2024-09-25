"""
DevTechBytes
https://www.youtube.com/@DevTechBytes
"""

class Config:
    PAGE_TITLE = "Image Analyzer"

    OLLAMA_MODELS = ('llava:latest', 'moondream:latest', 'bakllava')

    SYSTEM_PROMPT = f"""You are a helpful chatbot that has access to the following 
                    open-source vision models {OLLAMA_MODELS}.
                    You can can answer questions about images."""
    