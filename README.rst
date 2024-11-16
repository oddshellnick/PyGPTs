PyGPTs: Seamless Integration with Hugging Face and AI APIs
==========================================================

PyGPTs simplifies interaction with various AI models, including Hugging Face Transformers and pre-trained models available through APIs like Google Gemini.  It provides a streamlined interface for managing models, pipelines, and tokenizers, handling rate limits, and accessing different model configurations.


Key Features
------------

* **Hugging Face Integration:** Easily load and utilize pre-trained models from Hugging Face's `transformers` library.  Configure models, tokenizers, and pipelines with flexible settings.
* **Gemini API Support:** Interact with Google's Gemini models through a dedicated wrapper. Manage API keys, track usage limits, and handle different model versions.
* **Rate Limiting:** Built-in rate limiting for Gemini API calls to avoid exceeding quotas and ensure continuous operation.
* **Multiple Model Management:** The `GeminiManager` allows using multiple Gemini models with different API keys, automatically switching between them based on availability and usage limits.
* **Simplified Interface:** PyGPTs provides a clean and easy-to-use API for generating text, managing chat sessions, and accessing model information.
* **Extensible Design:** Built with modularity in mind, PyGPTs can be extended to support other AI APIs and model providers.


Installation
------------

.. code-block::

    pip install PyGPTs


Modules:
--------

* **`PyGPTs.Gemini`:**  Provides classes for interacting with Google Gemini:
* **`PyGPTs.HuggingFace`:**  Provides classes for seamless integration with Hugging Face:


Usage Examples
--------------

**Gemini:**

.. code-block:: python

    from PyGPTs.Gemini import GeminiSettings, Gemini

    settings = GeminiSettings(api_key="YOUR_API_KEY")
    gemini = Gemini(settings)

    gemini.start_chat()
    response = gemini.send_message("Hello, Gemini!", chat_index=0)
    print(response.text)

**Hugging Face:**

.. code-block:: python

    from PyGPTs.HuggingFace.Transformers import HuggingFaceTransformerSettings, HuggingFaceTransformer
    from transformers import AutoModelForCausalLM

    settings = HuggingFaceTransformerSettings(
        pretrained_model_name_or_path="gpt2",
        model_class=AutoModelForCausalLM,
        task="text-generation"
    )

    transformer = HuggingFaceTransformer(settings)
    generated_text = transformer.generate_content("Once upon a time")
    print(generated_text)


This library offers a powerful and convenient way to integrate various AI models into your projects. Its flexible design and comprehensive feature set make it a valuable tool for developers working with large language models and other AI-driven applications.


Future Notes
------------

PyGPTs is an actively developing project. We are continually working on expanding its capabilities, including adding support for new AI models and APIs, improving performance, and enhancing the user experience. Contributions, feature requests, and bug reports are welcome! We encourage you to get involved and help shape the future of PyGPTs.
