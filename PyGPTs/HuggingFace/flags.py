class HuggingFacePipelineTask:
    """
    Stores string identifiers for Hugging Face pipeline tasks.

    Attributes:
        audio_classification (str): Audio classification task.
        automatic_speech_recognition (str): Automatic speech recognition task.
        depth_estimation (str): Depth estimation task.
        document_question_answering (str): Document question answering task.
        feature_extraction (str): Feature extraction task.
        fill_mask (str): Fill-mask task.
        image_classification (str): Image classification task.
        image_feature_extraction (str): Image feature extraction task.
        image_segmentation (str): Image segmentation task.
        image_to_image (str): Image-to-image task.
        image_to_text (str): Image-to-text task.
        mask_generation (str): Mask generation task.
        ner (str): Named entity recognition task.
        object_detection (str): Object detection task.
        question_answering (str): Question answering task.
        sentiment_analysis (str): Sentiment analysis task.
        summarization (str): Summarization task.
        table_question_answering (str): Table question answering task.
        text2text_generation (str): Text-to-text generation task.
        text_classification (str): Text classification task.
        text_generation (str): Text generation task.
        text_to_audio (str): Text-to-audio task.
        text_to_speech (str): Text-to-speech task.
        token_classification (str): Token classification task.
        translation (str): Translation task.
        translation_XX_to_YY (str): Translation from language XX to YY task.
        video_classification (str): Video classification task.
        visual_question_answering (str): Visual question answering task.
        vqa (str): Visual question answering task (alternative).
        zero_shot_classification (str): Zero-shot classification task.
        zero_shot_image_classification (str): Zero-shot image classification task.
        zero_shot_audio_classification (str): Zero-shot audio classification task.
        zero_shot_object_detection (str): Zero-shot object detection task.

    :Usage:
        from PyGPTs.HuggingFace.Pipelines import HuggingFacePipelineTask

        task = HuggingFacePipelineTask.text_generation
    """
    audio_classification = "audio-classification"
    automatic_speech_recognition = "automatic-speech-recognition"
    depth_estimation = "depth-estimation"
    document_question_answering = "document-question-answering"
    feature_extraction = "feature-extraction"
    fill_mask = "fill-mask"
    image_classification = "image-classification"
    image_feature_extraction = "image-feature-extraction"
    image_segmentation = "image-segmentation"
    image_to_image = "image-to-image"
    image_to_text = "image-to-text"
    mask_generation = "mask-generation"
    ner = "ner"
    object_detection = "object-detection"
    question_answering = "question-answering"
    sentiment_analysis = "sentiment-analysis"
    summarization = "summarization"
    table_question_answering = "table-question-answering"
    text2text_generation = "text2text-generation"
    text_classification = "text-classification"
    text_generation = "text-generation"
    text_to_audio = "text-to-audio"
    text_to_speech = "text-to-speech"
    token_classification = "token-classification"
    translation = "translation"
    translation_XX_to_YY = "translation_XX_to_YY"
    video_classification = "video-classification"
    visual_question_answering = "visual-question-answering"
    vqa = "vqa"
    zero_shot_classification = "zero-shot-classification"
    zero_shot_image_classification = "zero-shot-image-classification"
    zero_shot_audio_classification = "zero-shot-audio-classification"
    zero_shot_object_detection = "zero-shot-object-detection"
