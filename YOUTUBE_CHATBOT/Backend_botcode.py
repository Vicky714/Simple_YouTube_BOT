def chatbot(user_input,video_id):
    import os
    import warnings
    from dotenv import load_dotenv
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    # Suppress warnings and logs
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    # Load environment variables
    load_dotenv()

    # YouTube Transcript
    video_id = f"{video_id}"
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi', 'en'])
        transcript = " ".join(chunk['text'] for chunk in transcript_list)
        print(transcript)
    except TranscriptsDisabled:
        print('No Caption available for this video')
        return

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embedding model and vector store
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # model_name='hkunlp/instructor-xl'
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    # LLM setup using Groq
    
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="llama3-70b-8192",
        temperature=0.7,
    )

    # Prompt template
    prompt = PromptTemplate(
        template='''
        You are a helpful assistant.
        Answer ONLY from the provided transcripts context.
        If the context is insufficient, just say you don't know.
        
        {context}
        Question: {question}
        ''',
        input_variables=['context', 'question']
    )

    # Format retrieved documents
    def format_docs(retrieved_docs):
        return '\n\n'.join(docs.page_content for docs in retrieved_docs)

    # Retrieval and response chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    # Run chatbot with sample question
    response = main_chain.invoke(user_input)
    return response
