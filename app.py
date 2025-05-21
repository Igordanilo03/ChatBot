import os
import tempfile
import streamlit as st
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

persist_directory = './faiss_index'

def process_file(file, file_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    if file_type == 'pdf':
        loader = PyPDFLoader(temp_file_path)
    else:  # markdown
        loader = UnstructuredMarkdownLoader(temp_file_path)
    
    docs = loader.load()
    os.remove(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        separators=['\n\n', '\n', ' ', ''],
        keep_separator=False,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    if os.path.exists(persist_directory):
        try:
            vector_store = FAISS.load_local(
                folder_path=persist_directory,
                embeddings=OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            st.error(f"Erro ao carregar o vector store: {e}")
            return None
    return None

def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings()
        )
    vector_store.save_local(persist_directory)
    return vector_store

def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    system_prompt = '''
Você é um assistente de conhecimento geral, altamente versátil, com expertise em diversas áreas, incluindo ciência, tecnologia, negócios, artes, saúde, entre outros. Sua missão é fornecer respostas precisas, úteis e bem estruturadas para uma ampla gama de perguntas, utilizando um tom profissional, amigável e acessível. Responda em formato de markdown, utilizando listas, tabelas, cabeçalhos ou outros elementos visuais para organizar as informações de forma clara e interativa.

### Contexto
- **Conhecimento Geral**: Utilize seu conhecimento amplo e atualizado para responder perguntas em qualquer área, fornecendo informações detalhadas e confiáveis. Se a pergunta for ambígua, peça esclarecimentos para garantir precisão.
- **Gerenciamento de Projetos**: Quando o usuário perguntar "qual o próximo passo do projeto?" ou algo semelhante, utilize as informações fornecidas no contexto do projeto (detalhes do projeto) para:
  - **Resumir o progresso**: Explique onde o projeto está atualmente, destacando etapas concluídas e o status atual.
  - **Sugerir próximos passos**: Baseado nas informações do projeto, sugira ações específicas, priorizando tarefas lógicas e viáveis.
  - **Fornecer lembretes**: Identifique prazos, dependências ou riscos potenciais e alerte o usuário sobre eles.
  - **Oferecer sugestões**: Proponha ideias para otimizar o projeto, como ferramentas, estratégias ou abordagens alternativas.
- **Contexto do Projeto**: O planejamento do projeto "Agente de Busca e Transcrição" está armazenado em um banco vetorial baseado no arquivo `planejamento_agente_busca_transcricao.md`. Consulte o contexto fornecido para responder perguntas relacionadas ao projeto.
- **Falta de Informações**: Se o contexto do banco vetorial for insuficiente, informe educadamente: "Não tenho informações suficientes no contexto fornecido. Por favor, forneça mais detalhes para que eu possa ajudar adequadamente."

### Instruções
1. **Formato de Resposta**:
   - Use markdown para estruturar respostas com clareza (títulos, listas, tabelas, etc.).
   - Quando apropriado, inclua visualizações interativas, como tabelas de progresso ou listas de tarefas.
   - Mantenha as respostas concisas, mas completas, adaptando o nível de detalhe ao pedido do usuário.
2. **Gerenciamento de Projetos**:
   - Ao responder sobre projetos, organize as informações em seções claras, como:
     - **Status Atual**: Resumo do progresso.
     - **Próximos Passos**: Tarefas específicas a serem realizadas.
     - **Lembretes**: Prazos ou riscos.
     - **Sugestões**: Ideias para melhorar o projeto.
3. **Interatividade**:
   - Quando possível, sugira ferramentas ou métodos visuais (ex.: Kanban, cronogramas) que o usuário pode adotar.
   - Adapte as respostas ao tom do usuário (ex.: mais técnico ou mais simples, conforme necessário).

**Contexto do Projeto**: {context}

**Pergunta do Usuário**: {input}
'''
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )
    
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')

vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='PyShiki',
    page_icon='🤖',
)

st.header('Sou Shiki, seu assistente pessoal!')

with st.sidebar:
    st.header('Upload de arquivos')
    upload_files = st.file_uploader(
        label='Faça upload de arquivos PDF ou Markdown.',
        type=['pdf', 'md'],
        accept_multiple_files=True,
    )
    
    if upload_files:
        with st.spinner('Processando dados...'):
            all_chunks = []
            for upload_file in upload_files:
                file_type = upload_file.name.split('.')[-1].lower()
                chunks = process_file(file=upload_file, file_type=file_type)
                all_chunks.extend(chunks)
            
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )
    
    model_option = [
        'gpt-3.5-turbo',
        'gpt-4o',
        'gpt-4o-mini',
    ]
    
    selected_model = st.selectbox(
        label='Selecione seu modelo',
        options=model_option,
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('Faça uma pergunta')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))
    
    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})
    
    response = ask_question(
        model=selected_model,
        query=question,
        vector_store=vector_store,
    )
    
    st.chat_message('ai').write(response)
    st.session_state.messages.append({'role': 'ai', 'content': response})