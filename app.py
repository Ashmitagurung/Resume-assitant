import streamlit as st
import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import time

# Set page configuration
st.set_page_config(
    page_title="Multi-Resume Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("📄 Multi-Resume Assistant")
st.markdown("""
This app helps you analyze and manage multiple resumes. Upload PDF resumes, ask questions about them,
find candidates with specific skills, and get suggestions for resume improvements.
""")

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'resume_dir' not in st.session_state:
    st.session_state.resume_dir = tempfile.mkdtemp()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'all_roles' not in st.session_state:
    st.session_state.all_roles = []


# Function to process PDF documents with improved metadata extraction
def process_pdfs(directory_path):
    """Load and process PDFs from a directory with improved metadata extraction"""
    try:
        # Using DirectoryLoader to load all PDFs from the directory
        loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        st.success(f"Successfully loaded {len(documents)} pages from all PDFs")

        if not documents:
            st.warning("No documents were loaded. Please check that valid PDF files were uploaded.")
            return None

        # Add metadata to track which document each chunk came from
        for doc in documents:
            # Extract filename from source path
            filename = os.path.basename(doc.metadata.get('source', 'unknown'))
            doc.metadata['filename'] = filename

            # Try to extract role from content or filename (improved approach)
            content_lower = doc.page_content.lower()

            # Define role mapping with variations
            role_keywords = {
                'AI Engineer': ['ai engineer', 'artificial intelligence engineer', 'machine learning engineer',
                                'ml engineer'],
                'Geomatics Engineer': ['geomatics engineer', 'geomatics', 'geospatial engineer', 'geodetic engineer'],
                'Data Scientist': ['data scientist', 'data science', 'analytics scientist', 'data analyst'],
                'Software Engineer': ['software engineer', 'software developer', 'programmer', 'developer'],
                'UI/UX Designer': ['ui designer', 'ux designer', 'ui/ux', 'user interface', 'user experience',
                                   'product designer'],
                'Project Manager': ['project manager', 'program manager', 'product manager', 'scrum master'],
                'DevOps Engineer': ['devops', 'devops engineer', 'site reliability engineer', 'sre'],
                'Business Analyst': ['business analyst', 'business analytics', 'systems analyst'],
                'Network Engineer': ['network engineer', 'network administrator', 'network architect']
            }

            assigned_role = 'Unknown Role'

            # Check both content and filename for role keywords
            for role, keywords in role_keywords.items():
                if any(keyword in content_lower for keyword in keywords) or any(
                        keyword in filename.lower() for keyword in keywords):
                    assigned_role = role
                    break

            doc.metadata['role'] = assigned_role

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"Split documents into {len(chunks)} chunks")

        return chunks

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None


# Function to create embeddings and vector store with validation
def create_vector_store(chunks):
    """Create embeddings and vector store from document chunks with validation"""
    try:
        if not chunks:
            raise ValueError("No document chunks available to create vector store")

        # Switch to a stronger embedding model
        with st.spinner("Initializing embedding model (BAAI/bge-small-en-v1.5)..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'}
            )

        persist_directory = "./chroma_db"
        # Remove existing DB if it exists to avoid conflicts
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        with st.spinner(f"Creating vector store with {len(chunks)} chunks..."):
            start_time = time.time()

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )

            vectorstore.persist()
            creation_time = time.time() - start_time
            st.success(f"Vector store created and persisted in {creation_time:.2f} seconds")

        # Validation - check if embeddings can be retrieved
        with st.spinner("Validating vector store..."):
            test_query = "skills"
            test_results = vectorstore.similarity_search(test_query, k=2)

            if test_results:
                st.success(f"Successfully validated vector store with {len(test_results)} test results")
            else:
                st.warning("Validation found no results in test query")

        return vectorstore

    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


# Function to initialize the LLM using the API key from secrets
def initialize_llm():
    """Initialize the language model using API key from secrets"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["groq"]["api_key"]

        with st.spinner("Initializing language model..."):
            llm = ChatGroq(
                model_name="Llama-3.3-70b-Versatile",
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=1000,  # Increased token limit for more detailed responses
                api_key=api_key,
            )

            # Test the LLM to ensure it works
            test_response = llm.invoke("Give a brief response to test if you're working properly.")
            st.success("LLM connection successful!")

            return llm

    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None


# Function to set up the retrieval system
def setup_retrieval_system(llm, vectorstore):
    """Set up the RAG retrieval system"""
    try:
        with st.spinner("Setting up RAG retrieval system..."):
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}  # Increased to 8 to get more context from multiple documents
            )

            # Create a custom QA prompt template for better instructions
            qa_template = """You are a Resume Analysis Expert assistant.

            You need to answer questions about multiple resumes. Use ONLY the context provided below to answer. If you don't know the answer based on the context, say "I don't have that information in the provided resumes."

            When answering, always:
            1. Specify which resume (by role and filename) you're referring to
            2. Use direct quotes from the resumes when appropriate
            3. Organize information clearly

            Context about the resumes:
            {context}

            Question: {query}

            Answer:"""

            # Use the prompt in the chain
            prompt = PromptTemplate(template=qa_template, input_variables=["context", "query"])

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            st.success("RAG pipeline created successfully!")

            return qa_chain

    except Exception as e:
        st.error(f"Error setting up retrieval system: {str(e)}")
        return None


# Function to get all unique roles in the database
def get_all_roles(vectorstore):
    """Get all unique roles from the vector store."""
    try:
        # Query the vector store to get a sample of documents
        all_docs = vectorstore.similarity_search(
            query="all resumes",
            k=100  # Get a large sample to ensure we capture all roles
        )

        roles = set()
        for doc in all_docs:
            role = doc.metadata.get('role', 'Unknown Role')
            roles.add(role)

        return list(roles)
    except Exception as e:
        st.error(f"Error getting roles: {str(e)}")
        return []


# Utility functions for working with resumes
def extract_resume_info(docs):
    """Extract resume information from the retrieved documents."""
    resumes = {}

    for doc in docs:
        filename = doc.metadata.get('filename', 'unknown')
        role = doc.metadata.get('role', 'Unknown Role')

        if filename not in resumes:
            resumes[filename] = {
                'role': role,
                'content': [],
                'source': doc.metadata.get('source', '')
            }

        resumes[filename]['content'].append(doc.page_content)

    return resumes


def get_resume_by_role(vectorstore, role):
    """Find a resume by role."""
    # Create a metadata filter for the role
    filter_query = {"role": {"$eq": role}}

    # Search for documents with that role
    docs = vectorstore.similarity_search(
        query=f"resume for {role}",
        k=10,
        filter=filter_query
    )

    if not docs:
        # Try a more general search if exact match fails
        docs = vectorstore.similarity_search(
            query=f"resume for {role}",
            k=10
        )

    return extract_resume_info(docs)


def query_rag(qa_chain, question):
    """Basic RAG query function."""
    with st.spinner("Analyzing resumes..."):
        result = qa_chain({"query": question})
        return result


# Main app organization
def main():
    # Check if Groq API key is available in secrets
    if "groq" not in st.secrets or "api_key" not in st.secrets["groq"]:
        st.error("Groq API key not found in secrets. Please add it to .streamlit/secrets.toml")
        st.code("""
        # Example secrets.toml structure:
        [groq]
        api_key = "your_groq_api_key_here"
        """)
        return

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Display API status
        st.success("✅ Groq API key loaded from secrets")

        # File uploader
        uploaded_files = st.file_uploader("Upload Resume PDFs",
                                          type="pdf",
                                          accept_multiple_files=True,
                                          help="Upload one or more PDF resumes to analyze")

        # Process button
        process_button = st.button("Process Resumes", disabled=not uploaded_files)

        # Info section
        st.markdown("---")
        st.markdown("### Usage Tips")
        st.markdown("""
        1. Upload PDF resumes
        2. Process the resumes
        3. Ask questions or use the tools
        """)

    # Main content area - split into tabs
    tab1, tab2, tab3 = st.tabs(["Query Resumes", "Search by Role", "Resume Modification"])

    # Process files when button is clicked
    if process_button and uploaded_files:
        with st.spinner("Processing uploaded resumes..."):
            # Clear previous files
            for file_path in st.session_state.processed_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            st.session_state.processed_files = []

            # Save uploaded files to the temp directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(st.session_state.resume_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.processed_files.append(file_path)

            st.success(f"Saved {len(uploaded_files)} resumes to temporary directory")

            # Process PDFs
            chunks = process_pdfs(st.session_state.resume_dir)
            if not chunks:
                st.error("Failed to process PDFs. Please try again.")
                return

            # Create vector store
            vectorstore = create_vector_store(chunks)
            if not vectorstore:
                st.error("Failed to create vector store. Please try again.")
                return

            # Initialize LLM
            llm = initialize_llm()
            if not llm:
                st.error("Failed to initialize LLM. Please check your API key in the secrets.toml file.")
                return

            # Setup retrieval system
            qa_chain = setup_retrieval_system(llm, vectorstore)
            if not qa_chain:
                st.error("Failed to set up retrieval system. Please try again.")
                return

            # Get all roles
            all_roles = get_all_roles(vectorstore)

            # Save to session state
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = qa_chain
            st.session_state.all_roles = all_roles
            st.session_state.initialized = True

            st.success("✅ System initialized successfully!")

    # Tab 1: Free-form querying
    with tab1:
        st.header("Ask about the resumes")

        # Query input
        query = st.text_area("Enter your question:",
                             placeholder="Example: What skills do the candidates have?",
                             disabled=not st.session_state.initialized)

        query_button = st.button("Ask", disabled=not (st.session_state.initialized and query))

        # Display results for query
        if query_button and query:
            result = query_rag(st.session_state.qa_chain, query)

            st.subheader("Answer:")
            st.write(result["result"])

            st.subheader("Sources:")
            for i, doc in enumerate(result["source_documents"][:3]):  # Limit to 3 sources
                with st.expander(
                        f"Source {i + 1} from {doc.metadata.get('filename', 'unknown')} ({doc.metadata.get('role', 'Unknown Role')})"):
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    # Tab 2: Role-based search
    with tab2:
        st.header("Search by Role")

        if st.session_state.initialized and st.session_state.all_roles:
            selected_role = st.selectbox("Select a role:", st.session_state.all_roles)

            search_role_button = st.button("View Resume", key="search_role_button")

            if search_role_button and selected_role:
                with st.spinner(f"Retrieving resumes for role: {selected_role}"):
                    resumes = get_resume_by_role(st.session_state.vectorstore, selected_role)

                    if not resumes:
                        st.warning(f"No resumes found for role: {selected_role}")
                    else:
                        for filename, info in resumes.items():
                            st.subheader(f"Resume: {filename}")
                            st.write(f"**Role:** {info['role']}")

                            # Combine the content chunks and show a preview
                            full_content = " ".join(info['content'])
                            with st.expander("Resume Content Preview"):
                                st.write(full_content[:1000] + "..." if len(full_content) > 1000 else full_content)

                            # Run a few preset queries about this resume
                            with st.spinner("Analyzing resume details..."):
                                skills_result = query_rag(st.session_state.qa_chain,
                                                          f"What skills does the {selected_role} have based on {filename}?")
                                experience_result = query_rag(st.session_state.qa_chain,
                                                              f"What is the work experience of the {selected_role} in {filename}?")
                                education_result = query_rag(st.session_state.qa_chain,
                                                             f"What is the education background of the {selected_role} in {filename}?")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("##### Skills")
                                    st.write(skills_result["result"])

                                    st.markdown("##### Education")
                                    st.write(education_result["result"])

                                with col2:
                                    st.markdown("##### Experience")
                                    st.write(experience_result["result"])
        else:
            st.info("Please upload and process resumes first to view roles.")

    # Tab 3: Resume modification suggestions
    with tab3:
        st.header("Resume Modification Suggestions")

        if st.session_state.initialized and st.session_state.all_roles:
            mod_role = st.selectbox("Select resume by role:", st.session_state.all_roles, key="mod_role")

            mod_section = st.selectbox("Section to modify:",
                                       ["Skills", "Experience", "Education", "Projects", "Summary"])

            mod_action = st.selectbox("Action:", ["Add", "Update", "Delete"])

            mod_content = st.text_area("Content to add/update:",
                                       placeholder="Example: Proficient in Python and TensorFlow")

            modify_button = st.button("Generate Suggestion", disabled=not (mod_role and mod_content))

            if modify_button:
                with st.spinner("Generating modification suggestion..."):
                    # Get the current content of the section
                    current_section_query = query_rag(
                        st.session_state.qa_chain,
                        f"What is currently in the {mod_section} section of the {mod_role} resume?"
                    )

                    st.subheader("Modification Suggestion")

                    # Get resume names
                    resumes = get_resume_by_role(st.session_state.vectorstore, mod_role)
                    resume_names = list(resumes.keys())

                    # Functions to determine benefit of modification
                    def get_modification_benefit(action, section, content, role):
                        """Generate explanation of modification benefit."""
                        benefits = {
                            "add": {
                                "Skills": "showcase additional capabilities relevant to the position",
                                "Experience": "highlight relevant work history that strengthens the candidate's profile",
                                "Education": "provide additional academic credentials",
                                "Projects": "demonstrate practical application of skills",
                                "Summary": "better position the candidate for the target role"
                            },
                            "update": {
                                "Skills": "better represent the candidate's actual skill level and focus areas",
                                "Experience": "clarify responsibilities and achievements in previous roles",
                                "Education": "provide more accurate academic information",
                                "Projects": "highlight more relevant aspects of the project work",
                                "Summary": "align the profile more closely with industry expectations"
                            },
                            "delete": {
                                "Skills": "remove outdated or irrelevant skills",
                                "Experience": "eliminate positions that aren't relevant to the target role",
                                "Education": "remove unnecessary educational details",
                                "Projects": "focus on the most relevant project experience",
                                "Summary": "eliminate unnecessary information that distracts from core qualifications"
                            }
                        }

                        # Default benefit if specific section not found
                        default_benefit = "improve the overall quality and relevance of the resume"

                        # Get the benefit for the action and section, or use the default
                        action_benefits = benefits.get(action.lower(), {})
                        return action_benefits.get(section, default_benefit)

                    # Create a formatted box with the suggestion
                    st.markdown(f"""
                    ### MODIFICATION SUGGESTION FOR:
                    **Resume:** {resume_names[0] if resume_names else mod_role}
                    **Role:** {mod_role}

                    **Action:** {mod_action.upper()}
                    **Section:** {mod_section}

                    #### CURRENT CONTENT IN SECTION:
                    {current_section_query["result"]}

                    #### PROPOSED CHANGE:
                    {mod_content}

                    #### DETAILED EXPLANATION:
                    This modification would help {get_modification_benefit(mod_action, mod_section, mod_content, mod_role)}.

                    > **IMPORTANT NOTE:** This is a SUGGESTION ONLY.
                    > The Multi-Resume Assistant does not modify the actual PDF files.
                    > To implement this change, you would need to edit the resume manually using a PDF editor.
                    """)
        else:
            st.info("Please upload and process resumes first to suggest modifications.")

    # Display initialization status
    if not st.session_state.initialized:
        st.warning("Please upload PDF resumes and process them to begin.")


if __name__ == "__main__":
    main()

