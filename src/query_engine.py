"""
Query Engine Module
"""

from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from config import OPENAI_CONFIG, VECTOR_STORE_CONFIG
import re


QUERY_REFINEMENT_PROMPT = """You are a financial analyst assistant.
Your task is to analyze user queries about automotive company financial data (BMW, Ford, Tesla).

Available data: BMW, Ford, Tesla for years 2021, 2022, 2023.

Check if the query has enough information to answer:
- Company name (BMW, Ford, or Tesla) - can be single or multiple companies
- Year/time period (2021, 2022, 2023) - note that "past three years" from 2025 should reference available data (2021-2023)
- Metric (revenue, profit, etc.)

IMPORTANT:
- Only ask for clarification if CRITICAL information is missing
- If the query mentions multiple companies, keep ALL companies in the refined query
- If the query asks about "all companies", explicitly list BMW, Ford, and Tesla

If query is MISSING CRITICAL INFO, respond with:
CLARIFICATION_NEEDED: [Ask ONE specific question]

If query has enough information, refine it:
REFINED_QUERY: [Refined query with specific terminology]

Examples:

User Query: "What was the revenue?"
CLARIFICATION_NEEDED: Which company and year? (e.g., "BMW 2023", "Ford 2022", or "all companies")

User Query: "revenue in 2023"
CLARIFICATION_NEEDED: Which company - BMW, Ford, Tesla, or all companies?

User Query: "What was the revenue in 2023? Ford"
REFINED_QUERY: What was Ford's total revenue in 2023?

User Query: "Provide revenue summary for Tesla, BMW, and Ford over the past three years"
REFINED_QUERY: What was the total revenue for Tesla, BMW, and Ford for each year from 2021 to 2023? Provide a comparison across all three companies.

User Query: "compare revenue across all companies in 2022"
REFINED_QUERY: What was the total revenue for BMW, Ford, and Tesla in 2022? Provide a comparison.

User Query: "Total revenue in 2022 Ford"
REFINED_QUERY: What was Ford's total revenue in 2022?

Now analyze this query:
User Query: {query}

Response:"""


FINANCIAL_QA_PROMPT = """You are a financial analyst specializing in the automotive sector.
Use the provided context from annual reports to answer questions accurately.

Context from annual reports:
---------------------
{context_str}
---------------------

Question: {query_str}

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. SEARCH THE ENTIRE CONTEXT: The context above may contain information from multiple companies. Look through ALL of it carefully before answering.

2. MULTI-COMPANY QUERIES: When asked about multiple companies (BMW, Ford, Tesla):
   - Provide data for ALL mentioned companies
   - Search through the entire context for each company - don't stop after finding the first one
   - If you find data for only SOME companies, explicitly state which are missing
   - Example: "Based on the context: Tesla revenue was $X, BMW revenue was €Y. However, Ford data is not available in the provided context."

3. FORMATTING:
   - Provide specific numbers with units (e.g., "€111.2 billion", "$5.6 billion")
   - Always cite the year and company name exactly as it appears in the context
   - For comparisons, show data side-by-side in a clear format or table
   - For tables in context, preserve the structure

4. TIME PERIODS:
   - Available data covers years 2021, 2022, 2023
   - If query mentions "past three years" from 2025, note that data shown is for 2021-2023

5. MISSING DATA:
   - Only say data is missing if you've searched the ENTIRE context above
   - If information is completely missing for a company, clearly state: "Data for [Company] is not available in the provided context"

Be comprehensive and accurate. Double-check you've reviewed all relevant parts of the context.

Answer:"""


def extract_companies_from_query(query: str) -> list[str]:
    """
    Extract company names mentioned in the query

    Args:
        query: User query string

    Returns:
        List of company names found (BMW, Ford, Tesla)
    """
    companies = []
    query_lower = query.lower()

    if "bmw" in query_lower:
        companies.append("BMW")
    if "ford" in query_lower:
        companies.append("Ford")
    if "tesla" in query_lower:
        companies.append("Tesla")

    # Check for "all companies" or similar phrases
    if any(phrase in query_lower for phrase in ["all companies", "all three", "each company", "companies"]):
        return ["BMW", "Ford", "Tesla"]

    return companies


def create_multi_company_retriever(index, query: str, similarity_top_k: int):
    """
    Create a retriever that ensures coverage of all companies mentioned in query

    Args:
        index: VectorStoreIndex
        query: User query
        similarity_top_k: Number of documents to retrieve per company

    Returns:
        List of retrieved nodes from all mentioned companies
    """
    companies = extract_companies_from_query(query)

    if len(companies) <= 1:
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )
        return retriever.retrieve(query)

    all_nodes = []
    nodes_per_company = max(similarity_top_k // len(companies), 5)

    for company in companies:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="company",
                    value=company,
                    operator=FilterOperator.EQ
                )
            ]
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=nodes_per_company,
            filters=filters
        )

        company_nodes = retriever.retrieve(query)
        all_nodes.extend(company_nodes)

    return all_nodes


def refine_query(query: str, llm: OpenAI | None = None) -> str:
    """
    Refine user query for better retrieval using LLM

    Args:
        query: Original user query
        llm: OpenAI LLM instance (optional)

    Returns:
        Refined query string
    """
    if llm is None:
        llm = OpenAI(
            model=OPENAI_CONFIG["model"],
            temperature=0
        )

    prompt = QUERY_REFINEMENT_PROMPT.format(query=query)
    refined = llm.complete(prompt)

    return str(refined).strip()


def create_query_engine(index, use_query_refinement: bool = True):
    """
    Create query engine with optional query refinement

    Args:
        index: VectorStoreIndex from rag_pipeline
        use_query_refinement: Whether to use query refinement

    Returns:
        Configured query engine
    """
    llm = OpenAI(
        model=OPENAI_CONFIG["model"],
        temperature=OPENAI_CONFIG["temperature"]
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=VECTOR_STORE_CONFIG["similarity_top_k"]
    )
    qa_prompt = PromptTemplate(FINANCIAL_QA_PROMPT)

    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=qa_prompt,
        similarity_top_k=VECTOR_STORE_CONFIG["similarity_top_k"],
        verbose=True
    )
    if use_query_refinement:
        return QueryEngineWithRefinement(query_engine, llm)
    else:
        return query_engine


class QueryEngineWithRefinement:
    """
    Query engine wrapper that refines queries before retrieval
    """

    def __init__(self, query_engine, llm: OpenAI):
        self.query_engine = query_engine
        self.llm = llm

    def query(self, query_str: str):
        """
        Query with automatic refinement

        Args:
            query_str: Original user query

        Returns:
            Query response with refined query info
        """
        print(f"\nOriginal Query: {query_str}")
        refined_query = refine_query(query_str, self.llm)
        print(f"Refined Query: {refined_query}")
        print("-" * 70)

        response = self.query_engine.query(refined_query)

        response.metadata = response.metadata or {}
        response.metadata["original_query"] = query_str
        response.metadata["refined_query"] = refined_query

        return response

    def __getattr__(self, name):
        """Delegate other methods to wrapped query engine"""
        return getattr(self.query_engine, name)


def create_chat_engine(index, use_query_refinement: bool = True, chat_mode: str = "condense_question"):
    """
    Create chat engine with conversation memory and multi-company retrieval support

    Args:
        index: VectorStoreIndex from rag_pipeline
        use_query_refinement: Whether to use query refinement
        chat_mode: Chat mode - "condense_question" (default), "context", or "simple"

    Returns:
        Configured chat engine
    """
    llm = OpenAI(
        model=OPENAI_CONFIG["model"],
        temperature=OPENAI_CONFIG["temperature"]
    )

    qa_prompt = PromptTemplate(FINANCIAL_QA_PROMPT)

    chat_engine = index.as_chat_engine(
        llm=llm,
        chat_mode=chat_mode,
        text_qa_template=qa_prompt,
        similarity_top_k=VECTOR_STORE_CONFIG["similarity_top_k"],
        verbose=True
    )

    if use_query_refinement:
        return ChatEngineWithRefinement(chat_engine, llm, index)
    else:
        return chat_engine


class ChatEngineWithRefinement:
    """
    Chat engine wrapper that refines queries before retrieval
    Supports clarification questions for ambiguous queries
    Supports multi-company retrieval for better coverage
    """

    def __init__(self, chat_engine, llm: OpenAI, index=None):
        self.chat_engine = chat_engine
        self.llm = llm
        self.index = index

    def chat(self, message: str):
        """
        Chat with automatic query refinement and clarification

        Args:
            message: User message

        Returns:
            Chat response or tuple (True, clarification_question) if clarification needed
        """
        print(f"\nUser: {message}")

        refinement_result = refine_query(message, self.llm)

        if "CLARIFICATION_NEEDED:" in refinement_result:
            clarification = refinement_result.split("CLARIFICATION_NEEDED:")[1].strip()
            print(f"Clarification needed")
            print("-" * 70)
            return ("CLARIFICATION", clarification)

        if "REFINED_QUERY:" in refinement_result:
            refined_message = refinement_result.split("REFINED_QUERY:")[1].strip()
        else:
            refined_message = refinement_result

        print(f"Refined: {refined_message}")
        print("-" * 70)

        response = self.chat_engine.chat(refined_message)

        return response

    def reset(self):
        """Reset chat history"""
        self.chat_engine.reset()

    def __getattr__(self, name):
        """Delegate other methods to wrapped chat engine"""
        return getattr(self.chat_engine, name)
