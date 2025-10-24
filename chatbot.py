"""
Simple Interactive Chatbot for Financial Document Analysis

Ask questions about BMW, Ford, and Tesla financial performance.
Supports follow-up questions with conversation memory.
"""

import os
import sys
import time
from openai import RateLimitError, APIError
from src.document_processor import process_all_pdfs
from src.rag_pipeline import build_rag_pipeline
from src.query_engine import create_chat_engine
from src.warmup_cache import create_warmup_cache
from config import DATA_FOLDER, CACHE_DIR, CHATBOT_CONFIG, WARMUP_CACHE_CONFIG
from dotenv import load_dotenv
load_dotenv()


def check_for_clarification(query_text):
    """
    Check if the refined query needs clarification

    Args:
        query_text: The response from query refinement

    Returns:
        Tuple of (needs_clarification: bool, clarification_question: str or None)
    """
    if "CLARIFICATION_NEEDED:" in query_text:
        clarification = query_text.split("CLARIFICATION_NEEDED:")[1].strip()
        return True, clarification
    elif "REFINED_QUERY:" in query_text:
        refined = query_text.split("REFINED_QUERY:")[1].strip()
        return False, refined
    else:
        # Fallback: treat entire response as refined query
        return False, query_text


def chat_with_retry(chat_engine, user_input, max_retries=3):
    """
    Execute chat with exponential backoff retry for rate limits

    Args:
        chat_engine: Chat engine instance
        user_input: User's query
        max_retries: Maximum number of retry attempts

    Returns:
        Chat response or clarification question

    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = chat_engine.chat(user_input)
            return response

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\nRate limit reached. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("\nRate limit error: Maximum retries reached.")
                print("Please wait a moment and try again.")
                raise

        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\nAPI error occurred. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\nAPI error: Maximum retries reached.")
                raise


def main():
    """
    Main chatbot execution
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Financial Analysis Chatbot")
    parser.add_argument("--no-cache", action="store_true",
                       help="Skip using warm-up cache (generate fresh answers)")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not set in environment")
        print("Please set it via: export OPENAI_API_KEY='your-key-here'\n")
        sys.exit(1)

    print("\n" + "="*70)
    print("FINANCIAL ANALYSIS CHATBOT")
    print("="*70)
    print("\nAutomotive Sector Analysis: BMW, Ford, Tesla (2021-2023)")
    print("\nInitializing system...")

    try:
        print("  [1/3] Loading documents...")
        all_documents = process_all_pdfs(
            data_folder=DATA_FOLDER,
            use_cache=True,
            cache_dir=CACHE_DIR
        )
        print(f"        Loaded {len(all_documents)} documents")

        print("  [2/3] Building RAG pipeline...")
        index = build_rag_pipeline(all_documents, force_rebuild=False)
        print("        Vector store ready")

        print("  [3/3] Creating chat engine...")
        # Use "context" mode for warm-up - doesn't condense questions, keeps them as-is
        # This ensures general questions stay general (e.g., "all companies" not "BMW only")
        chat_engine = create_chat_engine(
            index,
            use_query_refinement=False,
            chat_mode="context"
        )
        print("        Chat engine ready")

    except Exception as e:
        print(f"\nInitialization failed: {e}")
        sys.exit(1)

    # Warm-up phase: Prime the context with document structure analysis
    print("\n" + "="*70)
    print("ANALYZING DOCUMENT STRUCTURE")
    print("="*70)
    print("\nInitializing context with warm-up queries...\n")

    warmup_questions = [
        # Most important questions for context building
        "List the main financial metrics available across these reports.",
        "What companies are covered in these financial documents?",
        "What years are covered in these financial reports?",
        "What are BMW or BMW Group, Ford, and Tesla's electrification strategies and target timelines?",
        "What specific sustainability and CO2 emission reduction goals are outlined by BMW or BMW Group, Ford, and Tesla?"
    ]

    # Initialize warm-up cache
    warmup_cache = create_warmup_cache()
    warmup_responses = []
    
    
    # Check cache availability (unless --no-cache is specified)
    use_cache = not args.no_cache and WARMUP_CACHE_CONFIG["enabled"]
    
    if use_cache:
        cache_stats = warmup_cache.get_cache_stats()
        if cache_stats["valid"] and cache_stats["total_answers"] > 0:
            print(f"✓ Using cached warm-up answers")
            print(f"  Found {cache_stats['total_answers']} cached answers")
            if cache_stats['created_at']:
                print(f"  Cache created: {cache_stats['created_at']}")
            print()
        else:
            print("No cache found, generating fresh answers...\n")
    else:
        print("Cache disabled, generating fresh answers...\n")

    # Get cached answers for all questions (if cache is enabled)
    if use_cache:
        cached_answers = warmup_cache.get_cached_answers(warmup_questions)
        cached_count = len([a for a in cached_answers.values() if a])
        total_questions = len(warmup_questions)
        
        # Use cache if we have at least 80% of the questions cached
        cache_threshold = 0.8
        if cached_count >= total_questions * cache_threshold:
            print(f"✓ Using cached warm-up answers ({cached_count}/{total_questions} questions cached)")
            print()
            
            for i, question in enumerate(warmup_questions, 1):
                cached_answer = cached_answers[question]
                
                if cached_answer is not None:
                    # Use cached answer
                    print(f"[{i}/{total_questions}] {question}")
                    print(f"     ✓ Using cached answer")
                    warmup_responses.append((question, cached_answer))
                    print()
                else:
                    # Skip missing questions if we have enough cached ones
                    print(f"[{i}/{total_questions}] {question}")
                    print(f"     ⚠ Skipping (not in cache)")
                    print()
            
            # No questions to process
            questions_to_process = []
        else:
            print(f"⚠ Insufficient cache coverage ({cached_count}/{total_questions} questions), generating fresh answers...\n")
            questions_to_process = [(i, q) for i, q in enumerate(warmup_questions, 1)]
    else:
        # Process all questions if cache is disabled
        questions_to_process = [(i, q) for i, q in enumerate(warmup_questions, 1)]

    # Process questions that weren't cached
    if questions_to_process:
        print(f"Processing {len(questions_to_process)} questions that need fresh answers...\n")

        for original_index, question in questions_to_process:
            try:
                print(f"[{original_index}/{len(warmup_questions)}] {question}")
                # Chat engine has query refinement disabled during warm-up
                response = chat_with_retry(chat_engine, question)

                # Check if it's a clarification (shouldn't happen with these questions)
                if isinstance(response, tuple) and response[0] == "CLARIFICATION":
                    print(f"     Skipping (clarification requested)")
                    continue

                answer = str(response)
                warmup_responses.append((question, answer))

                # Write answer immediately to cache (if cache is enabled)
                if use_cache:
                    warmup_cache.store_answer(question, answer)
                    print(f"     ✓ Context updated & cached\n")
                else:
                    print(f"     ✓ Context updated\n")

            except Exception as e:
                print(f"     ⚠ Warning: {e}")
                continue

    # Display summary to user
    print("\n" + "="*70)
    print("DOCUMENT CONTEXT SUMMARY")
    print("="*70)
    
    # Show cache statistics
    if use_cache:
        final_cache_stats = warmup_cache.get_cache_stats()
        print(f"\nCache Status: {final_cache_stats['total_answers']} answers cached")
        print()

    if warmup_responses:
        # Show companies covered
        if len(warmup_responses) > 0:
            print(f"\nCompanies: {warmup_responses[0][1]}")

        # Show years covered
        if len(warmup_responses) > 1:
            print(f"\nYears: {warmup_responses[1][1]}")

        # Show key metrics available
        if len(warmup_responses) > 2:
            print(f"\nKey Metrics: {warmup_responses[2][1]}")

        # Show data quality notes
        if len(warmup_responses) > 10:
            print(f"\nData Quality Notes: {warmup_responses[10][1]}\n")
    else:
        print("\nWarm-up analysis incomplete. Proceeding with queries...\n")

    print("="*70)
    print("\nContext loaded. The assistant now has background knowledge")
    print("about the available data and can answer your questions more accurately.\n")

    # Now switch to a new chat engine with condense_question mode and query refinement
    # for interactive user queries. We don't preserve warm-up history as it was just
    # for priming the vector store retrieval, not for conversation context.
    print("Setting up interactive chat mode...")
    chat_engine = create_chat_engine(
        index,
        use_query_refinement=CHATBOT_CONFIG["use_query_refinement"],
        chat_mode="condense_question"  # Condenses chat history for better follow-ups
    )
    print("Interactive mode ready.\n")

    print("="*70)
    print("INTERACTIVE MODE - Conversation Memory Enabled")
    print("="*70)
    print("\nAsk questions about BMW, Ford, and Tesla financial performance.")
    print("You can ask follow-up questions that reference previous answers.")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit chatbot")
    print("  'reset' - Clear conversation history")
    print("  'cache' - Show warm-up cache status")
    print("\nCommand Line Options:")
    print("  --no-cache - Skip using cache (generate fresh answers)\n")

    clarification_context = None  # Store context during clarification

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nExiting...\n")
                break

            if user_input.lower() == 'reset':
                chat_engine.reset()
                clarification_context = None
                print("\nConversation history cleared.\n")
                continue

            if user_input.lower() == 'cache':
                cache_stats = warmup_cache.get_cache_stats()
                print(f"\nWarm-up Cache Status:")
                print(f"  Available: {cache_stats['valid']}")
                print(f"  Total answers: {cache_stats['total_answers']}")
                if cache_stats['created_at']:
                    print(f"  Created: {cache_stats['created_at']}")
                print()
                continue


            # If we're in clarification mode, combine with previous context
            if clarification_context:
                combined_input = f"{clarification_context} {user_input}"
                print(f"Combined query: {combined_input}")
                clarification_context = None  # Clear after using
                user_input = combined_input

            response = chat_with_retry(chat_engine, user_input)

            # Check if response is a clarification request
            if isinstance(response, tuple) and response[0] == "CLARIFICATION":
                print(f"\nAssistant: {response[1]}\n")
                # Store the original query for context
                clarification_context = user_input
                continue

            # Normal response
            print(f"\nAssistant: {response}\n")

            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("Sources:")
                for i, node in enumerate(response.source_nodes[:2], 1):
                    metadata = node.node.metadata
                    print(f"  [{i}] {metadata.get('source', 'Unknown')} "
                          f"- {metadata.get('company', 'N/A')} "
                          f"({metadata.get('year', 'N/A')})")
                print()

        except KeyboardInterrupt:
            print("\n\nExiting...\n")
            break
        except RateLimitError:
            print("\nPlease try again in a few moments.\n")
        except APIError as e:
            print(f"\nOpenAI API error: {e}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
