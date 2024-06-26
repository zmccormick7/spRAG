import sys
import os

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.create_kb import create_kb_from_file
from sprag.knowledge_base import KnowledgeBase
from sprag.reranker import NoReranker, CohereReranker
from sprag.embedding import OpenAIEmbedding, CohereEmbedding

 
def test_create_kb_from_file():
    cleanup() # delete the KnowledgeBase object if it exists so we can start fresh
    
    # Get the absolute path of the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the dataset file
    file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")
    
    kb_id = "levels_of_agi"
    kb = create_kb_from_file(kb_id, file_path)

def cleanup():
    kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
    kb.delete()


if __name__ == "__main__":
    """
    # create the KnowledgeBase if it doesn't already exist
    try:
        test_create_kb_from_file()
    except ValueError as e:
        print(e)
    """

    #reranker = NoReranker(ignore_absolute_relevance=True)
    reranker = CohereReranker()
    
    # load the KnowledgeBase and query it
    kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True, reranker=reranker)
    #print (kb.chunk_db.get_all_doc_ids())

    #search_queries = ["What are the levels of AGI?"]
    #search_queries = ["Who is the president of the United States?"]
    #search_queries = ["AI"]
    #search_queries = ["What is the difference between AGI and ASI?"]
    search_queries = ["How does autonomy factor into AGI?"]
    #search_queries = ["Self-driving cars"]
    #search_queries = ["Methodology for determining levels of AGI"]
    #search_queries = ["What is Autonomy Level 3"]
    #search_queries = ["Use of existing AI benchmarks like Big-bench and HELM"]

    relevant_segments = kb.query(search_queries=search_queries, rse_params={})
    
    print ()
    for segment in relevant_segments:
        print (len(segment["text"]))
        print (segment["score"])
        print (segment["text"])
        print ("---\n")