import sys
import os
import time

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.create_kb import create_kb_from_file
from sprag.knowledge_base import KnowledgeBase
from sprag.rse import get_meta_document, get_relevance_values, get_best_segments

 
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

# subclass the KnowledgeBase class so we can override the query method
class KnowledgeBaseRSE(KnowledgeBase):
    def query(self, search_queries: list[str], rse_params: dict = {}, latency_profiling: bool = False) -> list[dict]:
        """
        Inputs:
        - search_queries: list of search queries
        - rse_params: dictionary containing the following parameters:
            - max_length: maximum length of a segment, measured in number of chunks
            - overall_max_length: maximum length of all segments combined, measured in number of chunks
            - minimum_value: minimum value of a segment, measured in relevance value
            - irrelevant_chunk_penalty: float between 0 and 1
            - overall_max_length_extension: the maximum length of all segments combined will be increased by this amount for each additional query beyond the first
            - decay_rate
            - top_k_for_document_selection: the number of documents to consider

        Returns relevant_segment_info, a list of segment_info dictionaries, ordered by relevance, that each contain:
        - doc_id: the document ID of the document that the segment is from
        - chunk_start: the start index of the segment in the document
        - chunk_end: the (non-inclusive) end index of the segment in the document
        - text: the full text of the segment
        """

        default_rse_params = {
            'max_length': 12,
            'overall_max_length': 30,
            'minimum_value': 0.7,
            'irrelevant_chunk_penalty': 0.18,
            'overall_max_length_extension': 6,
            'decay_rate': 20,
            'top_k_for_document_selection': 7
        }

        # set the RSE parameters
        max_length = rse_params.get('max_length', default_rse_params['max_length'])
        overall_max_length = rse_params.get('overall_max_length', default_rse_params['overall_max_length'])
        minimum_value = rse_params.get('minimum_value', default_rse_params['minimum_value'])
        irrelevant_chunk_penalty = rse_params.get('irrelevant_chunk_penalty', default_rse_params['irrelevant_chunk_penalty'])
        overall_max_length_extension = rse_params.get('overall_max_length_extension', default_rse_params['overall_max_length_extension'])
        decay_rate = rse_params.get('decay_rate', default_rse_params['decay_rate'])
        top_k_for_document_selection = rse_params.get('top_k_for_document_selection', default_rse_params['top_k_for_document_selection'])

        overall_max_length += (len(search_queries) - 1) * overall_max_length_extension # increase the overall max length for each additional query

        start_time = time.time()
        all_ranked_results = self.get_all_ranked_results(search_queries=search_queries)
        if latency_profiling:
            print(f"get_all_ranked_results took {time.time() - start_time} seconds to run for {len(search_queries)} queries")

        document_splits, document_start_points, unique_document_ids = get_meta_document(all_ranked_results=all_ranked_results, top_k_for_document_selection=top_k_for_document_selection)

        # verify that we have a valid meta-document - otherwise return an empty list of segments
        if len(document_splits) == 0:
            return []
        
        # get the length of the meta-document so we don't have to pass in the whole list of splits
        meta_document_length = document_splits[-1]

        # get the relevance values for each chunk in the meta-document and use those to find the best segments
        all_relevance_values = get_relevance_values(all_ranked_results=all_ranked_results, meta_document_length=meta_document_length, document_start_points=document_start_points, unique_document_ids=unique_document_ids, irrelevant_chunk_penalty=irrelevant_chunk_penalty, decay_rate=decay_rate)
        best_segments, scores = get_best_segments(all_relevance_values=all_relevance_values, document_splits=document_splits, max_length=max_length, overall_max_length=overall_max_length, minimum_value=minimum_value)
        
        # convert the best segments into a list of dictionaries that contain the document id and the start and end of the chunk
        relevant_segment_info = []
        for segment_index, (start, end) in enumerate(best_segments):
            # find the document that this segment starts in
            for i, split in enumerate(document_splits):
                if start < split: # splits represent the end of each document
                    doc_start = document_splits[i-1] if i > 0 else 0
                    relevant_segment_info.append({"doc_id": unique_document_ids[i], "chunk_start": start - doc_start, "chunk_end": end - doc_start}) # NOTE: end index is non-inclusive
                    break

            score = scores[segment_index]
            relevant_segment_info[-1]["score"] = score
        
        # retrieve the actual text for the segments from the database
        for segment_info in relevant_segment_info:
            segment_info["text"] = (self.get_segment_text_from_database(segment_info["doc_id"], segment_info["chunk_start"], segment_info["chunk_end"])) # NOTE: this is where the chunk header is added to the segment text

        return relevant_segment_info


if __name__ == "__main__":
    """
    # create the KnowledgeBase if it doesn't already exist
    try:
        test_create_kb_from_file()
    except ValueError as e:
        print(e)
    """
    
    # load the KnowledgeBase and query it
    kb = KnowledgeBaseRSE(kb_id="levels_of_agi", exists_ok=True)
    #print (kb.chunk_db.get_all_doc_ids())

    #search_queries = ["What are the levels of AGI?"]
    #search_queries = ["Who is the president of the United States?"]
    #search_queries = ["AI"]
    #search_queries = ["What is the difference between AGI and ASI?"]
    #search_queries = ["How does autonomy factor into AGI?"]
    #search_queries = ["Self-driving cars"]
    #search_queries = ["Methodology for determining levels of AGI"]
    #search_queries = ["What is Autonomy Level 3"]
    search_queries = ["Use of existing AI benchmarks like Big-bench and HELM"]

    relevant_segments = kb.query(search_queries=search_queries, rse_params={})
    
    print ()
    for segment in relevant_segments:
        print (len(segment["text"]))
        print (segment["score"])
        print (segment["text"])
        print ("---\n")