import pandas as pd
import os
import sys

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# import the necessary modules
from sprag.knowledge_base import KnowledgeBase
from sprag.auto_query import get_search_queries
from sprag.llm import OpenAIChatAPI, AnthropicChatAPI

AUTO_QUERY_GUIDANCE = """
The knowledge base contains SEC filings for publicly traded companies, like 10-Ks, 10-Qs, and 8-Ks. Keep this in mind when generating search queries. The things you search for should be things that are likely to be found in these documents.

When deciding what to search for, first consider the pieces of information that will be needed to answer the question. Then, consider what to search for to find those pieces of information. For example, if the question asks what the change in revenue was from 2019 to 2020, you would want to search for the 2019 and 2020 revenue numbers in two separate search queries, since those are the two separate pieces of information needed. You should also think about where you are most likely to find the information you're looking for. If you're looking for assets and liabilities, you may want to search for the balance sheet, for example.

If you're asked to calculate a financial ratio that isn't likely to directly appear in the documents, you may need to search for the components of that ratio so you can calculate it yourself.
""".strip()

RESPONSE_SYSTEM_MESSAGE = """
You are a response generation system. Please generate a response to the user input based on the provided context. Your response should be as concise as possible while still fully answering the user's question.

CONTEXT
{context}
""".strip()

def get_response(question: str, context: str):
    #client = OpenAIChatAPI(model="gpt-4-turbo", temperature=0.0)
    client = AnthropicChatAPI(model="claude-3-sonnet-20240229", temperature=0.0)
    chat_messages = [{"role": "system", "content": RESPONSE_SYSTEM_MESSAGE.format(context=context)}, {"role": "user", "content": question}]
    return client.make_llm_call(chat_messages)

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the dataset file
dataset_file_path = os.path.join(script_dir, "../../tests/data/financebench_sample_150.csv")

# Read in the data
df = pd.read_csv(dataset_file_path)

# get the questions and answers from the question column - turn into lists
questions = df.question.tolist()
answers = df.answer.tolist()

# load the knowledge base
kb = KnowledgeBase("finance_bench")

# set parameters for relevant segment extraction
rse_params = {
    'max_length': 10,
    'overall_max_length': 20,
    'overall_max_length_extension': 5,
    'irrelevant_chunk_penalty': 0.18,
    'minimum_value': 0.8,
}

# open text file to write results
with open("finance_bench_results.txt", "w") as f:

    # adjust range if you only want to run a subset of the questions (there are 150 total)
    for i in range(1):
        print (f"Question {i+1}")
        question = questions[i]
        answer = answers[i]
        search_queries = get_search_queries(question, max_queries=6, auto_query_guidance=AUTO_QUERY_GUIDANCE)
        relevant_segments = kb.query(search_queries, rse_params=rse_params)

        print ()
        for segment in relevant_segments:
            print (len(segment["text"]))
            print (segment["score"])
            #print (segment["text"])
            print ("---\n")

        context = "\n\n".join([segment['text'] for segment in relevant_segments])
        response = get_response(question, context)
        print (f"\nQuestion: {question}")
        print (f"\nSearch queries: {search_queries}")
        print (f"\nModel response: {response}")
        print (f"\nGround truth answer: {answer}")
        print ("\n---\n")

        f.write(f"Question {i+1}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Search queries: {search_queries}\n")
        f.write(f"Model response: {response}\n")
        f.write(f"Ground truth answer: {answer}\n")
        f.write("\n---\n")
