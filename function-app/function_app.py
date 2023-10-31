import azure.functions as func
import logging
from langchain.chains import RetrievalQA
from azure_search import create_azure_search
from llm import create_llm

def create_vector_store():
    return create_azure_search()

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.function_name(name="energy_copilot_test")
@app.route(route="energy_copilot_test")
def energy_copilot_test(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    db = create_vector_store()
    llm = create_llm()

    retriever=db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 50, 'score_threshold':0.8}
    )

    qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever,chain_type="stuff")

    question = req.params.get('question')
    if not question:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            question = req_body.get('question')

    if question:
        result = qa_chain({"query": question})
        answer = result['result']
        return func.HttpResponse(f"{answer}")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a question in the query string or in the request body for a personalized response.",
             status_code=200
        )

