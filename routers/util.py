from fastapi import APIRouter, Query
from rev import build_retrieval_chain, parse_csv_to_json, parse_user_stories

router = APIRouter()

@router.get("/generate-requirements")
def get_requirements(namespace: str = Query(..., description="Pinecone namespace to use")):
    try:
        question = (
            "Using the provided context, generate a structured list of system requirements in the following format: "
            "Id, Description, Category, Priority, Session, Sources.\n"
            "Return only a well-formatted CSV table containing the data. Do not include any explanatory text or additional content."
        )
        chain = build_retrieval_chain(namespace)
        csv_output = chain.invoke({"question": question})
        parsed_data = parse_csv_to_json(csv_output, namespace)
        return {
            "status": "success",
            "message": f"Requirements generated for namespace '{namespace}'",
            "data": parsed_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@router.get("/generate-epics")
def get_epics(namespace: str = Query(..., description="Pinecone namespace to use")):
    try:
        question = (
            "Based on the complete project transcript or context, produce a detailed and hierarchical list of epics and sub-epics.\n"
            "Each epic must include:\n"
            "- Epic_Id: Unique hierarchical identifier (e.g., 1, 1.1, 2, 2.1)\n"
            "- Epic_Title: Clear and specific title describing the epic.\n\n"
            "Return the results as a CSV table with columns: Epic_Id, Epic_Title. Do not include any extra commentary or text."
        )
        chain = build_retrieval_chain(namespace)
        csv_output = chain.invoke({"question": question})
        parsed_data = parse_csv_to_json(csv_output)
        return {
            "status": "success",
            "message": f"Epics generated",
            "data": parsed_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@router.get("/generate-user-stories")
def get_user_stories(namespace: str = Query(..., description="Pinecone namespace to use")):
    try:
        question = (
           "Extract ONLY user stories from the transcript in this format:\n\n"
            "User Stories\n\n"
            "• As a [role], I want to [goal], so that [reason]. speaker: [Speaker's name]\n\n"
            "Rules: Go in depth and extract all the user stories from the transcript and make sure to include all the details and depth."
        )
        chain = build_retrieval_chain(namespace)
        response = chain.invoke({"question": question})
        user_stories = parse_user_stories(response)
        return {
            "status": "success",
            "message": f"User stories extracted",
            "data": user_stories
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}
