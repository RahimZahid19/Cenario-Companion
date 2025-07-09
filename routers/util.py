from fastapi import APIRouter, Query, HTTPException
from rev import build_retrieval_chain, parse_csv_to_json, index, fetch_requirement_descriptions,generate_and_group_epics,parse_csv_to_grouped_json
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import io
import csv
router = APIRouter()

@router.get("/generate-requirements")
def get_requirements(
    project_id: str = Query(..., description="Project ID to use"),
    session_id: str = Query(None, description="Session ID to filter by (optional)")
):
    try:
        question = (
            "Using the provided context, generate a structured list of system requirements in the following format: "
            "Id, Description, Category, Priority, Session, Sources.\n"
            "Return only a well-formatted CSV table containing the data. Do not include any explanatory text or additional content."
        )
        chain = build_retrieval_chain(project_id, session_id)
        csv_output = chain.invoke({"question": question})
        parsed_data = parse_csv_to_json(csv_output, project_id, session_id)
        return {
            "status": "success",
            "message": "Requirements generated",
            "data": parsed_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


from fastapi import HTTPException

@router.delete("/delete-requirement")
def delete_requirement(
    project_id: str = Query(..., description="Project namespace in Pinecone"),
    session_id: str = Query(..., description="Session ID of the requirement"),
    requirement_id: str = Query(..., description="Requirement ID (e.g. REQ-001)")
):
    try:
        print(f"Deleting requirement: {requirement_id} from project: {project_id} for session: {session_id}")
        
        response = index.fetch(ids=[requirement_id], namespace=project_id)
        print("Pinecone fetch response:", response)

        vectors = getattr(response, "vectors", {})
        if requirement_id not in vectors:
            raise HTTPException(status_code=404, detail="Requirement ID not found in Pinecone.")

        metadata = vectors[requirement_id].metadata or {}
        print("Fetched metadata:", metadata)

        stored_session = metadata.get("session")
        if stored_session != session_id:
            raise HTTPException(
                status_code=400,
                detail=f"Session ID mismatch: expected {stored_session}, got {session_id}"
            )

        # Perform deletion
        index.delete(ids=[requirement_id], namespace=project_id)
        print(f"Deleted {requirement_id} from Pinecone.")

        return {
            "status": "success",
            "message": f"Requirement {requirement_id} deleted from project {project_id}, session {session_id}"
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print("Exception occurred while deleting requirement:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/generate-epics")
def get_epics(
    project_id: str = Query(..., description="Project ID to use"),
    session_id: str = Query(None, description="Session ID to filter by (optional)")
):
    try:
        descriptions = fetch_requirement_descriptions(project_id, session_id)

        if not descriptions:
            return {"status": "error", "message": "No requirements found", "data": []}

        parsed_data = generate_and_group_epics(descriptions, project_id, session_id)

        return {
            "status": "success",
            "message": "Epics generated",
            "data": parsed_data
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}
# @router.get("/generate-user-stories")
# def get_user_stories(
#     project_id: str = Query(..., description="Project ID to use"),
#     session_id: str = Query(None, description="Session ID to filter by (optional)")
# ):
#     try:
#         question = (
#            "Extract ONLY user stories from the transcript in this format:\n\n"
#             "User Stories\n\n"
#             "• As a [role], I want to [goal], so that [reason]. speaker: [Speaker's name]\n\n"
#             "Rules: Go in depth and extract all the user stories from the transcript and make sure to include all the details and depth."
#         )
#         chain = build_retrieval_chain(project_id, session_id)
#         response = chain.invoke({"question": question})
#         user_stories = parse_user_stories(response)
#         return {
#             "status": "success",
#             "message": f"User stories extracted",
#             "data": user_stories
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e), "data": []}
