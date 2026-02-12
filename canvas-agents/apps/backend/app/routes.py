from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from packages.schemas.materials import MaterialGenRequest, TeachingPacketResponse

api_bp = Blueprint("api", __name__)

@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})

@api_bp.post("/materials/generate")
def materials_generate():
    """MVP endpoint: accepts a request and returns a stub TeachingPacketResponse.

    Replace the stub call with LangGraph orchestrator:
        result = orchestrator.run(MaterialGenRequest)
    """
    try:
        payload = request.get_json(force=True, silent=False)
        req = MaterialGenRequest.model_validate(payload)
    except ValidationError as e:
        return jsonify({"error": "invalid_request", "details": e.errors()}), 400

    # TODO: call orchestrator
    resp = TeachingPacketResponse.stub(req)
    return jsonify(resp.model_dump())
