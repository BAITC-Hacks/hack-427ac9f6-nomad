"""Adapt existing regression fixtures to the two real API response schemas."""
import json
from types import SimpleNamespace
from src.services.extraction import StructureExtraction, FunctionExtraction


def staged_response(legacy):
    def respond(**kwargs):
        if kwargs["text_format"] is StructureExtraction:
            parsed = StructureExtraction(units=[
                {key: value for key, value in unit.model_dump().items() if key != "functions"}
                for unit in legacy.units
            ])
        else:
            registry = json.loads(kwargs["input"])["validated_units"]
            functions = []
            for unit in legacy.units:
                owners = [owner for owner in registry
                          if owner["canonical_name"].strip().casefold() == unit.name.strip().casefold()]
                for function in unit.functions:
                    functions.append(dict(
                        text=function.text,
                        owner_unit_id=owners[0]["id"] if owners else "unknown",
                        source_chunk_ids=list(dict.fromkeys(
                            function.source_chunk_ids + function.ownership_source_chunk_ids)),
                        ownership="ambiguous" if function.ownership == "inferred" else "direct",
                        supported_by_source=True,
                    ))
            parsed = FunctionExtraction(functions=functions)
        return SimpleNamespace(status="completed", output_parsed=parsed)
    return respond

