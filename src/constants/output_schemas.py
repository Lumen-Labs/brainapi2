"""
File: /output_schemas.py
Created Date: Wednesday October 29th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday October 29th 2025 7:50:18 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from pydantic import BaseModel, Field


class KGNeighbor(BaseModel):
    """
    Neighbor model.
    """

    uuid: str = Field(description="The UUID of the neighbor.")
    similarities: list[str] = Field(
        description="A list of string reasons that explain why the nodes are similar."
    )


class RetrieveNeighborsOutputSchema(BaseModel):
    """
    Output schema for the retrieve neighbors operation.
    """

    neighbors: list[KGNeighbor]
