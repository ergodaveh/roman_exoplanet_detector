"""
candidate_review.py
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from roman_exoplanet_detector.domain.observation import (
    Candidate
)


class CandidateStatus(Enum):

    PENDING = "Pending"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"
    NEEDS_STUDY = "Needs Study"
    POSSIBLE_BINARY = (
        "Possible Binary Lens"
    )


@dataclass
class ReviewRecord:

    reviewer: str

    status: CandidateStatus

    comments: str

    timestamp: str


class CandidateReview:

    def __init__(
        self,
        candidate: Candidate
    ):

        self.candidate = candidate

        self.reviews = []

    def add_review(
        self,
        reviewer: str,
        status: CandidateStatus,
        comments: str
    ):

        self.reviews.append(

            ReviewRecord(

                reviewer=reviewer,

                status=status,

                comments=comments,

                timestamp=
                datetime.utcnow()
                .isoformat()

            )
        )

    def latest_status(
        self
    ) -> CandidateStatus:

        if not self.reviews:

            return CandidateStatus.PENDING

        return self.reviews[-1].status

    def save(
        self,
        filename: str
    ):

        payload = {

            "candidate":

            self.candidate.summary(),

            "reviews":

            [

                {

                    "reviewer":
                        r.reviewer,

                    "status":
                        r.status.value,

                    "comments":
                        r.comments,

                    "timestamp":
                        r.timestamp
                }

                for r
                in self.reviews

            ]
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                payload,
                f,
                indent=4
            )

    @classmethod
    def from_candidate(
        cls,
        candidate: Candidate
    ):

        return cls(candidate)