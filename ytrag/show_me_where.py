"""Show Me Where - retrieval-first interface focused on timestamps.

Instead of generating an answer, just show the user where in the lectures
the topic was explained. This is instant, free, and unlimited because
no LLM is involved.

The timestamps ARE the product.
"""

from ytrag.index import search
from ytrag.models import Chunk
from ytrag.config import TOP_K, CONFIDENT_DISTANCE


def show_where(
    question: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    include_preview: bool = True,
    confidence_threshold: float | None = None,
) -> dict:
    """Find where in the lectures a topic was explained.

    This is the primary interface - timestamps without generation.

    Args:
        question: User's question
        top_k: Number of timestamps to return
        video_id: Optional filter to specific video
        include_preview: Include text preview of each moment
        confidence_threshold: Optional confidence cutoff (None = no filter)

    Returns:
        Dict with:
        - query: The question asked
        - confident: Whether the best match is high-confidence
        - moments: List of timestamp moments with lecture links
        - total: Total moments found
    """
    from ytrag.index import title_overlap

    question = question.strip()
    if not question:
        return {
            "query": question,
            "confident": False,
            "moments": [],
            "total": 0,
        }

    # Retrieve with no LLM-specific filtering
    hits = search(question, top_k=top_k, video_id=video_id, max_distance=2.0)

    if not hits:
        return {
            "query": question,
            "confident": False,
            "moments": [],
            "total": 0,
        }

    # Check if top match is confident
    top_chunk, top_distance = hits[0]
    cutoff = confidence_threshold if confidence_threshold is not None else CONFIDENT_DISTANCE
    is_confident = (
        title_overlap(question, top_chunk.video_title) > 0
        or top_distance <= cutoff
    )

    # Build moment list
    moments = []
    for idx, (chunk, distance) in enumerate(hits, start=1):
        moment = {
            "rank": idx,
            "lecture": chunk.video_title,
            "video_id": chunk.video_id,
            "timestamp": chunk.timestamp,
            "seconds": chunk.link_sec,  # For programmatic seeking
            "url": chunk.url,
            "distance": round(distance, 4),
        }

        if include_preview:
            # Extract a short preview of the moment
            lines = chunk.text.split('\n')
            # Skip the title line if present
            content_lines = [l.strip() for l in lines[1:] if l.strip()]
            preview = ' '.join(content_lines)[:200].strip()
            moment["preview"] = preview

        moments.append(moment)

    return {
        "query": question,
        "confident": is_confident,
        "moments": moments,
        "total": len(moments),
    }


def show_where_by_topic(
    topic: str,
    top_k: int = TOP_K * 2,  # Show more by default for browsing
) -> dict:
    """Find all mentions of a topic across all lectures.

    Useful for students browsing what's available on a topic.

    Args:
        topic: Topic or keyword to search for

    Returns:
        Dict with moments organized by lecture
    """
    moments = show_where(topic, top_k=top_k)

    # Organize by lecture
    by_lecture = {}
    for moment in moments["moments"]:
        lecture_id = moment["video_id"]
        lecture_title = moment["lecture"]

        if lecture_id not in by_lecture:
            by_lecture[lecture_id] = {
                "title": lecture_title,
                "moments": []
            }

        by_lecture[lecture_id]["moments"].append(moment)

    return {
        "query": moments["query"],
        "confident": moments["confident"],
        "by_lecture": by_lecture,
        "total_lectures": len(by_lecture),
        "total_moments": moments["total"],
    }


def show_where_in_lecture(
    question: str,
    video_id: str,
    top_k: int = TOP_K,
) -> dict:
    """Find where in a specific lecture something was explained.

    Useful when watching a lecture and wondering about a specific concept.

    Args:
        question: What to find
        video_id: Which lecture to search in
        top_k: How many moments to show

    Returns:
        Dict with moments in that lecture only
    """
    return show_where(question, top_k=top_k, video_id=video_id)
